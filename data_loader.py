import os
import subprocess
from tempfile import NamedTemporaryFile

from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data.sampler import Sampler

import librosa
import numpy as np
import scipy.signal
import torch
import torchaudio
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}


def load_audio(path):
    sound, _ = torchaudio.load(path, normalization=True)
    sound = sound.numpy()
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound


class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError


class NoiseInjection(object):
    def __init__(self,
                 path=None,
                 sample_rate=16000,
                 noise_levels=(0, 0.5)):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """
        if not os.path.exists(path):
            print("Directory doesn't exist: {}".format(path))
            raise IOError
        self.paths = path is not None and librosa.util.find_files(path)
        self.sample_rate = sample_rate
        self.noise_levels = noise_levels

    def inject_noise(self, data):
        noise_path = np.random.choice(self.paths)
        noise_level = np.random.uniform(*self.noise_levels)
        return self.inject_noise_sample(data, noise_path, noise_level)

    def inject_noise_sample(self, data, noise_path, noise_level):
        noise_len = get_audio_length(noise_path)
        data_len = len(data) / self.sample_rate
        noise_start = np.random.rand() * (noise_len - data_len)
        noise_end = noise_start + data_len
        noise_dst = audio_with_sox(noise_path, self.sample_rate, noise_start, noise_end)
        assert len(data) == len(noise_dst)
        noise_energy = np.sqrt(noise_dst.dot(noise_dst) / noise_dst.size)
        data_energy = np.sqrt(data.dot(data) / data.size)
        data += noise_level * noise_dst * data_energy / noise_energy
        return data


class SpectrogramParser(AudioParser):
    def __init__(self, audio_conf, normalize=False, augment=False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        super(SpectrogramParser, self).__init__()
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.normalize = normalize
        self.augment = augment
        self.fmin = audio_conf['fmin']  # ADD
        self.fmax = audio_conf['fmax']  # ADD
        self.n_mels = audio_conf['nmels']  # ADD
        self.feature = audio_conf['feature_type']  # ADD  'log_mel_spect', log_spect

        self.noiseInjector = NoiseInjection(audio_conf['noise_dir'], self.sample_rate,
                                            audio_conf['noise_levels']) if audio_conf.get(
            'noise_dir') is not None else None
        self.noise_prob = audio_conf.get('noise_prob')

    def parse_audio(self, audio_path):
        if self.augment:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate)
        else:
            y = load_audio(audio_path)
        if self.noiseInjector:
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                y = self.noiseInjector.inject_noise(y)
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)

        # compute feature and repeat last frame so that seq_len is a multiple of 8,
        # since theere are 3 bidirectional pyramidal listener layers

        if self.feature == 'log_spect':
            # STFT
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                             win_length=win_length, window=self.window)
            spect, phase = librosa.magphase(D)
            # S = log(S+1)
            spect = np.log1p(spect)  # shape (1 + nfft/2, t)

        elif self.feature == 'log_mel_spect':  # ADD

            # mel_basis = librosa.filters.mel(sr=self.sample_rate,
            #                                 n_fft=n_fft,
            #                                 n_mels=self.n_mels,
            #                                 fmin=self.fmin,
            #                                 fmax=self.fmax
            #                                 )
            # eps = np.spacing(1)
            # magspec = np.abs(librosa.stft(y + eps,
            #                               n_fft=n_fft,
            #                               win_length=win_length,
            #                               hop_length=hop_length
            #                               ))

            # powerspec = magspec ** 2

            # spect = np.dot(mel_basis, powerspec)  # shape = (n_mels, t)
            spect = librosa.feature.melspectrogram(y, sr=self.sample_rate, nmels=40, n_fft=n_fft, hop_length=hop_length)
            #(40,t)

        elif self.feature == 'mfcc':
            eps = np.spacing(1)
            spect = librosa.feature.mfcc(y + eps, sr=self.sample_rate)  # (20,t)

        time_steps = spect.shape[1]
        pad_steps = 8 - (time_steps % 8)

        spect = np.hstack((spect, np.tile(spect[:, [-1]], pad_steps)))

        spect = torch.FloatTensor(spect)

        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError


class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, audio_conf, manifest_filepath, labels, normalize=False, augment=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:
        /path/to/audio.wav,/path/to/audio.txt
        ...
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        super(SpectrogramDataset, self).__init__(audio_conf, normalize, augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]
        spect = self.parse_audio(audio_path)
        transcript, original_text = self.parse_transcript(transcript_path)
        return spect, transcript, original_text

    def parse_transcript(self, transcript_path):
        '''
        '*'  -->  <PAD> token  --> index 0
        '#' -->  <SOS> token  --> index 1
        '$' -->   <EOS> token  --> index 2
        '''
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        original_str = transcript
        assert isinstance(original_str, str)
        transcript = '#' + transcript + '$'
        transcript = list(filter(None, [float(self.labels_map.get(x)) for x in list(transcript)]))
        #print(transcript_path, original_str)
        return transcript, original_str

    def __len__(self):
        return self.size


def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    vocab_dim = 31
    minibatch_size = len(batch)
    original_text = []
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    max_label_len_sample = max(batch, key=lambda sample: len(sample[1]))
    # the above is a tuple(spect,transcript,original_text)
    max_label_len = len(max_label_len_sample[1])
    labels = torch.zeros([minibatch_size, max_label_len, vocab_dim], dtype=torch.float)  # (N,L,V)

    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)

    max_seqlength = longest_sample.size(1)
    freq_size = longest_sample.size(0)  # n_mfcc or freq bands
    inputs = torch.zeros(minibatch_size, freq_size, max_seqlength)  # (max_seq_len, batch, input_size)
    seq_lens = torch.IntTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = torch.zeros([minibatch_size, max_label_len - 1], dtype=torch.long)

    for x in range(minibatch_size):

        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        original_text.append(sample[2])
        target_sizes[x] = len(target)  # actual len of target without <pad> (but with <sos> and <eos> included)

        target += [0] * (max_label_len - len(target))  # pad with <pad> token
        indices = torch.tensor(target).unsqueeze(0).view(-1, 1).type(torch.LongTensor)

        targets[x] = torch.Tensor(target[1:])  # remove sos

        labels[x].scatter_(1, indices, 1.0)

        seq_length = int(tensor.size(1))
        inputs[x].narrow(1, 0, seq_length).copy_(tensor)
        seq_lens[x] = seq_length
        # targets.extend(target)
    #targets = torch.IntTensor(targets)
    inputs = inputs.transpose(0, 2).transpose(1, 2).contiguous()  # (TxNxH)
    labels = labels.transpose(0, 1)  # (LxNxV)
    targets = targets.view(-1).contiguous()  # (N*(L-1))
    return inputs, labels, seq_lens, target_sizes, original_text, targets


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)


class DistributedBucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1, num_replicas=None, rank=None):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(DistributedBucketingSampler, self).__init__(data_source)
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.data_source = data_source
        self.ids = list(range(0, len(data_source)))
        self.batch_size = batch_size
        self.bins = [self.ids[i:i + batch_size] for i in range(0, len(self.ids), batch_size)]
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.bins) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        offset = self.rank
        # add extra samples to make it evenly divisible
        bins = self.bins + self.bins[:(self.total_size - len(self.bins))]
        assert len(bins) == self.total_size
        samples = bins[offset::self.num_replicas]  # Get every Nth bin, starting from rank
        return iter(samples)

    def __len__(self):
        return self.num_samples

    def shuffle(self, epoch):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(epoch)
        bin_ids = list(torch.randperm(len(self.bins), generator=g))
        self.bins = [self.bins[i] for i in bin_ids]


def get_audio_length(path):
    output = subprocess.check_output(['soxi -D \"%s\"' % path.strip()], shell=True)
    return float(output)


def audio_with_sox(path, sample_rate, start_time, end_time):
    """
    crop and resample the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1".format(path, sample_rate,
                                                                                               tar_filename, start_time,
                                                                                               end_time)
        os.system(sox_params)
        y = load_audio(tar_filename)
        return y


def augment_audio_with_sox(path, sample_rate, tempo, gain):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                                      augmented_filename,
                                                                                      " ".join(sox_augment_params))
        os.system(sox_params)
        y = load_audio(augmented_filename)
        return y


def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15),
                                  gain_range=(-6, 8)):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value)
    return audio
