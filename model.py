import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import sys
torch.set_printoptions(threshold=10000)

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())


def gradient_hook(grad, *args):
    for arg in args:
        print(arg)
    print(grad)


def to_variable(tensor, requires_grad=False):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, requires_grad=requires_grad)


class Listener(nn.Module):  # combines RNN and Squash layers. Output hidden states.
    def __init__(self, embedding_dimension, hidden_dimension, n_layers, bidirectional=True):
        super(Listener, self).__init__()

        self.hidden_size = hidden_dimension
        self.embedding_dim = embedding_dimension
        self.n_layers = n_layers
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size=embedding_dimension, hidden_size=hidden_dimension, bidirectional=bidirectional)])
        for i in range(self.n_layers - 1):
            self.lstms.append(nn.LSTM(input_size=hidden_dimension, hidden_size=hidden_dimension, bidirectional=bidirectional))

        self.linear_key = nn.Linear(in_features=hidden_dimension, out_features=hidden_dimension)
        self.linear_values = nn.Linear(in_features=hidden_dimension, out_features=hidden_dimension)

    def forward(self, input, input_len):
        #  run inputs through batch rnn..
        # x  = input (T  x N x H) where H = #mfcc
        # seq len: (N,)

        h = input
        for i, lstm in enumerate(self.lstms):
            if i > 0:
                # After first lstm layer, pBiLSTM
                seq_len = h.size(0)
                if seq_len % 2 == 0:
                    h = h.permute(1, 0, 2).contiguous()
                    h = h.view(h.size(0), h.size(1) // 2, 2, h.size(2)).sum(2) / 2
                    h = h.permute(1, 0, 2).contiguous()
                    input_len /= 2
                else:
                    print("Odd seq len should not occur!!")
                    exit()
            # First BiLSTM
            packed_h = nn.utils.rnn.pack_padded_sequence(h, input_len)
            h, _ = lstm(packed_h)
            h, _ = nn.utils.rnn.pad_packed_sequence(h)      # seq_len * bs * (2 * hidden_dim)
            #h = self.locked_dropout(h, 0.3)
            # Summing forward and backward representation
            h = h.view(h.size(0), h.size(1), 2, -1).sum(2) / 2       # h = ( h_forward + h_backward ) / 2

        keys = self.linear_key(h)           # bs * seq_len/8 * 256
        values = self.linear_values(h)      # bs * seq_len/8 * 256

        return keys, values

    def __repr__(self):
        tempstr = ''
        return tempstr


class MyLSTM(nn.LSTMCell):
    def __init__(self, input_size, hidden_size):
        super(MyLSTM, self).__init__(input_size, hidden_size)
        self.h0 = nn.Parameter(torch.randn(1, hidden_size).type(torch.FloatTensor), requires_grad=True)
        self.c0 = nn.Parameter(torch.randn(1, hidden_size).type(torch.FloatTensor), requires_grad=True)

    def forward(self, x, h):
        return super(MyLSTM, self).forward(x, h)


class Speller(nn.Module):
    def __init__(self, output_classes, hidden_dimension, embedding_dimension, n_layers=1, max_decoding_length):
        super(Speller, self).__init__()

        self.vocab = output_classes
        self.hidden_size = hidden_dimension
        self.embedding_dim = embedding_dimension
        # self.is_stochastic = params.is_stochastic
        self.max_decoding_length = max_decoding_length
        self.embed = nn.Embedding(num_embeddings=output_classes, embedding_dim=self.hidden_size)
        self.n_layers = n_layers  # TODO: make adaptable

        self.lstm_cells = nn.ModuleList([
            MyLSTM(input_size=2 * self.hidden_size, hidden_size=2 * self.hidden_size),
            MyLSTM(input_size=2 * self.hidden_size, hidden_size=2 * self.hidden_size),
            MyLSTM(input_size=2 * self.hidden_size, hidden_size=self.hidden_size)
        ])

        # For attention
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)

        # For character projection
        self.projection_layer1 = nn.Linear(in_features=2 * self.hidden_size, out_features=self.hidden_size)
        self.non_linear = nn.LeakyReLU()
        self.projection_layer2 = nn.Linear(in_features=self.hidden_size, out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, keys, values, label, label_len, input_len):
        # Number of characters in the transcript
        embed = self.embed(label)          # bs * label_len * 256
        output = None
        hidden_states = []
        # Initial context
        query = self.linear(self.lstm_cells[2].h0.expand(embed.size(0), -1).contiguous())  # bs * 256, This is the query
        attn = torch.bmm(query.unsqueeze(1), keys.permute(1, 2, 0))  # bs * 1 * seq_len/8

        # Create mask
        a = torch.arange(input_len[0]).unsqueeze(0).expand(len(input_len), -1)
        b = input_len.unsqueeze(1).float()
        mask = a < b
        if torch.cuda.is_available():
            mask = torch.autograd.Variable(mask.unsqueeze(1).type(torch.FloatTensor)).cuda()
        else:
            mask = torch.autograd.Variable(mask.unsqueeze(1).type(torch.FloatTensor))
        #attn.data.masked_fill_((1 - mask).unsqueeze(1), -float('inf'))
        attn = F.softmax(attn, dim=2)
        attn = attn * mask
        attn = attn / attn.sum(2).unsqueeze(2)
        context = torch.bmm(attn, values.permute(1, 0, 2)).squeeze(1)

        for i in range(label_len.max() - 1):
            h = embed[:, i, :]                                  # bs * 256
            h = torch.cat((h, context), dim=1)                  # bs * 512
            for j, lstm in enumerate(self.lstm_cells):
                if i == 0:
                    h_x_0, c_x_0 = lstm(h, lstm.h0.expand(embed.size(0), -1).contiguous(),
                                        lstm.c0.expand(embed.size(0), -1).contiguous())       # bs * 512
                    hidden_states.append((h_x_0, c_x_0))
                else:
                    h_x_0, c_x_0 = hidden_states[j]
                    hidden_states[j] = lstm(h, h_x_0, c_x_0)
                h = hidden_states[j][0]

            query = self.linear(h)              # bs * 2048, This is the query
            attn = torch.bmm(query.unsqueeze(1), keys.permute(1, 2, 0))         # bs * 1 * seq_len/8
            #attn.data.masked_fill_((1 - mask).unsqueeze(1), -float('inf'))
            attn = F.softmax(attn, dim=2)
            attn = attn * mask
            attn = attn / attn.sum(2).unsqueeze(2)
            context = torch.bmm(attn, values.permute(1, 0, 2)).squeeze(1)       # bs * 256
            h = torch.cat((h, context), dim=1)

            # At this point, h is the embed from the 2 lstm cells. Passing it through the projection layers
            h = self.projection_layer1(h)
            h = self.non_linear(h)
            h = self.softmax(self.projection_layer2(h))
            # Accumulating the output at each timestep
            if output is None:
                output = h.unsqueeze(1)
            else:
                output = torch.cat((output, h.unsqueeze(1)), dim=1)
        return output                         # bs * max_label_seq_len * 33

    def decode(self, keys, values):
        """
        :param keys:
        :param values:
        :return: Returns the best decoded sentence
        """
        bs = 1  # batch_size for decoding
        output = []
        raw_preds = []

        for _ in range(100):
            hidden_states = []
            raw_pred = None
            raw_out = []
            # Initial context
            query = self.linear(self.lstm_cells[2].h0)  # bs * 256, This is the query
            attn = torch.bmm(query.unsqueeze(1), keys.permute(1, 2, 0))  # bs * 1 * seq_len/8
            attn = F.softmax(attn, dim=2)
            context = torch.bmm(attn, values.permute(1, 0, 2)).squeeze(1)

            h = self.embed(to_variable(torch.zeros(bs).long()))  # Start token provided for generating the sentence
            for i in range(self.max_decoding_length):
                h = torch.cat((h, context), dim=1)
                for j, lstm in enumerate(self.lstm_cells):
                    if i == 0:
                        h_x_0, c_x_0 = lstm(h, lstm.h0,
                                            lstm.c0)  # bs * 512
                        hidden_states.append((h_x_0, c_x_0))
                    else:
                        h_x_0, c_x_0 = hidden_states[j]
                        hidden_states[j] = lstm(h, h_x_0, c_x_0)
                    h = hidden_states[j][0]

                query = self.linear(h)  # bs * 256, This is the query
                attn = torch.bmm(query.unsqueeze(1), keys.permute(1, 2, 0))  # bs * 1 * seq_len/8
                # attn.data.masked_fill_((1 - mask).unsqueeze(1), -float('inf'))
                attn = F.softmax(attn, dim=2)
                context = torch.bmm(attn, values.permute(1, 0, 2)).squeeze(1)  # bs * 256
                h = torch.cat((h, context), dim=1)

                # At this point, h is the embed from the 2 lstm cells. Passing it through the projection layers
                h = self.projection_layer1(h)
                h = self.non_linear(h)
                h = self.projection_layer2(h)
                lsm = self.softmax(h)
                # if self.is_stochastic > 0:
                #     gumbel = torch.autograd.Variable(self.sample_gumbel(shape=h.size(), out=h.data.new()))
                #     h += gumbel
                # TODO: Do beam search later

                h = torch.max(h, dim=1)[1]
                raw_out.append(h.data.cpu().numpy()[0])
                if raw_pred is None:
                    raw_pred = lsm
                else:
                    raw_pred = torch.cat((raw_pred, lsm), dim=0)

                if h.data.cpu().numpy() == 0:
                    break

                # Primer for next character generation
                h = self.embed(h)
            output.append(raw_out)
            raw_preds.append(raw_pred)
        return output, raw_preds

    def __repr__(self):
        tempstr = ''
        return tempstr


class Las(nn.Module):
    def __init__(self, audio_conf,
                 labels="abc",
                 hidden_dimension, embedding_dimension, n_layers_enc, n_layers_dec, output_size):

        super(Las, self).__init__()
        self._rnn_type = rnn_type  # string
        self._audio_conf = audio_conf or {}  # dict
        self._labels = labels  # string

        self.hidden_dimension = hidden_dimension
        self.embedding_dimension = embedding_dimension
        self.n_layers_enc = n_layers_enc
        self.n_layers_dec = n_layers_dec
        self.output_size = output_size

       # default assuming feature_type is 'log_spect'
        self._listener_input_dim = int(math.floor((sample_rate * window_size) / 2) + 1)
        if self._audio_conf['feature_type'] == 'log_mel_spect':
            self._listener_input_dim = 40

        elif self._audio_conf['feature_type'] == 'mfcc':
            self._listener_input_dim = 20

        self.encoder = Listener()

        self.decoder = Speller()

    def forward(self, input, input_len, label=None, label_len=None, max_label_len=50):
        '''
        x: inputs (TxNxH)  H = #mfcc
        grund_truth: onehot labels ()
        seq_lens: input seq lens ()

        '''
        keys, values = self.encoder(input, input_len)
        if label is None:
            # During decoding of test data
            return self.decoder.decode(keys, values)
        else:
            # During training
            return self.decoder(keys, values, label, label_len, input_len)

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)

        model = cls(listener_hidden_size=package['encoder_hidden_size'],
                    nb_listener_layers=package['encoder_num_layers'],
                    listener_stacking_mode=package['encoder_stacking_mode'],
                    listener_stacking_degree=package['encoder_stacking_degree'],
                    listener_is_pyramidal=package['encoder_is_pyramidal'],
                    listener_is_bidirectional=package.get('encoder_is_bidirectional', True),
                    batch_norm=package['batch_norm'],
                    speller_hidden_size=package['decoder_hidden_size'],
                    nb_speller_layers=package['decoder_num_layers'],
                    labels=package['labels'],
                    audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[package['rnn_type']],
                    )

        model.load_state_dict(package['state_dict'], strict=False)
        for k in model.state_dict().keys():
            if k not in package['state_dict']:
                print("%s not in the saved model. Loading rest of the model anyway" % k)

        for x in model.encoder.rnns:
            x.flatten_parameters()
        return model

    @classmethod
    def load_model_package(cls, package):
        model = cls(listener_hidden_size=package['encoder_hidden_size'],
                    nb_listener_layers=package['encoder_num_layers'],
                    listener_stacking_mode=package['encoder_stacking_mode'],
                    listener_stacking_degree=package['encoder_stacking_degree'],
                    listener_is_pyramidal=package['encoder_is_pyrmidal'],
                    listener_is_bidirectional=package.get('encoder_is_bidirectional', True),
                    batch_norm=package['batch_norm'],
                    speller_hidden_size=package['decoder_hidden_size'],
                    nb_speller_layers=package['encoder_num_layers'],
                    labels=package['labels'],
                    audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[package['rnn_type']],
                    bidirectional=package.get('encoder_is_bidirectional', True))

        model.load_state_dict(package['state_dict'])
        return model

    def __str__(self):
        print("Encoder Properties")
        print("  RNN Type:         ", self._rnn_type.__name__.lower())
        print("  Encoder RNN Layers:       ", self.encoder_num_layers)
        print("  Encoder RNN Size:         ", self.encoder_hidden_size)
        print("  Encoder RNN Pyramidal?:         ", self.encoder_is_pyramidal)
        print("  Encoder RNN Bidirectional?:         ", self.encoder_is_bidirectional)
        print("  Encoder RNN Pyramidal stacking mode:         ", self.encoder_stacking_mode)
        print("  Encoder RNN Pyramidal stacking degree:         ", self.encoder_stacking_degree)
        print("Decoder Properties")
        print("  Decoder RNN Layers:       ", self.decoder_num_layers)
        print("  Decoder RNN Size:         ", self.decoder_hidden_size)
        print("  # Output Classes:          ", len(self._labels))
        print("")
        print(" Input Features")
        print("  Labels:           ", self._labels)
        print("  Sample Rate:      ", self._audio_conf.get("sample_rate", "n/a"))
        print("  Window Type:      ", self._audio_conf.get("window", "n/a"))
        print("  Window Size:      ", self._audio_conf.get("window_size", "n/a"))
        print("  Window Stride:    ", self._audio_conf.get("window_stride", "n/a"))
        print(" Input Feature Type:    ", self._audio_conf.get("feature_type", "n/a"))
        return " "

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
                  cer_results=None, wer_results=None, avg_loss=None, meta=None):

        package = {
            'version': model._version,
            'encoder_hidden_size': model.encoder_hidden_size,
            'encoder_num_layers': model.encoder_num_layers,
            'decoder_hidden_size': model.decoder_hidden_size,
            'decoder_num_layers': model.decoder_num_layers,
            'rnn_type': supported_rnns_inv.get(model._rnn_type, model._rnn_type.__name__.lower()),
            'audio_conf': model._audio_conf,
            'labels': model._labels,
            'state_dict': model.state_dict(),
            'encoder_stacking_mode': model.encoder_stacking_mode,
            'encoder_stacking_degree': model.encoder_stacking_degree,
            'encoder_is_pyramidal': model.encoder_is_pyramidal,
            'encoder_is_bidirectional': model.encoder_is_bidirectional,
            'batch_norm': model.batch_norm
        }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if avg_loss is not None:
            package['avg_loss'] = avg_loss
        if epoch is not None:
            package['epoch'] = epoch + 1  # increment for readability
        if iteration is not None:
            package['iteration'] = iteration
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results
        if meta is not None:
            package['meta'] = meta
        return package

    @staticmethod
    def get_labels(model):
        return model._labels

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params

    @staticmethod
    def get_audio_conf(model):
        return model._audio_conf

    @staticmethod
    def get_meta(model):
        m = model
        meta = {
            "version": m._version,
            "encoder_hidden_size": m.encoder_hidden_size,
            "encoder_hidden_layers": m.encoder_num_layers,
            "decoder_hidden_size": m.decoder_hidden_size,
            "decoder_hidden_layers": m.decoder_num_layers,
            "rnn_type": supported_rnns_inv[m._rnn_type]
        }
        return meta

    def __repr__(self):
        tempstr = ''
        return tempstr


if __name__ == '__main__':
    import os.path
    import argparse

    parser = argparse.ArgumentParser(description='LAS model information')
    parser.add_argument('--model-path', default='models/las_final.pth',
                        help='Path to model file created by training')
    args = parser.parse_args()
    package = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model = Las.load_model(args.model_path)

    print("Model name:         ", os.path.basename(args.model_path))
    print("LAS version: ", model._version)
    print("")
    print("Encoder Properties")
    print("  RNN Type:         ", model._rnn_type.__name__.lower())
    print("  Encoder RNN Layers:       ", model.encoder_num_layers)
    print("  Encoder RNN Size:         ", model.encoder_hidden_size)
    print("  Encoder RNN Pyramidal?:         ", model.encoder_is_pyrmidal)
    print("  Encoder RNN Bidirectional?:         ", model.encoder_is_bidirectional)
    print("  Encoder RNN Pyramidal stacking mode:         ", model.encoder_stacking_mode)
    print("  Encoder RNN Pyramidal stacking degree:         ", model.encoder_stacking_degree)
    print("Decoder Properties")
    print("  Decoder RNN Layers:       ", model.decoder_num_layers)
    print("  Decoder RNN Size:         ", model.decoder_hidden_size)
    print("  # Output Classes:          ", len(model._labels))
    print("")
    print("Model Features")
    print("  Labels:           ", model._labels)
    print("  Sample Rate:      ", model._audio_conf.get("sample_rate", "n/a"))
    print("  Window Type:      ", model._audio_conf.get("window", "n/a"))
    print("  Window Size:      ", model._audio_conf.get("window_size", "n/a"))
    print("  Window Stride:    ", model._audio_conf.get("window_stride", "n/a"))
    print(" Input Feature Type:    ", model._audio_conf.get("feature_type", "n/a"))

    if package.get('loss_results', None) is not None:
        print("")
        print("Training Information")
        epochs = package['epoch']
        print("  Epochs:           ", epochs)
        print("  Current Loss:      {0:.3f}".format(package['loss_results'][epochs - 1]))
        print("  Current CER:       {0:.3f}".format(package['cer_results'][epochs - 1]))
        print("  Current WER:       {0:.3f}".format(package['wer_results'][epochs - 1]))

    if package.get('meta', None) is not None:
        print("")
        print("Additional Metadata")
        for k, v in model._meta:
            print("  ", k, ": ", v)
