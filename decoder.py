import Levenshtein as Lev
import torch
from six.moves import xrange
import re


def filter_str(line):
    index_of_eos = line.find('$')
    if index_of_eos == -1:
        return line
    line = line[:index_of_eos] #truncate to eos
    line = re.sub('[#*]', '', line) #clean up the rest of text
    # print("returning from re", line)
    return line


class LasDecoder(object):
    """
    Arguments:
        labels (string): mapping from integers to characters.
        pad_index (int, optional): index for the blank '_' character. Defaults to 0.
        space_index (int, optional): index for the space ' ' character. Defaults to 30.
    """

    def __init__(self, labels, pad_index=0):
        # e.g. labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ#"
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        self.pad_index = pad_index
        self.vocab_dim = len(labels)
        space_index = len(labels)  # To prevent errors in decode, we add an out of bounds index for the space
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index

    def wer(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))

    def cer(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        return Lev.distance(s1, s2)

    def batch_decode(self, probs):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription
        Arguments:
            probs: Tensor of character probabilities, #(N x max_step-1 x num_classes) = (NxL-1xV)

        Returns:
            string: list of sequences of the model's best guess for the transcription
        """
        # print(probs.size(-1))
        # print(self.vocab_dim)
        assert probs.size(-1) == self.vocab_dim

        indices = torch.argmax(probs, dim=-1)  # argmax across D dimension: indices: (NxL-1)
        # print("indices (N x L-1): ", indices)

        batch_transcript = []
        for i in range(indices.size(0)):
            # print(list(indices[i].cpu().numpy()))
            l = [self.int_to_char[j] for j in list(indices[i].cpu().numpy())]
            # print(l)
            decoded_str = ''.join(l)
            # print("decoded_str", decoded_str)
            # if i == 0 or i == 1:
            #     print("sample decoded string: %s" % decoded_str)
            batch_transcript.append(decoded_str)

        # filter out '*' (<Pad>), '#'<sos>, '$' <eos> tokens

        '''
        1. locate first occurence of <eos> '$' token:
        2. Neglect everything after it
        3. filter out <pad> '*'  and <sos>'#' tokens before it
        '''

        batch_transcript = [filter_str(transc) for transc in batch_transcript]
        # print("batch_transcript", batch_transcript)

        return batch_transcript
