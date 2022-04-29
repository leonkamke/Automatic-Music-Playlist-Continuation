import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import csv
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import gensim


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, pre_trained_embedding, hid_dim, n_layers, dropout=0):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        # input shape of embedding: (*) containing the indices
        # output shape of embedding: (*, embed_dim == 100)
        self.embedding = pre_trained_embedding
        # input shape of LSTM has to be (batch_size, seq_len, embed_dim == 100) when batch_first=True
        # output shape of LSTM: output.shape == (batch_size, seq_len, hid_dim)  when batch_first=True
        #                       h_n.shape == (n_layers, batch_size, hid_dim)
        #                       c_n.shape == (n_layers, batch_size, hid_dim)
        self.rnn = nn.LSTM(100, hid_dim, n_layers, batch_first=True, dropout=dropout)
        # input shape of Linear: (*, hid_dim)
        # output shape of Linear: (*, vocab_size)
        self.fc_out = nn.Linear(hid_dim, vocab_size)

    def forward(self, input):
        # input.shape == (batch_size, seq_len)
        x = self.embedding(input)
        # x.shape == (batch_size, seq_len, embed_dim == 100), when batch_first=True
        x, (h_n, c_n) = self.rnn(x)
        # x.shape == (batch_size, seq_len, hid_dim), when batch_first=True
        # x = self.fc_out(x)
        # returned_matrix.shape == (batch_size, seq_len, vocab_size)
        return self.fc_out(x)


if __name__ == '__main__':
    """# test_batch.shape == (2, 3, 3)
    src = torch.FloatTensor([[[0.5, 0.5, 0.5],
                                     [0.3, 0.3, 0.3],
                                     [0.4, 0.4, 0.4]],
                                    [[0.1, 0.1, 0.1],
                                    [0.2, 0.2, 0.2],
                                   [0.9, 0.9, 0.9]]])

    rnn = nn.LSTM(3, 2, 1, batch_first=True)
    output, (hn, cn) = rnn(src)
    # output.shape == (2, 3, 2)
    fc_out = nn.Linear(2, 10)
    output = fc_out(output)
    # output.shape == (2, 3, 10)
    print(output.shape)"""

    """# IMPORTANT!!! This is how Cross entropy works!!!
    # batch_size = 2; trg_len = 3, output_dim = 6
    trg = torch.LongTensor([4, 2, 2, 0, -1, 1])
    output = torch.FloatTensor([[0, 0, 0, 0, 1, 0],
                                [1, 1, 10, 1, 1, 1],
                                [10, 10, 20, 10, 10, 10],
                                [1, -1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 10],
                                [0, 1, 0, 0, 0, 0]])
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    loss = criterion(output, trg)
    print(loss.item())"""

    """from torch.nn.utils.rnn import pad_sequence
    a = torch.LongTensor([1, 2])
    b = torch.LongTensor([1, 10, 4, 0])
    c = torch.LongTensor([1, 2, 3, 4, 3, 4, -1, -1])
    out = pad_sequence([a, b, c], batch_first=True)
    print(out.size())
    print(out)"""

    """model = gensim.models.Word2Vec.load("./models/gensim_word2vec/100_thousand_playlists/word2vec-song-vectors.model")
    model.wv.get_index('spotify:track:2xvWdq0XdHYATHsD3i5QMq')"""

    x = [[[1, 2, 3],
          [2, 5, 2],
          [3, 4, 400],
          [0, -1, 1]]]
    x = torch.tensor(x)
    print(x.argmax(dim=2)[0])

