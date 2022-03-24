import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import time
import math
import random

def create_dict():
    with open('data/spotify_million_playlist_dataset_csv/data/vocabulary.csv', encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv file
        track_to_index = {}
        index_to_track = {}
        for index, row in enumerate(csv_reader):
            track_to_index[row[1]] = int(row[0])
            index_to_track[row[0]] = row[1]
        return track_to_index, index_to_track

def create_train_valid_test_data(num_rows_train, num_rows_valid, num_rows_test):
    # read training data from "track_sequences"
    # rows(track_sequences) = 1.000.000 (0 to 999.999)
    train_data = []
    valid_data = []
    test_data = []
    with open('data/spotify_million_playlist_dataset_csv/data/id_sequences.csv', encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv file
        for index, row_str in enumerate(csv_reader):
            row = [int(id) for id in row_str]
            if index >= num_rows_train + num_rows_valid + num_rows_test:
                break
            elif index < num_rows_train:
                if len(row) <= 3:
                    continue
                i = int(len(row) / 2 + 1)
                src = torch.tensor(row[0:i]).to(torch.int32)
                trg = torch.Tensor(row[i: len(row)]).to(torch.int32)
                train_data.append([src, trg])
            elif index < num_rows_train + num_rows_valid:
                if len(row) <= 3:
                    continue
                i = int(len(row) / 2 + 1)
                src = torch.Tensor(row[0:i]).to(torch.int32)
                trg = torch.Tensor(row[i: len(row)]).to(torch.int32)
                valid_data.append([src, trg])
            else:
                if len(row) <= 3:
                    continue
                i = int(len(row) / 2 + 1)
                src = torch.Tensor(row[0:i]).to(torch.int32)
                trg = torch.Tensor(row[i: len(row)]).to(torch.int32)
                test_data.append([src, trg])
    # shape of train data: (num_rows_train, 2)
    # shape of validation data: (num_rows_valid, 2)
    # shape of test data: (num_rows_test, 2)
    return np.array(train_data, dtype=object), np.array(valid_data, dtype=object), np.array(test_data, dtype=object)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src len, batch size, hid dim* n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers ' n directions, batch size, hid dim]
        # outputs are always from the top hidden layer
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]
        #input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]
        print(embedded.shape)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        #batch_size = len(src)
        #trg_len = len(trg)
        batch_size = 1
        trg_len = 1
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        hidden = hidden.unsqueeze(0)    # hidden.shape == (1, 2, 50)
        cell = cell.unsqueeze(0)
        # first input to the decoder is the <sos> tokens
        #input = trg[1, :]
        input = trg
        for t in range(0, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            # place predictions in a tensor holding predictions for each token
            print(output.shape)
            print(outputs[t].shape)
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
        return outputs


def train(model, train_data, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    num_iterations = 0
    for i in range(0, len(train_data), BATCH_SIZE):
        num_iterations += 1
        print(num_iterations)
        src = torch.Tensor(train_data[i:BATCH_SIZE, 0])
        trg = train_data[i:BATCH_SIZE, 1]
        optimizer.zero_grad()
        output = model(src, trg)
        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        output = output[0:].view(-1, output_dim)
        trg = trg[0:].view(-1)
        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_data)


if __name__ == '__main__':
    print("load train-, validation-, and test-data...")
    train_data, valid_data, test_data = create_train_valid_test_data(10000, 100, 100)
    track_to_index, index_to_track = create_dict()
    print("finish")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 1

    INPUT_DIM = 10000 #len(track_to_index)
    OUTPUT_DIM = 10000 # len(track_to_index)
    ENC_EMB_DIM = 100
    DEC_EMB_DIM = 100
    HID_DIM = 50
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    print("create model...")
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    print("finish")

    print("init weights...")
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
    model.apply(init_weights)
    print("finish")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss()   # maybe add ignore_index = ...

    # train(model, train_data, optimizer, criterion, 1)
    src = torch.IntTensor([1, 2, 3, 4])
    print(type(src))
    trg = torch.IntTensor([5, 6, 7, 8])
    model(src, trg)
