import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import time
import math
import random


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
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
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
        embedded = self.dropout(self.embedding(input))
        # increase dimension of embedded from 1 to 2
        embedded = torch.unsqueeze(embedded, 0)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = 0.9
        self.training = True
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg):
        hidden, cell = self.encoder(src)
        # hidden.shape == cell.shape == (lstm_num_layers, lstm_hidden_size)
        input = src[-1]
        predictions = torch.zeros(len(trg), self.decoder.output_dim)
        for t in range(0, len(trg)):
            pred, hidden, cell = self.decoder(input, hidden, cell)
            # pred.shape = (1, vocab_size)
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < self.teacher_forcing_ratio
            # get the highest predicted token from our predictions
            best_candidate = pred[0].argmax()
            # best_candidate.shape == scalar
            predictions[t] = pred[0]
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force and self.training else best_candidate
        # predictions.shape == (len(trg), vocab_size)
        return predictions


def train(model, train_data, optimizer, criterion, batch_size=1, clip=1, epochs=6):
    model.train()
    epoch_loss = 0
    num_iterations = 0
    for x in range(0, epochs):
        for i in range(0, len(train_data), batch_size):
            num_iterations += 1
            print(num_iterations)
            src = train_data[i, 0]
            trg = train_data[i, 1]
            optimizer.zero_grad()
            output = model(src, trg)
            # output.shape(1, vocab_size)
            trg = trg.type(torch.LongTensor)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            print("loss.item: ", loss.item())
            epoch_loss += loss.item()
    return epoch_loss / (len(train_data)*epochs)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 1
    vocab_size = 500  # len(vocabulary)
    EMB_DIM = 50
    HID_DIM = 25
    N_LAYERS = 5
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    # training parameters
    num_playlists = 6
    num_epochs = 5
    # print options
    torch.set_printoptions(profile="full")

    print("create model...")
    enc = Encoder(vocab_size, EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(vocab_size, EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
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

    '''# test model on single sequence as input
    src = torch.IntTensor([1, 2, 3, 4])
    trg = torch.IntTensor([5, 6, 7, 8])
    prediction = model(src, trg)
    print(prediction)'''

    optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss()  # maybe add ignore_index = ...

    # read data from csv file
    train_data, test_data, valid_data = create_train_valid_test_data(num_playlists, 10, 10)

    x = train(model, train_data, optimizer, criterion, epochs=num_epochs)
    model.training = False
    print(x)
    output = model(torch.tensor([4, 5, 6, 7, 8]), torch.tensor([9, 10, 11, 12]))
    print(output.shape)
    print("sequence prediction: ", torch.argmax(output, dim=1))
    print("output vector: ", output[0])
