import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import random
import gensim
import os


def get_word2vec_model():
    print("load word2vec from file")
    model = gensim.models.Word2Vec.load("./models/gensim_word2vec/100_thousand_playlists/word2vec-song-vectors.model")
    print("word2vec loaded from file")
    return model


def create_train_valid_test_data(num_rows_train, num_rows_valid, num_rows_test, word2vec):
    # read training data from "track_sequences"
    train_data = []
    valid_data = []
    test_data = []
    with open('data/spotify_million_playlist_dataset_csv/data/track_sequences.csv', encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv file
        for index, row in enumerate(csv_reader):
            if index >= num_rows_train + num_rows_valid + num_rows_test:
                break
            elif index < num_rows_train:
                if len(row) <= 3:
                    continue
                i = int(len(row) / 2 + 1)
                i = 4
                src = row[2:i]
                trg = row[i: len(row)]
                train_data.append([src, trg])
            elif index < num_rows_train + num_rows_valid:
                if len(row) <= 3:
                    continue
                i = int(len(row) / 2 + 1)
                i = 4
                src = row[2:i]
                trg = row[i: len(row)]
                valid_data.append([src, trg])
            else:
                if len(row) <= 3:
                    continue
                i = int(len(row) / 2 + 1)
                i = 4
                src = row[2:i]
                trg = row[i: len(row)]
                test_data.append([src, trg])
    # shape of train data: (num_rows_train, 2)
    # shape of validation data: (num_rows_valid, 2)
    # shape of test data: (num_rows_test, 2)
    for idx, row in enumerate(train_data):
        src = []
        trg = []
        for track_uri in row[0]:
            src.append(word2vec.wv.get_index(track_uri))
        for track_uri in row[1]:
            trg.append(word2vec.wv.get_index(track_uri))
        train_data[idx] = [torch.IntTensor(src), torch.IntTensor(trg)]
    for idx, row in enumerate(valid_data):
        src = []
        trg = []
        for track_uri in row[0]:
            src.append(word2vec.wv.get_index(track_uri))
        for track_uri in row[1]:
            trg.append(word2vec.wv.get_index(track_uri))
        valid_data[idx] = [torch.Tensor(src), torch.Tensor(trg)]
    for idx, row in enumerate(test_data):
        src = []
        trg = []
        for track_uri in row[0]:
            src.append(word2vec.wv.get_index(track_uri))
        for track_uri in row[1]:
            trg.append(word2vec.wv.get_index(track_uri))
        test_data[idx] = [torch.Tensor(src), torch.Tensor(trg)]

    return train_data, valid_data, test_data


class Encoder(nn.Module):
    def __init__(self, pre_trained_embedding, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = pre_trained_embedding
        self.rnn = nn.LSTM(100, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        # embedded.shape == (26, 100)
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, embedding_pre_trained, vocab_size, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = vocab_size
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = embedding_pre_trained
        self.rnn = nn.LSTM(100, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, self.output_dim)
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
        self.teacher_forcing_ratio = 0.0
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


def train(model, train_data, optimizer, criterion, device, batch_size=1, clip=1, epochs=2):
    model.train()
    epoch_loss = 0
    num_iterations = 0
    for x in range(0, epochs):
        for i in range(0, len(train_data)):
            num_iterations += 1
            print(num_iterations)
            src = train_data[i][0].to(device)
            trg = train_data[i][1].to(device)
            optimizer.zero_grad()
            output = model(src, trg)
            # output.shape(1, vocab_size)
            trg = trg.type(torch.LongTensor)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            print("loss = ", loss.item())
            epoch_loss += loss.item()
    return epoch_loss / (len(train_data)*epochs)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HID_DIM = 100
    N_LAYERS = 1
    ENC_DROPOUT = 0.0
    DEC_DROPOUT = 0.0
    # training parameters
    num_playlists_for_training = 1
    num_epochs = 100
    learning_rate = 0.01
    # print options
    # torch.set_printoptions(profile="full")

    print("create model...")
    word2vec = get_word2vec_model()
    weights = torch.FloatTensor(word2vec.wv.vectors)
    # weights.shape == (2262292, 100)
    vocab_size = len(word2vec.wv)
    embedding_pre_trained = nn.Embedding.from_pretrained(weights)

    enc = Encoder(embedding_pre_trained, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(embedding_pre_trained, vocab_size, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    print("finish")

    """print("init weights...")

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
    model.apply(init_weights)
    print("finish")"""

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters(), learning_rate)

    criterion = nn.CrossEntropyLoss()  # maybe add ignore_index = ...

    # read data from csv file
    print("Create train data...")
    train_data, test_data, valid_data = create_train_valid_test_data(num_playlists_for_training, 0, 0, word2vec)
    print("Created train data")

    if not os.path.isfile("models/pytorch/seq2seq_no_batch_pretrained_emb.pth"):
        x = train(model, train_data, optimizer, criterion, device, epochs=num_epochs, batch_size=10)
        torch.save(model.state_dict(), 'models/pytorch/seq2seq_no_batch_pretrained_emb.pth')
    else:
        model.load_state_dict(torch.load('models/pytorch/seq2seq_no_batch_pretrained_emb.pth'))
        # evaluate model:
        model.eval()
        with torch.no_grad():
            model.training = False
            with open('data/spotify_million_playlist_dataset_csv/data/track_sequences.csv', encoding='utf8') as read_obj:
                test_playlist = []
                csv_reader = csv.reader(read_obj)
                for idx, row in enumerate(csv_reader):
                    if idx == 1:
                        test_playlist = row
                        break
                idx_half = int(len(test_playlist)/2)
                src_uri = test_playlist[2:idx_half]
                trg_uri = test_playlist[idx_half:len(test_playlist)]
                print(src_uri)
                print(trg_uri)
                src = []
                trg = []
                for uri in src_uri:
                    src.append(word2vec.wv.get_index(uri))
                for uri in trg_uri:
                    trg.append(word2vec.wv.get_index(uri))
                print(src)
                print(trg)
                src = torch.LongTensor(src).to(device)
                trg = torch.LongTensor(trg).to(device)
                print(src)
                print(trg)
                output = model(src, trg)
                output = torch.argmax(output, dim=1)
                trg_model = []
                for track_idx in output:
                    trg_model.append(word2vec.wv.index_to_key[track_idx])
                print("Target: ", trg)
                print("Model output: ", trg_model)


