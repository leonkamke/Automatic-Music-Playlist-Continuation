import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import random
import gensim


def get_word2vec_model():
    print("load word2model from file")
    model = gensim.models.Word2Vec.load("./models/gensim_word2vec/word2vec-song-vectors.model")
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
                src = row[2:i]
                trg = row[i: len(row)]
                train_data.append([src, trg])
            elif index < num_rows_train + num_rows_valid:
                if len(row) <= 3:
                    continue
                i = int(len(row) / 2 + 1)
                src = row[2:i]
                trg = row[i: len(row)]
                valid_data.append([src, trg])
            else:
                if len(row) <= 3:
                    continue
                i = int(len(row) / 2 + 1)
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
            src.append(word2vec.wv.get_vector(track_uri))
        for track_uri in row[1]:
            trg.append(word2vec.wv.get_vector(track_uri))
        train_data[idx] = [torch.Tensor(src), torch.Tensor(trg)]
    for idx, row in enumerate(valid_data):
        src = []
        trg = []
        for track_uri in row[0]:
            src.append(word2vec.wv.get_vector(track_uri))
        for track_uri in row[1]:
            trg.append(word2vec.wv.get_vector(track_uri))
        valid_data[idx] = [torch.Tensor(src), torch.Tensor(trg)]
    for idx, row in enumerate(test_data):
        src = []
        trg = []
        for track_uri in row[0]:
            src.append(word2vec.wv.get_vector(track_uri))
        for track_uri in row[1]:
            trg.append(word2vec.wv.get_vector(track_uri))
        test_data[idx] = [torch.Tensor(src), torch.Tensor(trg)]

    return train_data, valid_data, test_data


class Encoder(nn.Module):
    def __init__(self, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(100, hid_dim, n_layers, dropout=dropout)

    def forward(self, src):
        outputs, (hidden, cell) = self.rnn(src)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = vocab_size
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(100, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, self.output_dim)

    def forward(self, input, hidden, cell):
        # increase dimension of embedded from 1 to 2
        input = torch.unsqueeze(input, 0)
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, word2vec, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.word2vec = word2vec
        self.teacher_forcing_ratio = 0.5
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
            key = word2vec.wv.index_to_key[best_candidate]
            input = trg[t] if teacher_force and self.training else torch.Tensor(word2vec.wv.get_vector(key))
        # predictions.shape == (len(trg), vocab_size)
        return predictions


def train(model, train_data, optimizer, criterion, batch_size=1, clip=1, epochs=2):
    model.train()
    epoch_loss = 0
    num_iterations = 0
    for x in range(0, epochs):
        for i in range(0, len(train_data), batch_size):
            num_iterations += 1
            print(num_iterations)
            src = train_data[i][0]  # src.shape = (26, 100) = trg.shape
            trg = train_data[i][1]
            optimizer.zero_grad()
            output = model(src, trg)
            # output.shape(1, vocab_size)
            trg_for_loss = []
            # trg_for_loss has to be a list of the track indices
            for idx, vec in enumerate(trg):
                model.word2vec.wv.inde
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            print("loss = ", loss.item())
            epoch_loss += loss.item()
    return epoch_loss / (len(train_data)*epochs)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HID_DIM = 50
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    # training parameters
    num_playlists_for_training = 50
    num_epochs = 2
    # print options
    # torch.set_printoptions(profile="full")

    print("create model...")
    word2vec = get_word2vec_model()
    vocab_size = len(word2vec.wv)
    enc = Encoder(HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(vocab_size, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, word2vec, device).to(device)
    print("finish")

    print("init weights...")
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
    model.apply(init_weights)
    print("finish")

    def count_parameters(model):
        n = 0
        for p in model.parameters():
            if p.requires_grad:
                n += p.numel()
        return n
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters(), lr=0.1)

    criterion = nn.CrossEntropyLoss()  # maybe add ignore_index = ...

    # read data from csv file
    train_data, valid_data, test_data = create_train_valid_test_data(num_playlists_for_training, 0, 0, word2vec)

    x = train(model, train_data, optimizer, criterion, epochs=num_epochs)

    #torch.save(model.state_dict(), 'models/pytorch/seq2seq_no_batch.pth')
    # model.load_state_dict(torch.load('models/pytorch/seq2seq_no_batch.pth'))

    ''''# evaluate model:
    model.eval()
    with torch.no_grad():
        model.training = False
        #print(x)
        output = model(torch.tensor([4, 5, 6, 7, 8]), torch.tensor([9, 10, 11, 12]))
        print(output.shape)
        print("sequence prediction: ", torch.argmax(output, dim=1))
        print("output vector: ", output[0])'''
