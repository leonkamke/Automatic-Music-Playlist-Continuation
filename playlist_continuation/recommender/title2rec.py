"""
Character level Recurrent Neural Network which creates recommendations for a given playlist title
"""

import shutil
from playlist_continuation.data_preprocessing import build_character_vocab as cv
import torch
import torch.nn as nn
import torch.optim as optim
import os
import playlist_continuation.data_preprocessing.load_data as ld
from torch.utils.data import DataLoader
import playlist_continuation.evaluation.eval as eval
from playlist_continuation.config import load_attributes as la


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.1, 0.1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Title2Rec(nn.Module):
    def __init__(self, num_chars, num_tracks, hid_dim, n_layers, reducedTrackId2trackId, dropout=0):
        super(Title2Rec, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.input_size = num_chars
        self.reducedTrackId2trackId = reducedTrackId2trackId

        # input shape of embedding: (*) containing the indices
        # output shape of embedding: (*, embed_dim == 100)
        self.embedding = nn.Embedding(self.input_size, 100)
        # input shape of LSTM has to be (batch_size, seq_len, embed_dim == 100) when batch_first=True
        # output shape of LSTM: output.shape == (batch_size, seq_len, hid_dim)  when batch_first=True
        #                       h_n.shape == (n_layers, batch_size, hid_dim)
        #                       c_n.shape == (n_layers, batch_size, hid_dim)
        self.rnn = nn.LSTM(100, hid_dim, n_layers, batch_first=True)
        # input shape of Linear: (*, hid_dim)
        # output shape of Linear: (*, vocab_size)
        self.fc_out = nn.Linear(hid_dim, num_tracks)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # input.shape == (batch_size, seq_len)
        x = self.embedding(input)
        # x.shape == (batch_size, seq_len, embed_dim == 100)
        x, (h_n, c_n) = self.rnn(x)
        # x.shape == (batch_size, seq_len, hid_dim), when batch_first=True
        x = self.fc_out(x)
        # x.shape == (batch_size, num_tracks)
        x = self.sigmoid(x)

        return x

    def predict(self, title, num_predictions):
        input = cv.title2index_seq(title)
        input = torch.LongTensor(input)
        input = input.to(torch.device("cuda"))
        # input.shape == seq_len
        x = self.forward(input)
        # x.shape == (seq_len, num_tracks)
        x = x[-1]
        # x.shape == (num_tracks)
        _, top_k = torch.topk(x, dim=0, k=num_predictions)
        # top_k.shape == (num_predictions)
        output = []
        for reduced_track_id in top_k:
            track_id = self.reducedTrackId2trackId[int(reduced_track_id)]
            output.append(track_id)
        # output has to be a list of track_id's
        # outputs.shape == (num_predictions)
        return output


def train(model, dataloader, optimizer, criterion, device, num_epochs, max_norm):
    model.train()
    num_iterations = 1
    for epoch in range(num_epochs):
        for i, (src, trg) in enumerate(dataloader):
            src = src.to(device)
            # src.shape = (batch_size, num_characters_title)
            trg = trg.to(device)
            # trg.shape = src.shape = (batch_size, num_tracks)
            optimizer.zero_grad()
            output = model(src)[:, -1, :]
            # output.shape = (batch_size, num_tracks)
            del src
            # output.shape = (batch_size, seq_len, vocab_size)
            loss = criterion(output, trg)
            del trg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            print("epoch ", epoch + 1, " iteration ", num_iterations, " loss = ", loss.item())
            num_iterations += 1
        num_iterations = 1


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    device = torch.device(la.get_device())

    print("load dictionaries from file")
    reducedTrackUri2reducedId = ld.get_reducedTrackUri2reducedTrackID()
    reducedArtistUri2reducedId = ld.get_reducedArtistUri2reducedArtistID()
    reducedAlbumUri2reducedId = ld.get_reducedAlbumUri2reducedAlbumID()
    reduced_trackId2trackId = ld.get_reduced_trackid2trackid()
    trackId2reducedTrackId = ld.get_trackid2reduced_trackid()
    trackId2reducedArtistId = ld.get_trackid2reduced_artistid()
    trackId2reducedAlbumId = ld.get_trackid2reduced_albumid()
    print("loaded dictionaries from file")

    # Training and model parameters
    learning_rate = la.get_learning_rate()
    num_epochs = la.get_num_epochs()
    batch_size = la.get_batch_size()
    num_playlists_for_training = la.get_num_playlists_training()
    # VOCAB_SIZE == 2262292
    NUM_TRACKS = len(reducedTrackUri2reducedId)
    NUM_CHARS = 43
    HID_DIM = la.get_recurrent_dimension()
    N_LAYERS = la.get_num_recurrent_layers()
    max_norm = 5

    print("create Title2Rec model...")
    model = Title2Rec(NUM_CHARS, NUM_TRACKS, HID_DIM, N_LAYERS, reduced_trackId2trackId)
    print("finished")

    print("init weights...")
    model.apply(init_weights)
    print("finished")

    print(f'The model has {count_parameters(model):,} trainable parameters')
    print("The number of tracks is: ", NUM_TRACKS)

    optimizer = optim.Adam(model.parameters(), learning_rate)
    # optimizer = optim.SGD(model.parameters(), learning_rate)
    criterion = nn.BCELoss()

    print("Create train data...")
    # (self, reducedTrackuri_2_id, reducedArtisturi_2_id, reducedAlbumuri_2_id, num_rows_train)
    dataset = ld.Title2RecDataset(reducedTrackUri2reducedId, reducedArtistUri2reducedId, reducedAlbumUri2reducedId,
                                  num_playlists_for_training)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    print("Created train data")

    foldername = la.get_folder_name()
    save_file_name = "/seq2seq_v4_track_album_artist.pth"

    model.to(device)
    os.mkdir(la.output_path_model() + foldername)
    shutil.copyfile("../config/attributes", la.output_path_model() + foldername + "/attributes.txt")
    # def train(model, src, trg, optimizer, criterion, device, batch_size=10, clip=1, epochs=2)
    train(model, dataloader, optimizer, criterion, device, num_epochs, max_norm)
    torch.save(model.state_dict(), la.output_path_model() + foldername + save_file_name)

    model.load_state_dict(torch.load(la.output_path_model() + foldername + save_file_name))
    device = torch.device("cpu")
    model.to(device)

    # evaluate model:
    model.eval()
    print("load dictionaries for evaluating the model")
    trackId2artistId = ld.get_trackid2artistid()
    trackUri2trackId = ld.get_trackuri2id()
    artistUri2artistId = ld.get_artist_uri2id()
    print("finished")
    # def evaluate_model(model, trackId2artistId, trackUri2trackId, artistUri2artistId, start_idx, end_idx, device)
    results_str = eval.evaluate_title2rec_model(model, trackId2artistId, trackUri2trackId, artistUri2artistId,
                                                la.get_start_idx(), la.get_end_idx(), device)

    # write results in a file with setted attributes
    f = open(la.output_path_model() + foldername + "/results.txt", "w")
    f.write(results_str)
    f.write("\nseq2seq_v4_nlll: ")
    f.close()
