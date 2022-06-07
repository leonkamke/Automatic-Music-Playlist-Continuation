"""
training: train by computing the Cross Entropy Loss based on the shifted target (the output
            is shifted in one timestamp in comparison to the input)
prediction: do_rank (take k largest values (indices) for the prediction)
            do_mean_rank (take mean over all vector's correlating to each timestamp and then take the k
                            largest values (indices) for the prediction)
"""
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import os
import data_preprocessing.load_data as ld
from torch.utils.data import DataLoader
import evaluation.eval as eval
import load_attributes as la


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.1, 0.1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Autoencoder(nn.Module):
    def __init__(self, hid_dim, num_tracks, num_artists, num_albums, trackId2reducedTrackId, trackId2reducedArtistId,
                 trackId2reducedAlbumId, reducedTrackId2trackId):

        super(Autoencoder, self).__init__()
        self.trackId2reducedTrackId = trackId2reducedTrackId
        self.trackId2reducedArtistId = trackId2reducedArtistId
        self.trackId2reducedAlbumId = trackId2reducedAlbumId
        self.reducedTrackId2trackId = reducedTrackId2trackId

        self.hid_dim = hid_dim
        self.num_tracks = num_tracks
        self.num_artists = num_artists
        self.num_albums = num_albums
        self.input_size = self.num_tracks + self.num_artists  # + self.num_albums

        self.dropout = nn.Dropout(0.15)
        # input_size -> hid_dim
        self.encoder = torch.nn.Sequential(
            self.dropout,
            nn.Linear(self.input_size, hid_dim),
            nn.Sigmoid()
        )

        # hid_dim -> input_size
        self.decoder = torch.nn.Sequential(
            nn.Linear(hid_dim, self.input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        decoded = self.decoder(self.encoder(x))
        return decoded

    def predict(self, input, num_predictions):
        # input is a list of track_id's
        # input.shape == (seq_len)
        input_vector = self.map_sequence2vector(input)
        # input_vector.shape == (num_tracks + num_artists)
        # forward the vector through the autoencoder
        output_vector = self.forward(input_vector)[0:self.num_tracks]
        # get the top k indices/tracks
        _, top_k = torch.topk(output_vector, k=num_predictions)
        # transform the indices of the whole word2vec model
        output = []
        for reduced_track_id in top_k:
            track_id = self.reducedTrackId2trackId[int(reduced_track_id)]
            output.append(track_id)
        # output has to be a list of track_id's
        # outputs.shape == (num_predictions)
        return output

    def predict_(self, input, num_predictions):
        # input is a list of track_id's
        # input.shape == (seq_len)
        input_vector = self.map_sequence2vector(input)
        # input_vector.shape == (num_tracks + num_artists)
        # forward the vector through the autoencoder
        output_vector = self.forward(input_vector)[0:self.num_tracks]
        # Don't predict the input values -> set propabilities of input track_id's to 0
        for track_id in input:
            if track_id in self.trackId2reducedTrackId:
                track_id_reduced = self.trackId2reducedTrackId[track_id]
                output_vector[track_id_reduced] = 0
        # get the top k indices/tracks
        _, top_k = torch.topk(output_vector, k=num_predictions)
        # transform the indices of the whole word2vec model
        output = []
        for reduced_track_id in top_k:
            track_id = self.reducedTrackId2trackId[int(reduced_track_id)]
            output.append(track_id)
        # output has to be a list of track_id's
        # outputs.shape == (num_predictions)
        return output

    def map_sequence2vector(self, sequence):
        # input.shape == (seq_len)
        track_vector = torch.zeros(self.num_tracks)
        artist_vector = torch.zeros(self.num_artists)
        album_vector = torch.zeros(self.num_albums)
        # map sequence to vector of 1s and 0s (vector.shape == (input_size))
        for track_id in sequence:
            track_id = int(track_id)
            if track_id in self.trackId2reducedTrackId:
                new_track_id = self.trackId2reducedTrackId[track_id]
                track_vector[new_track_id] = 1
            if track_id in self.trackId2reducedArtistId:
                new_artist_id = self.trackId2reducedArtistId[track_id]
                artist_vector[new_artist_id] = 1
            """if track_id in self.trackId2reducedAlbumId:
                new_album_id = self.trackId2reducedAlbumId[track_id]
                album_vector[new_album_id] = 1"""
        return torch.cat((track_vector, artist_vector))
        # return torch.cat((track_vector, artist_vector, album_vector))


def train(model, dataloader, optimizer, criterion, device, num_epochs, max_norm):
    model.train()
    num_iterations = 1
    for epoch in range(num_epochs):
        for i, (src, trg) in enumerate(dataloader):
            src = src.to(device)
            trg = trg.to(device)
            optimizer.zero_grad()
            output = model(src)
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
    NUM_ARTISTS = len(reducedArtistUri2reducedId)
    NUM_ALBUMS = len(reducedAlbumUri2reducedId)
    print("track_size = ", NUM_TRACKS)
    print("artist_size = ", NUM_ARTISTS)
    print("album_size = ", NUM_ALBUMS)
    HID_DIM = 256
    max_norm = 5

    print("create Autoencoder model...")
    # (self, hid_dim, num_tracks, num_artists, num_albums, trackId2reducedTrackId, trackId2reducedArtistId,
    #                  reducedTrackId2trackId)
    model = Autoencoder(HID_DIM, NUM_TRACKS, NUM_ARTISTS, NUM_ALBUMS, trackId2reducedTrackId, trackId2reducedArtistId,
                        trackId2reducedAlbumId, reduced_trackId2trackId)
    print("finished")

    print("init weights...")
    model.apply(init_weights)
    print("finished")

    print(f'The model has {count_parameters(model):,} trainable parameters')
    print("The size of the input-layer is: ", model.input_size)

    optimizer = optim.Adam(model.parameters(), learning_rate)
    # optimizer = optim.SGD(model.parameters(), learning_rate)
    criterion = nn.BCELoss()

    print("Create train data...")
    dataset = ld.AutoencoderDataset(reducedTrackUri2reducedId, reducedArtistUri2reducedId, reducedAlbumUri2reducedId,
                                    num_playlists_for_training)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    print("Created train data")

    foldername = la.get_folder_name()
    save_file_name = "/autoencoder.pth"

    model.to(device)
    os.mkdir(la.output_path_model() + foldername)
    shutil.copyfile("attributes", la.output_path_model() + foldername + "/attributes.txt")
    train(model, dataloader, optimizer, criterion, device, num_epochs, max_norm)
    torch.save(model.state_dict(), la.output_path_model() + foldername + save_file_name)

    model.load_state_dict(torch.load(la.output_path_model() + foldername + save_file_name))
    device = torch.device("cpu")
    model.to(device)
    # evaluate model:
    model.eval()
    trackId2artistId = ld.get_trackid2artistid()
    trackUri2trackId = ld.get_trackuri2id()
    artistUri2artistId = ld.get_artist_uri2id()
    # def evaluate_model(model, trackId2artistId, trackUri2trackId, artistUri2artistId, start_idx, end_idx, device)
    results_str = eval.evaluate_model(model, trackId2artistId, trackUri2trackId, artistUri2artistId,
                                      la.get_start_idx(), la.get_end_idx(), device)
    # write results in a file with setted attributes
    f = open(la.output_path_model() + foldername + "/results.txt", "w")
    f.write("autoencoder: \n ")
    f.write(results_str)
    f.close()
