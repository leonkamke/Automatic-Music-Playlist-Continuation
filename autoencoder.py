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
    def __init__(self, hid_dim, track2vec, artist2vec, album2vec, track2vec_reduced, artist2vec_reduced,
                 album2vec_reduced, track2artist, track2album):
        super(Autoencoder, self).__init__()
        self.track2vec = track2vec
        self.track2vec_reduced = track2vec_reduced
        self.artist2vec = artist2vec
        self.artist2vec_reduced = artist2vec_reduced
        self.album2vec = album2vec
        self.album2vec_reduced = album2vec_reduced

        self.track2artist = track2artist
        self.track2album = track2album

        self.hid_dim = hid_dim
        self.num_tracks = len(track2vec_reduced.wv)
        self.num_artists = len(artist2vec_reduced.wv)
        self.num_albums = len(album2vec_reduced.wv)
        self.input_size = self.num_tracks + self.num_artists + self.num_albums

        self.dropout = nn.Dropout(0.2)
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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def map_sequence2vector(self, sequence):
        # input.shape == (seq_len)
        track_vector = torch.zeros(self.num_tracks)
        artist_vector = torch.zeros(self.num_artists)
        album_vector = torch.zeros(self.num_albums)
        # map sequence to vector of 1s and 0s (vector.shape == (input_size))
        for track_id in sequence:
            track_uri = self.track2vec.wv.index_to_key[track_id]
            if track_uri in self.track2vec_reduced.wv.key_to_index:
                new_track_id = self.track2vec_reduced.wv.key_to_index[track_uri]
                track_vector[new_track_id] = 1

            artist_id = self.track2artist[int(track_id)]
            artist_uri = self.artist2vec.wv.index_to_key[artist_id]
            if artist_uri in self.artist2vec_reduced.wv.key_to_index:
                new_artist_id = self.artist2vec_reduced.wv.key_to_index[artist_uri]
                artist_vector[new_artist_id] = 1

            album_id = self.track2album[int(track_id)]
            album_uri = self.album2vec.wv.index_to_key[album_id]
            if album_uri in self.album2vec_reduced.wv.key_to_index:
                new_album_id = self.album2vec_reduced.wv.key_to_index[album_uri]
                album_vector[new_album_id] = 1

        return torch.cat((track_vector, artist_vector, album_vector))

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
        for track_id in top_k:
            track_uri = self.track2vec_reduced.wv.index_to_key[track_id]
            new_track_id = self.track2vec.wv.key_to_index[track_uri]
            output.append(new_track_id)
        # output has to be a list of track_id's
        # outputs.shape == (num_predictions)
        return output


def train(model, dataloader, optimizer, criterion, device, num_epochs, max_norm):
    model.train()
    num_iterations = 1
    for epoch in range(num_epochs):
        for i, (src, trg) in enumerate(dataloader):
            #src = src.to(device)
            #trg = trg.to(device)
            # src.shape = trg.shape = (batch_size, len(word2vec_tracks.wv))
            optimizer.zero_grad()
            output = model(src)
            del src
            # output.shape = (batch_size, seq_len, vocab_size)
            loss = criterion(output, trg)
            del trg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            print("epoch ", epoch+1, " iteration ", num_iterations, " loss = ", loss.item())
            num_iterations += 1
        num_iterations = 1


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    device = torch.device(la.get_device())

    print("load dictionaries from file")
    reducedTrackUri2reducedId = ld.get_reducedTrackUri2reducedTrackID()


    trackuri_2_id = ld.get_trackuri2id()
    artisturi_2_id = ld.get_artist_uri2id()
    albumuri_2_id = ld.get_albums_uri2id()

    track2artist_dict = ld.get_trackid2artistid()
    track2album_dict = ld.get_trackid2albumid()
    print("loaded dictionaries from file")

    print("track_size = ", len(trackuri_2_id))
    print("artist_size = ", len(artisturi_2_id))
    print("album_size = ", len(albumuri_2_id))

    # Training and model parameters
    learning_rate = la.get_learning_rate()
    num_epochs = la.get_num_epochs()
    batch_size = la.get_batch_size()
    num_playlists_for_training = la.get_num_playlists_training()
    # VOCAB_SIZE == 2262292
    NUM_TRACKS = len(trackuri_2_id)
    NUM_ARTISTS = len(artisturi_2_id)
    HID_DIM = 256
    max_norm = 5

    print("create Autoencoder model...")
    # (self, hid_dim, track2vec, artist2vec, album2vec, track2vec_reduced, artist2vec_reduced,
    #                  album2vec_reduced, track2artist, track2album, dropout=0):
    model = Autoencoder(HID_DIM, word2vec_tracks, word2vec_artists, word2vec_albums, word2vec_tracks_reduced,
                        word2vec_artists_reduced, word2vec_albums_reduced,
                        track2artist_dict, track2album_dict)
    print("finished")

    print("init weights...")
    model.apply(init_weights)
    print("finished")

    print(f'The model has {count_parameters(model):,} trainable parameters')
    print("The size of the vocabulary is: ", NUM_TRACKS)

    optimizer = optim.Adam(model.parameters(), learning_rate)
    # optimizer = optim.SGD(model.parameters(), learning_rate)
    criterion = nn.BCELoss()

    print("Create train data...")
    # dataset = ld.NextTrackDatasetShiftedTarget(word2vec_tracks, num_playlists_for_training)
    dataset = ld.AutoencoderDataset(reducedTrackUri2reducedId, artisturi_2_id, albumuri_2_id,
                                    num_playlists_for_training)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=6,
                            collate_fn=ld.collate_fn_autoencoder)
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
    results_str = eval.evaluate_model(model, la.get_start_idx(), la.get_end_idx(), device)

    # write results in a file with setted attributes
    f = open(la.output_path_model() + foldername + "/results.txt", "w")
    f.write("autoencoder: \n ")
    f.write(results_str)
    f.close()
