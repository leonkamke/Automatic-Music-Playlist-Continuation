"""
training: train by computing the Cross Entropy Loss based on the shifted target (the output
            is shifted in one timestamp in comparison to the input)
prediction: do_rank (take k largest values (indices) for the prediction)
            do_mean_rank (take mean over all vector's correlating to each timestamp and then take the k
                            largest values (indices) for the prediction)
"""
import shutil
import gensim
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
    def __init__(self, num_tracks, num_artists, hid_dim, track2vec, track2vec_reduced, track2artist, artist2vec,
                 dropout=0):
        super(Autoencoder, self).__init__()
        self.track2vec = track2vec
        self.track2vec_reduced = track2vec_reduced
        self.track2artist = track2artist
        self.artist2vec = artist2vec

        self.hid_dim = hid_dim
        self.num_tracks = num_tracks
        self.num_artists = num_artists
        self.input_size = num_tracks + num_artists

        # input_size -> hid_dim
        self.encoder = torch.nn.Sequential(
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
        # map sequence to vector of 1s and 0s (vector.shape == (input_size))
        for track_id in sequence:
            track_uri = self.track2vec.wv.index_to_key[track_id]
            if track_uri in self.track2vec_reduced.wv.key_to_index:
                new_track_id = self.track2vec_reduced.wv.key_to_index[track_uri]
                track_vector[new_track_id] = 1

                artist_id = self.track2artist[int(track_id)]
                artist_uri = self.artist2vec.wv.index_to_key[artist_id]
                artist_id_reduced = self.artist2vec_reduced.wv.key_to_index[artist_uri]
                artist_vector[artist_id_reduced] = 1

        return torch.cat((track_vector, artist_vector))

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
            src = src.to(device)
            trg = trg.to(device)
            # src.shape = trg.shape = (batch_size, len(word2vec_tracks.wv))
            optimizer.zero_grad()
            output = model(src)
            # output.shape = (batch_size, seq_len, vocab_size)
            loss = criterion(output, trg)
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

    print("load word2vec from file")
    word2vec_tracks = gensim.models.Word2Vec.load(la.path_track_to_vec_model())
    word2vec_tracks_reduced = gensim.models.Word2Vec.load(la.path_track_to_vec_reduced_model())
    word2vec_artists = gensim.models.Word2Vec.load(la.path_artist_to_vec_model())
    word2vec_artists_reduced = gensim.models.Word2Vec.load(la.path_artist_to_vec_reduced_model())
    print("word2vec loaded from file")

    print("load track2artist dict")
    track2artist_dict = ld.get_artist_dict(word2vec_tracks, word2vec_artists)
    print("loaded dict")

    # Training and model parameters
    learning_rate = la.get_learning_rate()
    num_epochs = la.get_num_epochs()
    batch_size = la.get_batch_size()
    num_playlists_for_training = la.get_num_playlists_training()
    # VOCAB_SIZE == 2262292
    NUM_TRACKS = len(word2vec_tracks_reduced.wv)
    NUM_ARTISTS = len(word2vec_artists_reduced.wv)
    HID_DIM = la.get_recurrent_dimension()
    HID_DIM = 256
    max_norm = 5

    print("create Autoencoder model...")
    # self, num_tracks, num_artists, hid_dim, track2vec, track2vec_reduced, track2artist, artist2vec,
    #                  dropout=0
    model = Autoencoder(NUM_TRACKS, NUM_ARTISTS, HID_DIM, word2vec_tracks, word2vec_tracks_reduced, track2artist_dict,
                        word2vec_artists_reduced)
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
    dataset = ld.AutoencoderDatasetOld(word2vec_tracks_reduced, word2vec_artists_reduced, num_playlists_for_training)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    print("Created train data")

    foldername = la.get_folder_name()
    save_file_name = "/autoencoder.pth"

    model.to(device)
    #os.mkdir(la.output_path_model() + foldername)
    #shutil.copyfile("attributes", la.output_path_model() + foldername + "/attributes.txt")
    # def train(model, src, trg, optimizer, criterion, device, batch_size=10, clip=1, epochs=2)
    #train(model, dataloader, optimizer, criterion, device, num_epochs, max_norm)
    #torch.save(model.state_dict(), la.output_path_model() + foldername + save_file_name)

    model.load_state_dict(torch.load(la.output_path_model() + foldername + save_file_name))
    device = torch.device("cpu")
    model.to(device)
    # evaluate model:
    model.eval()
    results_str = eval.evaluate_model_old(model, word2vec_tracks, word2vec_artists, la.get_start_idx(), la.get_end_idx(),
                                      device)

    # write results in a file with setted attributes
    f = open(la.output_path_model() + foldername + "/results.txt", "w")
    f.write(results_str)
    f.write("\nautoencoder ")
    f.close()