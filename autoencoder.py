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
    def __init__(self, num_tracks, num_artists, hid_dim, dropout=0):
        super(Autoencoder, self).__init__()
        self.hid_dim = hid_dim
        self.num_tracks = num_tracks
        self.num_artists = num_artists
        self.input_size = num_tracks

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

    def predict(self, input, num_predictions):
        # input is a list of track_id's
        # input.shape == (seq_len)
        # map sequence to vector of 1s and 0s (vector.shape == (input_size))
        # forward the vector through the autoencoder
        # result is again a vector of 1s and 0s


        # output has to be a list of track_id's
        # outputs.shape == (num_predictions)
        outputs = []
        return outputs


def train(model, dataloader, optimizer, criterion, device, num_epochs, max_norm):
    model.train()
    num_iterations = 1
    for epoch in range(num_epochs):
        for i, src in enumerate(dataloader):
            src = src.to(device)
            # src.shape = (batch_size, len(word2vec_tracks.wv))
            optimizer.zero_grad()
            output = model(src)
            # output.shape = (batch_size, seq_len, vocab_size)
            loss = criterion(output, src)
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

    print("load pretrained embedding layer...")

    print("load word2vec from file")
    word2vec_tracks_reduced = gensim.models.Word2Vec.load(la.path_track_to_vec_reduced_model())
    word2vec_artists = gensim.models.Word2Vec.load(la.path_artist_to_vec_model())
    print("word2vec loaded from file")

    # Training and model parameters
    learning_rate = la.get_learning_rate()
    num_epochs = la.get_num_epochs()
    batch_size = la.get_batch_size()
    num_playlists_for_training = la.get_num_playlists_training()
    # VOCAB_SIZE == 2262292
    NUM_TRACKS = len(word2vec_tracks_reduced.wv)
    NUM_ARTISTS = len(word2vec_artists.wv)
    HID_DIM = la.get_recurrent_dimension()
    HID_DIM = 256
    max_norm = 5

    print("create Seq2Seq model...")
    model = Autoencoder(NUM_TRACKS, NUM_ARTISTS, HID_DIM)
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
    dataset = ld.AutoencoderDataset(word2vec_tracks_reduced, word2vec_artists, num_playlists_for_training)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    print("Created train data")

    foldername = la.get_folder_name()
    save_file_name = "/autoencoder.pth"

    model.to(device)
    os.mkdir(la.output_path_model() + foldername)
    shutil.copyfile("attributes", la.output_path_model() + foldername + "/attributes.txt")
    # def train(model, src, trg, optimizer, criterion, device, batch_size=10, clip=1, epochs=2)
    train_shifted_target(model, dataloader, optimizer, criterion, device, num_epochs, max_norm)
    torch.save(model.state_dict(), la.output_path_model() + foldername + save_file_name)

    model.load_state_dict(torch.load(la.output_path_model() + foldername + save_file_name))
    device = torch.device("cuda")
    model.to(device)
    # evaluate model:
    model.eval()
    results_str = eval.evaluate_model(model, word2vec_tracks, word2vec_artists, la.get_start_idx(), la.get_end_idx(),
                                      device)

    # write results in a file with setted attributes
    f = open(la.output_path_model() + foldername + "/results.txt", "w")
    f.write(results_str)
    f.write("\nautoencoder ")
    f.close()
