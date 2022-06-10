"""
training: train by computing the Cross Entropy Loss based on only one target (the last prediction)
prediction: do_rank (take k largest values (indices) for the prediction)
            do_mean_rank (take mean over all vector's correlating to each timestamp and then take the k
                            largest values (indices) for the prediction)"""
import gensim
import torch
import torch.nn as nn
import torch.optim as optim
import os
import data_preprocessing.load_data as ld
from torch.utils.data import DataLoader
import evaluation.eval as eval
import load_attributes as la
import shutil


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.1, 0.1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, pre_trained_embedding, hid_dim, n_layers, dropout=0.0):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        # input shape of embedding: (*) containing the indices
        # output shape of embedding: (*, embed_dim == 200)
        self.embedding = pre_trained_embedding
        # input shape of LSTM has to be (batch_size, seq_len, embed_dim == 200) when batch_first=True
        # output shape of LSTM: output.shape == (batch_size, seq_len, hid_dim)  when batch_first=True
        #                       h_n.shape == (n_layers, batch_size, hid_dim)
        #                       c_n.shape == (n_layers, batch_size, hid_dim)
        self.rnn = nn.LSTM(300, hid_dim, n_layers, batch_first=True, dropout=dropout)
        # input shape of Linear: (*, hid_dim)
        # output shape of Linear: (*, vocab_size)
        self.fc_out = nn.Linear(hid_dim, vocab_size)

    def forward(self, input):
        # input.shape == (batch_size, seq_len)
        x = self.embedding(input)
        # x.shape == (batch_size, seq_len, embed_dim == 300)

        x, (h_n, c_n) = self.rnn(x)
        # x.shape == (batch_size, seq_len, hid_dim), when batch_first=True

        x = self.fc_out(x)
        # x.shape == (batch_size, seq_len, vocab_size)

        return x

    def predict(self, input, num_predictions):
        # input.shape == seq_len
        x = self.forward(input)
        # x.shape == (seq_len, vocab_size)
        x = x[-1]
        # x.shape == (vocab_size)
        _, top_k = torch.topk(x, dim=0, k=num_predictions)
        # top_k.shape == (num_predictions)
        return top_k

    def predict_do_summed_rank(self, input, num_predictions):
        # input.shape == seq_len
        x = self.forward(input)
        # x.shape == (seq_len, vocab_size)
        x = torch.mean(x, dim=0)
        # x.shape == (vocab_size)
        _, top_k = torch.topk(x, dim=0, k=num_predictions)
        # top_k.shape == (num_predictions)
        return top_k


"""def train_one_target(model, dataloader, optimizer, criterion, device, num_epochs):
    model.train()
    num_iterations = 1
    batch_size = dataloader.batch_size
    vocab_size = model.vocab_size
    for epoch in range(num_epochs):
        for i, (src, trg, src_len) in enumerate(dataloader):
            src = src.to(device)
            # src.shape == (batch_size, seq_len)
            trg = trg.to(device)
            # trg.shape == (batch_size)
            optimizer.zero_grad()
            batch_output = torch.zeros((batch_size, vocab_size)).to(device)
            for idx, src_i in enumerate(src):
                # src_i.shape == (seq_len)
                # model(src_i).shape == (seq_len, vocab_size)
                # model(src_i)[-1].shape == (vocab_size)
                index = src_len[idx] - 1
                # the output of the model is the last predicted track_id
                batch_output[idx] = model(src_i)[index]
                # output.shape == (vocab_size)
            # batch_output.shape = (batch_size, vocab_size)
            loss = criterion(batch_output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
            optimizer.step()
            print("epoch ", epoch + 1, " iteration ", num_iterations, " loss = ", loss.item())
            num_iterations += 1
        num_iterations = 1"""


def train_one_target(model, dataloader, optimizer, criterion, device, num_epochs, max_norm):
    model.train()
    num_iterations = 1
    for epoch in range(num_epochs):
        for i, (src, trg) in enumerate(dataloader):
            src = src.to(device)
            # src.shape == (batch_size, seq_len)
            trg = trg.to(device)
            # trg.shape = (batch_size)
            optimizer.zero_grad()
            output = model(src)[:, -1, :]
            # output.shape = (batch_size, vocab_size)
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
    print("word2vec loaded from file")

    print("load pretrained embedding layer")
    weights = torch.load(la.path_embedded_weights(), map_location=device)
    # weights.shape == (2262292, 200)
    # pre_trained embedding reduces the number of trainable parameters from  454 million to 228,572,292
    embedding_pre_trained = nn.Embedding.from_pretrained(weights)
    print("finished")

    # Training and model parameters
    learning_rate = la.get_learning_rate()
    num_epochs = la.get_num_epochs()
    batch_size = la.get_batch_size()
    num_playlists_for_training = la.get_num_playlists_training()
    # VOCAB_SIZE == 2262292
    VOCAB_SIZE = len(word2vec_tracks.wv)
    HID_DIM = la.get_recurrent_dimension()
    N_LAYERS = la.get_num_recurrent_layers()
    max_norm = 5
    num_steps = 10

    print("create Seq2Seq model...")
    model = Seq2Seq(VOCAB_SIZE, embedding_pre_trained, HID_DIM, N_LAYERS)
    print("finished")

    print("init weights...")
    model.apply(init_weights)
    print("finished")

    print(f'The model has {count_parameters(model):,} trainable parameters')
    print("The size of the vocabulary is: ", VOCAB_SIZE)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("Create train data...")
    # dataset = ld.NextTrackDatasetShiftedTarget(word2vec_tracks, num_playlists_for_training)
    dataset = ld.NextTrackDatasetOnlyOneTargetFixedStep(word2vec_tracks, num_playlists_for_training, num_steps)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    print("Created train data")

    foldername = la.get_folder_name()
    save_file_name = "/seq2seq_v3_track_album_artist.pth"

    model.to(device)
    os.mkdir(la.output_path_model() + foldername)
    shutil.copyfile("../attributes", la.output_path_model() + foldername + "/attributes.txt")
    # def train(model, src, trg, optimizer, criterion, device, batch_size=10, clip=1, epochs=2)
    train_one_target(model, dataloader, optimizer, criterion, device, num_epochs, max_norm)
    torch.save(model.state_dict(), la.output_path_model() + foldername + save_file_name)

    model.load_state_dict(torch.load(la.output_path_model() + foldername + save_file_name))
    device = torch.device("cpu")
    model.to(device)
    # evaluate model:
    model.eval()
    # word2vec_tracks already initialised above
    word2vec_artists = gensim.models.Word2Vec.load(la.path_artist_to_vec_model())
    results_str = eval.evaluate_model(model, word2vec_tracks, word2vec_artists, la.get_start_idx(), la.get_end_idx(),
                                      device)

    # write results in a file with setted attributes
    f = open(la.output_path_model() + foldername + "/results.txt", "w")
    f.write(results_str)
    f.write("\nseq2seq_v3_track_album_artist: ")
    f.close()