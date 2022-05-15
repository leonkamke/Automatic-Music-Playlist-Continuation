"""
training: train by computing the Cross Entropy Loss based on the shifted target (the output
            is shifted in one timestamp in comparison to the input)
prediction: do_rank (take k largest values (indices) for the prediction)
            do_mean_rank (take mean over all vector's correlating to each timestamp and then take the k
                            largest values (indices) for the prediction)
"""
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
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, pre_trained_embedding, hid_dim, n_layers, dropout=0):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        # input shape of embedding: (*) containing the indices
        # output shape of embedding: (*, embed_dim == 100)
        self.embedding = pre_trained_embedding
        # input shape of LSTM has to be (batch_size, seq_len, embed_dim == 100) when batch_first=True
        # output shape of LSTM: output.shape == (batch_size, seq_len, hid_dim)  when batch_first=True
        #                       h_n.shape == (n_layers, batch_size, hid_dim)
        #                       c_n.shape == (n_layers, batch_size, hid_dim)
        self.rnn = nn.LSTM(100, hid_dim, n_layers, batch_first=True, dropout=dropout)
        # input shape of Linear: (*, hid_dim)
        # output shape of Linear: (*, vocab_size)
        self.fc_out = nn.Linear(hid_dim, vocab_size)

    def forward(self, input):
        # input.shape == (batch_size, seq_len)
        x = self.embedding(input)
        # x.shape == (batch_size, seq_len, embed_dim == 100)

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
        print(x.shape)
        # x.shape == (seq_len, vocab_size)
        x = torch.mean(x, dim=0)
        # x.shape == (vocab_size)
        _, top_k = torch.topk(x, dim=0, k=num_predictions)
        # top_k.shape == (num_predictions)
        return top_k


def train_shifted_target(model, dataloader, optimizer, criterion, device, num_epochs, clip=1):
    model.train()
    num_iterations = 1
    for epoch in range(num_epochs):
        for i, (src, trg) in enumerate(dataloader):
            src = src.to(device)
            trg = trg.to(device)
            # trg.shape = src.shape = (batch_size, seq_len)
            optimizer.zero_grad()
            output = model(src)
            # output.shape = (batch_size, seq_len, vocab_size)
            loss = criterion(output.permute(0, 2, 1), trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            print("epoch ", epoch+1, " iteration ", num_iterations, " loss = ", loss.item())
            num_iterations += 1
        num_iterations = 1


if __name__ == '__main__':
    print("load pretrained embedding layer...")

    print("load word2vec from file")
    word2vec_tracks = gensim.models.Word2Vec.load(la.path_track_to_vec_model())
    print("word2vec loaded from file")

    weights = torch.FloatTensor(word2vec_tracks.wv.get_normed_vectors())
    # weights.shape == (2262292, 100)
    # pre_trained embedding reduces the number of trainable parameters from 34 mill to 17 mill
    embedding_pre_trained = nn.Embedding.from_pretrained(weights)
    print("finished")

    # Training and model parameters
    learning_rate = la.get_learning_rate()
    num_epochs = la.get_num_epochs()
    batch_size = la.get_batch_size()
    num_playlists_for_training = la.get_num_playlists_training()
    # VOCAB_SIZE == 169657
    VOCAB_SIZE = len(word2vec_tracks.wv)
    HID_DIM = la.get_recurrent_dimension()
    N_LAYERS = la.get_num_recurrent_layers()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    device = torch.device(la.get_device())

    print("create Seq2Seq model...")
    model = Seq2Seq(VOCAB_SIZE, embedding_pre_trained, HID_DIM, N_LAYERS).to(device)
    print("finished")

    print("init weights...")
    model.apply(init_weights)
    print("finished")

    print(f'The model has {count_parameters(model):,} trainable parameters')
    print("The size of the vocabulary is: ", VOCAB_SIZE)

    optimizer = optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    print("Create train data...")
    # dataset = ld.NextTrackDatasetShiftedTarget(word2vec_tracks, num_playlists_for_training)
    dataset = ld.NextTrackDatasetShiftedTarget(word2vec_tracks, num_playlists_for_training)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=ld.collate_fn_shifted_target)
    print("Created train data")

    if not os.path.isfile(la.output_path_model() + '/seq2seq_v4.pth'):
        # def train(model, src, trg, optimizer, criterion, device, batch_size=10, clip=1, epochs=2)
        train_shifted_target(model, dataloader, optimizer, criterion, device, num_epochs)
        torch.save(model.state_dict(), la.output_path_model() + '/seq2seq_v4.pth')
    else:
        model.load_state_dict(torch.load(la.output_path_model() + '/seq2seq_v4.pth'))
        # evaluate model:
        model.eval()
        # word2vec_tracks already initialised above
        word2vec_artists = gensim.models.Word2Vec.load(la.path_artist_to_vec_model())
        eval.evaluate_model(model, word2vec_tracks, word2vec_artists, la.get_start_idx(), la.get_end_idx(), device)
