"""
Encoder-Decoder recurrent neural network
training: Seq2Seq model which takes the output of the decoder into account for computing the
            Cross Entropy Loss
prediction:
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


class Encoder(nn.Module):
    def __init__(self, vocab_size, pre_trained_embedding, hid_dim, n_layers, dropout=0):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        # input shape of embedding: (*) containing the indices
        # output shape of embedding: (*, embed_dim == 100)
        self.embedding = pre_trained_embedding
        # input shape of LSTM has to  be(batch_size, seq_len, embed_dim == 100) when batch_first=True
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
        # x.shape == (batch_size, seq_len, embed_dim == 100)
        x, (h_n, c_n) = self.rnn(x)
        # x.shape == (batch_size, seq_len, hid_dim), when batch_first=True

        x = self.fc_out(x)
        # x.shape == (batch_size, seq_len, vocab_size)

        # for each batch and sequence only return last vector
        x = x[:, -1, :]
        # x.shape == (batch_size, vocab_size)

        x = torch.argmax(x, dim=1)
        # x.shape == (batch_size)

        return x, (h_n, c_n)


class Decoder(nn.Module):
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
        self.rnn = nn.LSTM(300, hid_dim, n_layers, batch_first=True, dropout=dropout)
        # input shape of Linear: (*, hid_dim)
        # output shape of Linear: (*, vocab_size)
        self.fc_out = nn.Linear(hid_dim, vocab_size)

    def forward(self, input, h_n, c_n):
        # input.shape == (batch_size)
        x = self.embedding(input)
        x = torch.unsqueeze(x, dim=1)
        # x.shape == (batch_size, 1, embed_dim)

        # UNSQUEECE input such that the lstm understands that this is not a sequence of length(batch_size)!!!
        # x.shape == (batch_size, embed_dim == 100) but x.shape has to be (1, batch_size, embed_dim==100)!!!
        # h_n.shape == (1, batch_size, embed_dim)
        # x must have shape (batch_size, sequence_length, embed_dim)
        x, (h_n, c_n) = self.rnn(x, (h_n, c_n))
        # x.shape == (batch_size, 1, hid_dim), when batch_first=True

        x = self.fc_out(x.squeeze(1))
        # x.shape == (batch_size, vocab_size)

        return x, (h_n, c_n)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, vocab_size, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.vocab_size = vocab_size

    def forward(self, src, num_predictions):
        # src.shape == (batch_size, src_len)
        batch_size = src.shape[0]
        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, num_predictions, self.vocab_size).to(self.device)

        x, (h_n, c_n) = self.encoder(src)
        # x.shape == (batch_size)
        for i in range(num_predictions):
            output, (h_n, c_n) = self.decoder(x, h_n, c_n)
            # output.shape == (batch_size, 1, vocab_size)
            # safe the output in outputs
            outputs[:, i, :] = output
            # set next input for decoder
            x = torch.argmax(output, dim=1)
            # x.shape == (batch_size)

        # outputs.shape == (batch_size, num_predictions, self.vocab_size)
        return outputs

    def predict(self, input, num_predictions):
        input = torch.unsqueeze(input, dim=0)
        x = self.forward(input, num_predictions)
        # x.shape == (1, num_predictions, vocab_size)
        x = torch.squeeze(x)
        # x.shape == (num_predictions, vocab_size)
        x = torch.argmax(x, dim=1)
        # x.shape == (num_predictions)
        return x

    def predict_1(self, input, num_predictions):
        input = torch.unsqueeze(input, dim=0)
        x = self.forward(input, num_predictions)
        # x.shape == (1, num_predictions, vocab_size)
        x = torch.squeeze(x)
        # x.shape == (num_predictions, vocab_size)
        x = torch.mean(x, dim=0)
        # x.shape == (vocab_size)
        _, top_k = torch.topk(x, dim=0, k=num_predictions)
        # top_k.shape == (num_predictions)
        return top_k


def train(model, dataloader, optimizer, criterion, device, num_epochs, max_norm):
    model.train()
    num_iterations = 1
    for epoch in range(num_epochs):
        for i, (src, trg, trg_len) in enumerate(dataloader):
            src = src.to(device)
            trg = trg.to(device)
            # trg.shape = src.shape = (batch_size, seq_len)
            optimizer.zero_grad()
            output = model(src, src.shape[1])
            # output.shape = (batch_size, seq_len, vocab_size)
            # but Cross Entropy Loss requires output.shape = (batch_size, vocab_size, seq_len)
            loss = criterion(output.permute(0, 2, 1), trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            print("epoch ", epoch+1, " iteration ", num_iterations, " loss = ", loss.item())
            num_iterations += 1
        num_iterations = 1


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(la.get_device())

    print("load pretrained embedding layer...")

    print("load word2vec from file")
    word2vec_tracks = gensim.models.Word2Vec.load(la.path_track_to_vec_model())
    print("word2vec loaded from file")

    # weights = torch.FloatTensor(word2vec_tracks.wv.get_normed_vectors())
    weights = torch.load(la.path_embedded_weights(), map_location=device)
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
    max_norm = 5
    num_steps = 10

    print("create Seq2Seq model...")
    # Encoder params: (vocab_size, pre_trained_embedding, hid_dim, n_layers, dropout=0)
    encoder = Encoder(VOCAB_SIZE, embedding_pre_trained, HID_DIM, N_LAYERS).to(device)
    # Decoder params: (vocab_size, pre_trained_embedding, hid_dim, n_layers, dropout=0):
    decoder = Decoder(VOCAB_SIZE, embedding_pre_trained, HID_DIM, N_LAYERS).to(device)
    # Seq2Seq params: (encoder, decoder, vocab_size, device)
    model = Seq2Seq(encoder, decoder, VOCAB_SIZE, device)
    print("finished")

    print("init weights...")
    model.apply(init_weights)
    print("finished")

    print(f'The model has {count_parameters(model):,} trainable parameters')
    print("The size of the vocabulary is: ", VOCAB_SIZE)

    optimizer = optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    print("Create train data...")
    dataset = ld.PlaylistDatasetFixedStep(word2vec_tracks, num_playlists_for_training, num_steps)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    print("Created train data")

    foldername = la.get_folder_name()
    save_file_name = "/seq2seq_v2_track_album_artist.pth"

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
    # word2vec_tracks already initialised above
    word2vec_artists = gensim.models.Word2Vec.load(la.path_artist_to_vec_model())
    results_str = eval.evaluate_model(model, word2vec_tracks, word2vec_artists, la.get_start_idx(), la.get_end_idx(),
                                      device)

    # write results in a file with setted attributes
    f = open(la.output_path_model() + foldername + "/results.txt", "w")
    f.write(results_str)
    f.write("\nseq2seq_v2")
    f.close()
