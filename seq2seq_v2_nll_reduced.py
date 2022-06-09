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
    def __init__(self, num_tracks, pre_trained_embedding, hid_dim, n_layers, dropout=0):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.num_tracks = num_tracks

        # input shape of embedding: (*) containing the indices
        # output shape of embedding: (*, embed_dim == 100)
        self.embedding = pre_trained_embedding
        # input shape of LSTM has to  be(batch_size, seq_len, embed_dim == 100) when batch_first=True
        # output shape of LSTM: output.shape == (batch_size, seq_len, hid_dim)  when batch_first=True
        #                       h_n.shape == (n_layers, batch_size, hid_dim)
        #                       c_n.shape == (n_layers, batch_size, hid_dim)
        self.rnn = nn.LSTM(100, hid_dim, n_layers, batch_first=True, dropout=dropout)
        # input shape of Linear: (*, hid_dim)
        # output shape of Linear: (*, vocab_size)
        self.fc_out = nn.Linear(hid_dim, num_tracks)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input):
        # input.shape == (batch_size, seq_len)
        x = self.embedding(input)
        # x.shape == (batch_size, seq_len, embed_dim == 100)
        x, (h_n, c_n) = self.rnn(x)
        # x.shape == (batch_size, seq_len, hid_dim), when batch_first=True

        x = self.fc_out(x)[:, 1, :]
        # x.shape == (batch_size, seq_len, num_tracks)
        x = self.log_softmax(x)

        # for each batch and sequence only return last vector
        # x = x[:, -1, :]
        # x.shape == (batch_size, num_tracks)

        x = torch.argmax(x, dim=1)
        # x.shape == (batch_size)

        return x, (h_n, c_n)


class Decoder(nn.Module):
    def __init__(self, num_tracks, pre_trained_embedding, hid_dim, n_layers, reducedTrackId2trackId, dropout=0):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.num_tracks = num_tracks
        self.reducedTrackId2trackId = reducedTrackId2trackId

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
        self.fc_out = nn.Linear(hid_dim, num_tracks)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, h_n, c_n):
        # input.shape == (batch_size)
        # convert input from reduced id's to normal id's
        with torch.no_grad():
            for i, reducedTrackId in enumerate(input):
                trackId = self.reducedTrackId2trackId[int(reducedTrackId)]
                input[i] = trackId

        x = self.embedding(input)
        # UNSQUEECE input such that the lstm understands that this is not a sequence of length(batch_size)!!!
        x = torch.unsqueeze(x, dim=1)
        # x.shape == (batch_size, 1, embed_dim)

        # x.shape == (batch_size, embed_dim == 100) but x.shape has to be (1, batch_size, embed_dim==100)!!!
        # h_n.shape == (1, batch_size, embed_dim)
        # x must have shape (batch_size, sequence_length, embed_dim)
        x, (h_n, c_n) = self.rnn(x, (h_n, c_n))
        # x.shape == (batch_size, 1, hid_dim), when batch_first=True

        x = self.fc_out(x.squeeze(1))
        # x.shape == (batch_size, num_tracks)
        x = self.log_softmax(x)
        return x, (h_n, c_n)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, num_tracks, reduced_trackId2trackId, device):
        super().__init__()
        self.reducedTrackId2trackId = reduced_trackId2trackId
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.num_tracks = num_tracks

    def forward(self, src, num_predictions):
        # src.shape == (batch_size, src_len)
        batch_size = src.shape[0]
        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, num_predictions, self.num_tracks).to(device)

        x, (h_n, c_n) = self.encoder(src)
        # x.shape == (batch_size) and contains indices of reduced track id's !!
        for i in range(num_predictions):
            output, (h_n, c_n) = self.decoder(x, h_n, c_n)
            # output.shape == (batch_size, 1, num_tracks)
            # safe the output in outputs
            outputs[:, i, :] = output
            # set next input for decoder
            x = torch.argmax(output, dim=1)
            # x.shape == (batch_size)

        # outputs.shape == (batch_size, num_predictions, self.num_tracks)
        return outputs

    def predict(self, input, num_predictions):
        input = torch.LongTensor(input)
        input = torch.unsqueeze(input, dim=0)
        x = self.forward(input, num_predictions)
        # x.shape == (1, num_predictions, num_tracks)
        x = torch.squeeze(x)
        # x.shape == (num_predictions, vocab_size)
        x = torch.argmax(x, dim=1)
        # x.shape == (num_predictions)
        output = []
        for reducedTrackId in x:
            output.append(self.reducedTrackId2trackId[int(reducedTrackId)])
        return output

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
        output = []
        for id in top_k:
            output.append(self.id_dict[int(id)])
        output = torch.LongTensor(output)
        return output


def train(model, dataloader, optimizer, criterion, device, num_epochs, max_norm):
    model.train()
    num_iterations = 1
    for epoch in range(num_epochs):
        for i, (src, trg) in enumerate(dataloader):
            src = src.to(device)
            trg = trg.to(device)
            # trg.shape = src.shape = (batch_size, seq_len)
            optimizer.zero_grad()
            output = model(src, src.shape[1])
            output = output.permute(0, 2, 1)
            # output.shape = (batch_size, seq_len, vocab_size)
            # but Cross Entropy Loss requires output.shape = (batch_size, vocab_size, seq_len)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            print("epoch ", epoch + 1, " iteration ", num_iterations, " loss = ", loss.item())
            num_iterations += 1
        num_iterations = 1


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(la.get_device())

    print("load dictionaries from file")
    reducedTrackUri2reducedId = ld.get_reducedTrackUri2reducedTrackID()
    reducedArtistUri2reducedId = ld.get_reducedArtistUri2reducedArtistID()
    reducedAlbumUri2reducedId = ld.get_reducedAlbumUri2reducedAlbumID()
    reduced_trackId2trackId = ld.get_reduced_trackid2trackid()
    trackId2reducedTrackId = ld.get_trackid2reduced_trackid()
    trackId2reducedArtistId = ld.get_trackid2reduced_artistid()
    trackId2reducedAlbumId = ld.get_trackid2reduced_albumid()
    trackId2artistId = ld.get_trackid2artistid()
    trackUri2trackId = ld.get_trackuri2id()
    artistUri2artistId = ld.get_artist_uri2id()
    print("loaded dictionaries from file")

    print("load pretrained embedding layer...")
    # weights = torch.FloatTensor(word2vec_tracks.wv.get_normed_vectors())
    weights = torch.load(la.path_embedded_weights_tracks(), map_location=device)
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
    NUM_TRACKS = len(reducedTrackUri2reducedId)
    HID_DIM = la.get_recurrent_dimension()
    N_LAYERS = la.get_num_recurrent_layers()
    num_steps = 20
    max_norm = 5

    print("create EncoderDecoder model...")
    # Encoder params: (vocab_size, pre_trained_embedding, hid_dim, n_layers, dropout=0)
    encoder = Encoder(NUM_TRACKS, embedding_pre_trained, HID_DIM, N_LAYERS).to(device)
    # Decoder params: (vocab_size, pre_trained_embedding, hid_dim, n_layers, dropout=0):
    decoder = Decoder(NUM_TRACKS, embedding_pre_trained, HID_DIM, N_LAYERS, reduced_trackId2trackId).to(device)
    # Seq2Seq params: (encoder, decoder, vocab_size, device)
    model = Seq2Seq(encoder, decoder, NUM_TRACKS, reduced_trackId2trackId, device).to(device)
    model.to(device)
    print("finished")

    print("init weights...")
    model.apply(init_weights)
    print("finished")

    print(f'The model has {count_parameters(model):,} trainable parameters')
    print("The size of the vocabulary is: ", NUM_TRACKS)

    optimizer = optim.Adam(model.parameters(), learning_rate)
    criterion = nn.NLLLoss(ignore_index=-1)

    print("Create train data...")
    dataset = ld.EncoderDecoderReducedFixedStep(trackUri2trackId, reducedTrackUri2reducedId,
                                                num_playlists_for_training, num_steps)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    print("Created train data")

    foldername = la.get_folder_name()
    save_file_name = "/seq2seq_v2_reduced_nll.pth"

    os.mkdir(la.output_path_model() + foldername)
    shutil.copyfile("attributes", la.output_path_model() + foldername + "/attributes.txt")
    # def train(model, src, trg, optimizer, criterion, device, batch_size=10, clip=1, epochs=2)

    train(model, dataloader, optimizer, criterion, device, num_epochs, max_norm)
    torch.save(model.state_dict(), la.output_path_model() + foldername + save_file_name)

    model.load_state_dict(torch.load(la.output_path_model() + foldername + save_file_name))
    device = torch.device("cuda")
    model.to(device)
    # evaluate model:
    model.eval()
    # word2vec_tracks already initialised above
    # def evaluate_model(model, trackId2artistId, trackUri2trackId, artistUri2artistId, start_idx, end_idx, device)
    results_str = eval.evaluate_model(model, trackId2artistId, trackUri2trackId, artistUri2artistId,
                                      la.get_start_idx(), la.get_end_idx(), device)

    # write results in a file with setted attributes
    f = open(la.output_path_model() + foldername + "/results.txt", "w")
    f.write("\nseq2seq_v2_nll_reduced: ")
    f.write(results_str)
    f.close()
