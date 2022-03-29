import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gensim
import os
import data_preprocessing.load_data as ld


def get_word2vec_model():
    print("load word2vec from file")
    model = gensim.models.Word2Vec.load("./models/gensim_word2vec/word2vec-song-vectors.model")
    print("word2vec loaded from file")
    return model


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

    def forward(self, input, prd_len):
        # input.shape == (batch_size, seq_len)
        x = self.embedding(input)
        # x.shape == (batch_size, seq_len, embed_dim == 100), when batch_first=True
        x, (h_n, c_n) = self.rnn(x)
        # x.shape == (batch_size, seq_len, hid_dim), when batch_first=True
        # x = self.fc_out(x)
        # x.shape == (batch_size, seq_len, vocab_size)
        return self.fc_out(x)

    def rank(self, output, n):
        # output.shape (batch_size, seq_len, vocab_size)
        x = torch.zeros(output.size(dim=0), n)
        for idx, sequence in enumerate(output):
            x[idx], _ = torch.topk(output[idx][-1], n)
        return x
        # returned prediction shape: (batch_size, n)


def train(model, src, trg, optimizer, criterion, device, batch_size=10, epochs=2, clip=1):
    model.train()
    epoch_loss = 0
    num_iterations = 0
    for x in range(0, epochs):
        for i in range(0, len(src), batch_size):
            src_i = src[i:i+batch_size].to(device)
            trg_i = trg[i:i+batch_size].to(device)
            optimizer.zero_grad()
            output = model(src_i, trg_i)
            # output.shape = (batch_size, seq_len, vocab_size)
            output = output.view(-1, model.vocab_size)
            trg_i = trg_i.view(-1)
            print(" trg_i.shape == ", trg_i.shape)
            print(" output.shape == ", output.shape)
            loss = criterion(output, trg_i)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            num_iterations += 1
            print("epoch ", x, " iteration ", num_iterations, " loss = ", loss.item())
            epoch_loss += loss.item()
        num_iterations = 0


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("load pretrained embedding layer...")
    word2vec = get_word2vec_model()
    weights = torch.FloatTensor(word2vec.wv.vectors)
    # weights.shape == (2262292, 100)
    # pre_trained embedding reduces the number of trainable parameters from 34 mill to 17 mill
    embedding_pre_trained = nn.Embedding.from_pretrained(weights)

    print("create Seq2Seq model")
    # Seq2Seq parameters
    # VOCAB_SIZE == 169657
    VOCAB_SIZE = len(word2vec.wv)
    HID_DIM = 100
    N_LAYERS = 1
    model = Seq2Seq(VOCAB_SIZE, embedding_pre_trained, HID_DIM, N_LAYERS).to(device)
    print("finish")

    print(f'The model has {count_parameters(model):,} trainable parameters')

    """print("init weights...")
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
    model.apply(init_weights)
    print("finish")"""

    # Training parameters
    learning_rate = 0.003
    num_epochs = 2
    batch_size = 10
    num_playlists_for_training = 300

    optimizer = optim.Adam(model.parameters(), learning_rate)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    print("Create train data...")
    src, trg = ld.create_train_data(num_playlists_for_training, 0, 0, word2vec)
    print("Created train data")

    if not os.path.isfile("models/pytorch/seq2seq_no_batch_pretrained_emb.pth"):
        # def train(model, src, trg, optimizer, criterion, device, batch_size=10, clip=1, epochs=2)
        train(model, src, trg, optimizer, criterion, device, batch_size=batch_size, epochs=num_epochs)
        torch.save(model.state_dict(), 'models/pytorch/seq2seq_no_batch_pretrained_emb.pth')
    else:
        model.load_state_dict(torch.load('models/pytorch/seq2seq_no_batch_pretrained_emb.pth'))
        # evaluate model:"""
