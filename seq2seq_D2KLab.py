import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gensim
import os
import data_preprocessing.load_data as ld
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint


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

        self.layers = nn.Sequential(
            self.embedding,
            self.rnn,
            self.fc_out
        )

    def forward(self, input):
        # input.shape == (batch_size, seq_len)
        x = self.embedding(input)
        x.requires_grad = True
        # x = checkpoint(self.embedding, input)
        # x.shape == (batch_size, seq_len, embed_dim == 100), when batch_first=True

        # x, (h_n, c_n) = self.rnn(x)
        x, (h_n, c_n) = checkpoint(self.rnn, x)
        # x.shape == (batch_size, seq_len, hid_dim), when batch_first=True

        x = checkpoint(self.fc_out, x[:, -1, :])
        # x = self.fc_out(x)
        # x.shape == (batch_size, seq_len, vocab_size)
        return x

    def predict(self, input, len):
        x = self.forward(input)
        # x.shape == (batch_size, seq_len, vocab_size)
        x = x.argmax(dim=2)


def train(model, dataloader, optimizer, criterion, device, num_epochs, clip=1):
    model.train()
    num_iterations = 1
    for epoch in range(num_epochs):
        for i, (src, trg) in enumerate(dataloader):
            src = src.to(device)
            trg = trg.to(device)
            # trg.shape = (batch_size, seq_len)
            optimizer.zero_grad()
            output = model(src)
            for ind in range(len(output)):
                output[ind, :] = output[ind, -1]
            del src
            # output.shape = (batch_size, seq_len, vocab_size)
            output = output.view(-1, model.vocab_size)
            # output.shape = (batch_size * seq_len, vocab_size)
            trg = trg.view(-1)
            # trg.shape = (batch_size * seq_len)
            print("trg shape view ", trg.shape)
            loss = criterion(output, trg)
            del trg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            print("epoch ", epoch+1, " iteration ", num_iterations, " loss = ", loss.item())
            num_iterations += 1
        num_iterations = 1


if __name__ == '__main__':
    print("load pretrained embedding layer...")
    word2vec = ld.get_word2vec_model("10_thousand_playlists")
    weights = torch.FloatTensor(word2vec.wv.vectors)
    # weights.shape == (2262292, 100)
    # pre_trained embedding reduces the number of trainable parameters from 34 mill to 17 mill
    embedding_pre_trained = nn.Embedding.from_pretrained(weights)
    print("finished")

    # Training and model parameters
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 5
    num_playlists_for_training = 100
    # VOCAB_SIZE == 169657
    VOCAB_SIZE = len(word2vec.wv)
    HID_DIM = 100
    N_LAYERS = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    dataset = ld.PlaylistDataset(word2vec, num_playlists_for_training)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=ld.collate_fn)
    print("Created train data")

    if not os.path.isfile("models/pytorch/seq2seq_no_batch_pretrained_emb.pth"):
        # def train(model, src, trg, optimizer, criterion, device, batch_size=10, clip=1, epochs=2)
        train(model, dataloader, optimizer, criterion, device, num_epochs)
        torch.save(model.state_dict(), 'models/pytorch/seq2seq_no_batch_pretrained_emb.pth')
    else:
        model.load_state_dict(torch.load('models/pytorch/seq2seq_no_batch_pretrained_emb.pth'))
        # evaluate model:"""
        model.eval()
        src1, trg1 = dataset[0]
        src1 = src1.to(device)
        prediction = model(src1)
        prediction = prediction.argmax(dim=1)
        print(prediction.shape)
        print("prediction: ", prediction)
        print("target: ", trg1)

