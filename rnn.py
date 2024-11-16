import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import string
from tqdm import tqdm
import json
from argparse import ArgumentParser

unk = '<UNK>'

# Load pretrained embeddings like GloVe
def load_glove_embeddings(filepath, embedding_dim):
    embedding_index = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vector
    return embedding_index

# Create an embedding matrix based on vocab
def create_embedding_matrix(vocab, embedding_index, embedding_dim):
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, idx in vocab.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
        else:
            embedding_matrix[idx] = embedding_index.get(unk, np.random.normal(size=(embedding_dim,)))
    return torch.tensor(embedding_matrix, dtype=torch.float32)

class RNN(nn.Module):
    def __init__(self, input_dim, h, pretrained_embeddings):
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1

        # Use pretrained embeddings and freeze them
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        
        self.rnn = nn.LSTM(input_dim, h, self.numOfLayer, batch_first=False)
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        embedded = self.embedding(inputs)  # Get embeddings (seq_length, batch_size, embedding_dim)
        
        h0 = torch.zeros(self.numOfLayer, inputs.size(1), self.h).to(inputs.device)
        c0 = torch.zeros(self.numOfLayer, inputs.size(1), self.h).to(inputs.device)

        output, (hidden, cell_state) = self.rnn(embedded, (h0, c0))
        last_hidden = output[-1]  # Get last time step output
        logits = self.W(last_hidden)
        predicted_vector = self.softmax(logits)
        return predicted_vector

def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"]-1)))
    return tra, val

def convert_text_to_indices(text, word2index):
    return torch.tensor([word2index.get(word.lower(), word2index[unk]) for word in text], dtype=torch.long)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--embedding_file", required=True, help="path to pretrained embedding file (e.g., GloVe)")
    parser.add_argument("--embedding_dim", type=int, required=True, help="dimension of pretrained embeddings")
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)
    
    print("========== Loading embeddings ==========")
    embedding_index = load_glove_embeddings(args.embedding_file, args.embedding_dim)

    # Create vocabulary from training data
    vocab = {}
    for document, _ in train_data:
        for word in document:
            if word.lower() not in vocab:
                vocab[word.lower()] = len(vocab)
    vocab[unk] = len(vocab)

    print("========== Creating embedding matrix ==========")
    embedding_matrix = create_embedding_matrix(vocab, embedding_index, args.embedding_dim)

    print("========== Initializing model ==========")
    model = RNN(input_dim=args.embedding_dim, h=args.hidden_dim, pretrained_embeddings=embedding_matrix)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0

    for epoch in range(args.epochs):
        random.shuffle(train_data)
        model.train()
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_indices = convert_text_to_indices(input_words, vocab)
                input_indices = input_indices.unsqueeze(1)  # Shape: (seq_length, batch_size)
                
                output = model(input_indices)
                example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))

                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Train Accuracy: {correct / total:.4f}")
    epoch = epoch +1
