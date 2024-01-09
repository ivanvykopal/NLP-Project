import torch
import logging

logging.basicConfig(level=logging.INFO)


class POSLSTMClassifier(torch.nn.Module):
    def __init__(
        self,
        embedding_dim=100,
        hidden_dim=64,
        n_layers=2,
        output_dim=2,
        bidirectional=True,
        embedding_dict=None,
        vocabulary=None,
        padding_idx=0,
        batch_size=1,
        device=torch.device('cpu')
    ):
        super(POSLSTMClassifier, self).__init__()

        self.input_size_factor = 2 if bidirectional else 1
        self.padding_idx = padding_idx
        self.batch_size = batch_size
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        embedding_matrix = self.create_embedding_matrix(
            vocabulary=vocabulary,
            embedding_dict=embedding_dict,
            embedding_dim=embedding_dim
        )
        vocab_size = len(vocabulary)

        # # we will not train the embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weights = torch.nn.Parameter(
            torch.FloatTensor(embedding_matrix))
        self.embedding.weights.requires_grad = False
        # self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.dropout = torch.nn.Dropout(0.2)
        self.lstm = torch.nn.LSTM(
            embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional)
        # softmax layer
        self.fc = torch.nn.Linear(
            hidden_dim * self.input_size_factor, output_dim)
        # activation ReLU
        self.act = torch.nn.ReLU()
        # activation softmax
        self.out = torch.nn.Softmax(dim=1)

    def forward(self, inputs, return_activations=False):
        embedding_out = self.embedding(inputs)
        lstm_out, _ = self.lstm(embedding_out)

        outputs = self.dropout(lstm_out)
        outputs_act = self.fc(outputs)
        outputs = self.out(outputs_act)

        if return_activations:
            return outputs, outputs_act
        # print shape
        return outputs

    def create_embedding_matrix(self, vocabulary, embedding_dict=None, embedding_dim=100):
        # initialize with normal distribution but the result will by torch tensor
        embedding_matrix = torch.normal(
            0.0, 1.0, size=(len(vocabulary), embedding_dim))
        for word, index in vocabulary.items():
            if word in embedding_dict:
                embedding_matrix[index] = torch.tensor(embedding_dict[word])

        return embedding_matrix
