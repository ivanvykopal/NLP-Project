import torch
import logging

logging.basicConfig(level=logging.INFO)

class LSTMClassifier4(torch.nn.Module):
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
            dropout_size=0.5,
            device=torch.device('cpu')
        ):
        super(LSTMClassifier4, self).__init__()

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
        # self.embedding = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.dropout = torch.nn.Dropout(dropout_size)
        self.lstm1 = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional)
        # self.lstm2 = torch.nn.LSTM(hidden_dim * self.input_size_factor, hidden_dim, num_layers=n_layers, bidirectional=bidirectional)
        # softmax layer
        self.fc1 = torch.nn.Linear(hidden_dim * self.input_size_factor, hidden_dim)
        # activation ReLU
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        # activation softmax
        # self.softmax = torch.nn.Softmax(dim=1)
        # sigmoid
        # self.softmax = torch.nn.Sigmoid()
        self.sigmoid = torch.nn.Sigmoid()

    def init_hidden(self):
        h0 = torch.zeros(
            self.n_layers * self.input_size_factor,
            self.batch_size,
            self.hidden_dim
        ).to(self.device)
        c0 = torch.zeros(
            self.n_layers * self.input_size_factor, 
            self.batch_size, 
            self.hidden_dim
        ).to(self.device)
        return h0, c0
    
    def apply_rnn(self, embedding_out, lengths):
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedding_out, lengths, batch_first=True)
        activations, _ = self.lstm(packed_embedded, self.init_hidden())
        activations, _ = torch.nn.utils.rnn.pad_packed_sequence(activations, batch_first=True)

        indices = (lengths - 1).view(-1, 1).expand(
            activations.size(0), activations.size(2)
        ).unsqueeze(1)
        indices = indices.to(self.device)

        activations = activations.gather(1, indices).squeeze(1)
        return activations

    def forward(self, inputs, return_activations=False):
        batch_size = len(inputs)
        if batch_size != self.batch_size:
            # print(f'Batch size changed from {self.batch_size} to {batch_size}')
            self.batch_size = batch_size

        lengths = torch.LongTensor([len(sequence) for sequence in inputs])
        lengths, permutation_indices = lengths.sort(0, descending=True)

        padded_inputs = self.pad_sequences(inputs, padding_val=self.padding_idx)
        inputs = torch.LongTensor(padded_inputs)

        inputs = inputs[permutation_indices].to(self.device)
        embedding_out = self.embedding(inputs)
        # activations = self.apply_rnn(embedding_out, lengths)
        out, (hidden, cell) = self.lstm1(embedding_out)
        out = self.dropout(out)
        out = self.act(self.fc1(out[:, -1, :]))
        out = self.dropout(out)

        outputs_act = self.fc2(out)
        outputs = self.sigmoid(outputs_act)

        permutation_index_pairs = list(
            zip(permutation_indices.tolist(), list(range(len(permutation_indices))))
        )
        reordered_indices = [
            pair[1] for pair in sorted(permutation_index_pairs, key=lambda x: x[0])
        ]

        if return_activations:
            return outputs[reordered_indices], outputs_act[reordered_indices]
        # print shape
        return outputs[reordered_indices]

    def create_embedding_matrix(self, vocabulary, embedding_dict=None, embedding_dim=100):
        # initialize with normal distribution but the result will by torch tensor
        embedding_matrix = torch.normal(0.0, 1.0, size=(len(vocabulary), embedding_dim))
        for word, index in vocabulary.items():
            if word in embedding_dict:
                embedding_matrix[index] = torch.tensor(embedding_dict[word])

        return embedding_matrix
    
    def pad_sequences(self, sequences, padding_val=0, pad_left=False):
        sequence_length = max([len(sequence) for sequence in sequences])
        if not pad_left:
            return [
                sequence + [padding_val] * (sequence_length - len(sequence))
                for sequence in sequences
            ]
        return [
            [padding_val] * (sequence_length - len(sequence)) + sequence
            for sequence in sequences
        ]

        
