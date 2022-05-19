import torch
import torch.nn as nn


class Projection_LSTM(nn.Module):
    # embedding size should be same with word2vec feature size
    def __init__(self, embedding_size=25, lstm_hidden_size=4096):
        super(Projection_LSTM, self).__init__()

        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=lstm_hidden_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        return output.transpose(0, 1)[-1]
