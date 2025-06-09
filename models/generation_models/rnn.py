import torch
import torch.nn as nn
import torch.nn.functional as F
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, mode='sum'):
        super(RNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h_f = nn.Linear(input_size, hidden_size)
        self.h2h_f = nn.Linear(hidden_size, hidden_size)
        self.i2h_b = nn.Linear(input_size, hidden_size)
        self.h2h_b = nn.Linear(hidden_size, hidden_size)
        self.mode = mode

    def forward(self, x, hidden_state):
        b, seq_len, embed_size = x.size()
        out_forward = torch.zeros(b, seq_len, self.hidden_size).to(device)
        hidden_state_f = hidden_state
        for t in range(seq_len):
            x_t = x[:, t, :]
            x_t = self.i2h_f(x_t)
            hidden_state_f = self.h2h_f(hidden_state_f)
            hidden_state_f = F.tanh(x_t + hidden_state_f)
            out_forward[:, t, :] = hidden_state_f

        if not self.mode:
            return out_forward, hidden_state_f

        out_backward = torch.zeros(b, seq_len, self.hidden_size).to(device)
        hidden_state_b = hidden_state
        for t in reversed(range(seq_len)):
            x_t = x[:, t, :]
            x_t = self.i2h_b(x_t)
            hidden_state_b = self.h2h_b(hidden_state_b)
            hidden_state_b = F.tanh(x_t + hidden_state_b)
            out_backward[:, t, :] = hidden_state_b

        if self.mode == 'sum':  # sentiment analysis, enhances common features
            out = out_forward + out_backward
        elif self.mode == 'concat':  # NER, POS-tagging, when both directions valuable
            out = torch.cat((out_forward, out_backward), dim=-1)
        elif self.mode == 'max':  # Key features maximization, key phrases and anomaly detection tasks
            out = torch.max(out_forward, out_backward)
        elif self.mode == 'mean':  # Smoothens outliers, good for regression tasks, directions importance equal
            out = (out_forward + out_backward) / 2
        else:
            raise ValueError('Not existing mode')

        return out, hidden_state

    def init_zero_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False).to(device)


class RNNBlock(nn.Module):
    def __init__(self, input_size, hidden_size, modes=None, n_layers=2):
        super(RNNBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.modes = [modes] * n_layers if isinstance(modes, str) or modes is None else modes
        self.layers = nn.ModuleList(
            [RNNLayer(input_size, hidden_size, self.modes[0])] +
            [RNNLayer(hidden_size, hidden_size, self.modes[i + 1]) for i in range(n_layers - 1)]
        )

    def forward(self, x):
        hidden_state = None
        for layer in self.layers:
            if hidden_state is None:
                hidden_state = layer.init_zero_hidden(x.size(0))
            x, hidden_state = layer(x, hidden_state)
        return x, hidden_state


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, modes=None, drop_prob=0.1):
        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = RNNBlock(embedding_dim, hidden_dim, modes=modes, n_layers=n_layers)
        # self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers,dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.dropout(out)
        b, seq_len, _ = out.size()
        outputs = self.fc(out.view(b * seq_len, -1))
        out = outputs.view(b, seq_len, -1)
        return out


if __name__ == '__main__':
    epochs = 20
    embedding_dim_ = 400
    hidden_dim_ = 256
    n_layers_ = 2
    vocab_size = 8507
    modes = 'sum'

    model = RNN(vocab_size, embedding_dim_, hidden_dim_, n_layers_, modes).to(device)
    model(torch.randint(0, vocab_size, (50, 313)).to(device))
