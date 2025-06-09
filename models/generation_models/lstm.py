import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, mode='sum'):
        super(LSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.U_f = nn.Linear(input_size, hidden_size * 4)
        self.W_f = nn.Linear(hidden_size, hidden_size * 4)
        self.U_b = nn.Linear(input_size, hidden_size * 4)
        self.W_b = nn.Linear(hidden_size, hidden_size * 4)
        self.mode = mode

    def forward(self, x, hidden_state, cell_state):
        b, seq_len, embed_size = x.size()
        out_forward = torch.zeros(b, seq_len, self.hidden_size).to(device)
        hidden_state_f = hidden_state
        cell_state_f = cell_state
        for t in range(seq_len):
            x_t = x[:, t, :]
            f_t, i_t, g_t, o_t = (self.U_f(x_t) + self.W_f(hidden_state_f)).chunk(4, dim=-1)
            cell_state_f = cell_state_f * F.sigmoid(f_t) + F.sigmoid(i_t) * F.tanh(g_t)
            hidden_state_f = o_t * F.tanh(cell_state_f)
            out_forward[:, t, :] = hidden_state_f

        if not self.mode:
            return out_forward, hidden_state_f, cell_state_f

        out_backward = torch.zeros(b, seq_len, self.hidden_size).to(device)
        hidden_state_b = hidden_state
        cell_state_b = cell_state
        for t in reversed(range(seq_len)):
            x_t = x[:, t, :]
            f_t, i_t, g_t, o_t = (self.U_b(x_t) + self.W_b(hidden_state_b)).chunk(4, dim=-1)
            cell_state_b = cell_state_b * F.sigmoid(f_t) + F.sigmoid(i_t) * F.tanh(g_t)
            hidden_state_b = o_t * F.tanh(cell_state_b)
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

        return out, hidden_state, cell_state

    def init_zero_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False).to(device)

    def init_zero_cell(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False).to(device)


class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, modes=None, n_layers=2):
        super(LSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.modes = [modes] * n_layers if isinstance(modes, str) or modes is None else modes
        self.layers = nn.ModuleList(
            [LSTMLayer(input_size, hidden_size, self.modes[0])] +
            [LSTMLayer(hidden_size, hidden_size, self.modes[i + 1]) for i in range(n_layers - 1)]
        )

    def forward(self, x):
        hidden_state = None
        cell_state = None
        for layer in self.layers:
            if hidden_state is None:
                hidden_state = layer.init_zero_hidden(x.size(0))
            if cell_state is None:
                cell_state = layer.init_zero_cell(x.size(0))

            x, hidden_state, cell_state = layer(x, hidden_state, cell_state)
        return x, hidden_state


class LSTM(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, n_layers, vocab_size, modes=None, drop_prob=0.5):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = LSTMBlock(embedding_dim, hidden_dim, modes=modes, n_layers=n_layers)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out, hidden = self.lstm(embedded)
        out = self.dropout(out)
        b, seq_len, _ = out.size()
        out = self.fc(out.view(b * seq_len, -1))
        out = out.view(b, seq_len, -1)
        return out


if __name__ == '__main__':
    epochs = 20
    output_size_ = 3
    embedding_dim_ = 400
    hidden_dim_ = 256
    n_layers_ = 2
    vocab_size = 8507
    # modes = 'sum'

    model = LSTM(output_size_, embedding_dim_, hidden_dim_, n_layers_, vocab_size).to(device)
    model(torch.randint(0, vocab_size, (50, 313)).to(device))
