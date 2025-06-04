import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, mode='sum'):
        super(GRULayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_z = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.W_r = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.W_h = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.mode = mode

    def forward(self, x, hidden_state):
        b, seq_len, embed_size = x.size()
        out_forward = torch.zeros(b, seq_len, self.hidden_size).to(device)
        hidden_state_f = hidden_state
        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat((x_t, hidden_state_f), dim=1)
            z_t = F.sigmoid(self.W_z(combined))
            r_t = F.sigmoid(self.W_r(combined))
            combined_reset = torch.cat((x_t, r_t * hidden_state_f), dim=1)
            h_t = F.tanh(self.W_h(combined_reset))
            hidden_state_f = (1 - z_t) * hidden_state_f + z_t * h_t
            out_forward[:, t, :] = hidden_state_f

        if not self.mode:
            return out_forward, hidden_state_f

        out_backward = torch.zeros(b, seq_len, self.hidden_size).to(device)
        hidden_state_b = hidden_state
        for t in reversed(range(seq_len)):
            x_t = x[:, t, :]
            combined = torch.cat((x_t, hidden_state_b), dim=1)
            z_t = F.sigmoid(self.W_z(combined))
            r_t = F.sigmoid(self.W_r(combined))
            combined_reset = torch.cat((x_t, r_t * hidden_state_b), dim=1)
            h_t = F.tanh(self.W_h(combined_reset))
            hidden_state_b = (1 - z_t) * hidden_state_b + z_t * h_t
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


class GRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size, modes=None, n_layers=2):
        super(GRUBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.modes = [modes] * n_layers if isinstance(modes, str) or modes is None else modes
        self.layers = nn.ModuleList(
            [GRULayer(input_size, hidden_size, self.modes[0])] +
            [GRULayer(hidden_size, hidden_size, self.modes[i + 1]) for i in range(n_layers - 1)]
        )

    def forward(self, x):
        hidden_state = None
        for layer in self.layers:
            if hidden_state is None:
                hidden_state = layer.init_zero_hidden(x.size(0))
            x, hidden_state = layer(x, hidden_state)
        return x, hidden_state


class GRU(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, n_layers, vocab_size, modes=None, drop_prob=0.5):
        super(GRU, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = GRUBlock(embedding_dim, hidden_dim, modes=modes, n_layers=n_layers)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out, hidden = self.gru(embedded)
        out = self.dropout(out)
        out = self.fc(out[:, -1])
        return out


if __name__ == '__main__':
    epochs = 20
    output_size_ = 3
    embedding_dim_ = 400
    hidden_dim_ = 256
    n_layers_ = 2
    vocab_size = 8507
    modes = 'sum'

    model = GRU(output_size_, embedding_dim_, hidden_dim_, n_layers_, vocab_size, modes).to(device)
    model(torch.randint(0, vocab_size, (50, 313)).to(device))
