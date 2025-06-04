import torch
import torch.nn as nn
import torch.nn.functional as F
import random

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

    def forward(self, x, hidden_state=None):
        for layer in self.layers:
            if hidden_state is None:
                hidden_state = layer.init_zero_hidden(x.size(0))
            x, hidden_state = layer(x, hidden_state)
        return x, hidden_state


class GRUEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, modes=None):
        super(GRUEncoder, self).__init__()
        self.embedder = nn.Embedding(input_size, embedding_size)
        self.gru = GRUBlock(embedding_size, hidden_size, modes=modes, n_layers=n_layers)

    def forward(self, x):
        x = self.embedder(x)
        out, hidden = self.gru(x)
        return out, hidden


class GRUDecoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, modes=None, use_attention=False):
        super(GRUDecoder, self).__init__()
        self.use_attention = use_attention
        if self.use_attention:
            self.gru = GRUBlock(embedding_size + hidden_size, hidden_size, modes=modes, n_layers=n_layers)
        else:
            self.gru = GRUBlock(embedding_size, hidden_size, modes=modes, n_layers=n_layers)
        self.out = nn.Linear(hidden_size, input_size)
        self.decoder_embedder = nn.Embedding(input_size, embedding_size)
        self.W_enc_out = nn.Linear(hidden_size, hidden_size)
        self.W_hidden = nn.Linear(hidden_size, hidden_size)
        self.V_weights = nn.Linear(hidden_size, 1)

    def forward(self, x, enc_output, hidden_state):
        dec_input = self.decoder_embedder(x.unsqueeze(1))
        if self.use_attention:
            alignment = F.tanh(self.W_enc_out(enc_output) + self.W_hidden(hidden_state).unsqueeze(1))
            dots = self.V_weights(alignment).squeeze(-1).unsqueeze(1)
            attn = F.softmax(dots, dim=-1)
            context = torch.matmul(attn, enc_output)
            dec_input = torch.cat((dec_input, context), dim=-1)
        out, hidden = self.gru(dec_input, hidden_state)
        prediction = self.out(out.squeeze(1))
        return prediction, hidden


class GRUSeq2Seq(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, enc_layers, dec_layers, enc_vocab_size, dec_vocab_size,
                 encoder_modes=None, decoder_modes=None, attention=False):
        super(GRUSeq2Seq, self).__init__()
        self.dec_vocab_size = dec_vocab_size
        self.hidden_dim = hidden_dim

        self.encoder = GRUEncoder(enc_vocab_size, embedding_dim, hidden_dim,
                                  n_layers=enc_layers, modes=encoder_modes)
        self.decoder = GRUDecoder(dec_vocab_size, embedding_dim, hidden_dim,
                                  n_layers=dec_layers, modes=decoder_modes, use_attention=attention)
        self.to_out = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, enc_input, target, teacher_forcing_ratio=0.5):
        b, seq_len = target.size()
        outputs = torch.zeros(b, seq_len, self.dec_vocab_size).to(device)

        enc_output, hidden = self.encoder(enc_input)

        dec_input = target[:, 0]

        for t in range(seq_len):
            output, hidden = self.decoder(dec_input, enc_output, hidden)

            outputs[:, t, :] = output

            is_teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(dim=1)
            dec_input = target[:, t] if is_teacher_force else top1

        return outputs


if __name__ == '__main__':
    epochs = 20
    output_size_ = 3
    embedding_dim_ = 400
    hidden_dim_ = 256
    n_layers_ = 2
    vocab_size = 51773
    dec_vocab_size_ = 14144

    model = GRUSeq2Seq(embedding_dim_, hidden_dim_, enc_layers=3, dec_layers=1, enc_vocab_size=vocab_size,
                       dec_vocab_size=dec_vocab_size_, attention=True, encoder_modes='sum').to(device)
    model(torch.randint(0, vocab_size, (50, 80)).to(device), torch.randint(0, dec_vocab_size_, (50, 10)).to(device))
