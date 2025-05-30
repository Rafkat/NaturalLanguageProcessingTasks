import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Utils:
    pad_index = 0

    @staticmethod
    def get_pos_embed(len_seq, embedding_dim):
        pos_embed = torch.zeros(1, len_seq, embedding_dim)
        for k in range(len_seq):
            for i in torch.arange(int(embedding_dim / 2)):
                denominator = torch.pow(10000, 2 * i / embedding_dim)
                pos_embed[:, k, 2 * i] = torch.sin(k / denominator)
                pos_embed[:, k, 2 * i + 1] = torch.cos(k / denominator)
        return pos_embed

    @staticmethod
    def make_trg_mask(target):
        b, seq_len = target.shape
        trg_mask = torch.tril(torch.ones((seq_len, seq_len))).expand(b, 1, seq_len, seq_len).to(device)
        return trg_mask

    @classmethod
    def make_src_mask(cls, x):
        return (x != cls.pad_index).unsqueeze(1).unsqueeze(2).to(device)

    @classmethod
    def make_combined_mask(cls, target):
        look_ahead_mask = cls.make_trg_mask(target)
        padding_mask = cls.make_src_mask(target)
        return torch.logical_and(look_ahead_mask, padding_mask)


class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(self, input_size, head_dim, num_heads, dropout=0.):
        super(Attention, self).__init__()
        self.proj = not (num_heads == 1 and head_dim == input_size)
        hidden_size = head_dim * num_heads

        self.heads = num_heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(input_size, hidden_size)
        self.to_k = nn.Linear(input_size, hidden_size)
        self.to_v = nn.Linear(input_size, hidden_size)

        self.attn = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_size)
        self.to_out = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout),
        ) if self.proj else nn.Identity()

    def forward(self, q, k, v, mask=None):
        q = self.to_q(self.norm(q))
        k = self.to_k(self.norm(k))
        v = self.to_v(self.norm(v))

        q = q.reshape(q.size(0), q.size(1), self.heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(k.size(0), k.size(1), self.heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(v.size(0), v.size(1), self.heads, -1).permute(0, 2, 1, 3)

        dots = torch.matmul(q, k.transpose(-1, -2))

        if mask is not None:
            dots = dots.masked_fill(mask == 0, float('-1e20'))

        dots *= self.scale
        attn = self.dropout(self.attn(dots))

        out = torch.matmul(attn, v)
        out = out.view(out.size(0), out.size(2), -1)
        return self.to_out(out)


class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len_seq, input_size, depth, head_dim=64, num_heads=8, dropout=0.):
        super(Encoder, self).__init__()
        self.norm = nn.LayerNorm(input_size)
        self.enc_embedding = nn.Embedding(vocab_size, input_size)
        self.pos_embed = nn.Parameter(Utils.get_pos_embed(max_len_seq, input_size))
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(input_size, head_dim, num_heads, dropout),
                FeedForward(input_size, int(head_dim * num_heads), dropout)
            ]) for _ in range(depth)
        ])

    def forward(self, x, mask):
        x = self.enc_embedding(x)
        x = x + self.pos_embed
        for attn, ff in self.layers:
            x = attn(x, x, x, mask) + x
            x = self.norm(x)
            x = ff(x) + x
            x = self.norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len_seq, input_size, depth, head_dim=64, num_heads=8, dropout=0.):
        super(Decoder, self).__init__()
        self.norm = nn.LayerNorm(input_size)
        self.dec_embedding = nn.Embedding(vocab_size, input_size)
        self.pos_embed = nn.Parameter(Utils.get_pos_embed(max_len_seq, input_size))
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(input_size, head_dim, num_heads, dropout),
                FeedForward(input_size, int(head_dim * num_heads), dropout)
            ]) for _ in range(depth)
        ])

    def forward(self, x, enc_k, enc_v, look_ahead_mask, padding_mask):
        x = self.dec_embedding(x)
        x = x + self.pos_embed
        for attn, ff in self.layers:
            x = attn(x, x, x, look_ahead_mask) + x
            x = self.norm(x)
            x = attn(x, enc_k, enc_v, padding_mask) + x
            x = self.norm(x)
            x = ff(x) + x
            x = self.norm(x)
        return x


class Transformer(nn.Module):
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_len_seq, dec_len_seq, embedding_dim, enc_depth, dec_depth,
                 head_dim=64, num_heads=8, dropout=0., ):
        super(Transformer, self).__init__()
        self.pad_index = 0
        self.encoder = Encoder(enc_vocab_size, enc_len_seq, embedding_dim, enc_depth, head_dim, num_heads, dropout)
        self.decoder = Decoder(dec_vocab_size, dec_len_seq, embedding_dim, dec_depth, head_dim, num_heads, dropout)
        self.fc = nn.Linear(embedding_dim, dec_vocab_size)

    def forward(self, enc_input, target, enc_padding_mask=None, look_ahead_mask=None):
        if not enc_padding_mask:
            enc_padding_mask = Utils.make_src_mask(enc_input)
        if not look_ahead_mask:
            look_ahead_mask = Utils.make_combined_mask(target)
        x = self.encoder(enc_input, enc_padding_mask)
        x = self.decoder(target, x, x, look_ahead_mask, enc_padding_mask)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    epochs = 20
    output_size_ = 3
    embedding_dim_ = 400
    hidden_dim_ = 256
    n_layers_ = 2
    vocab_size = 51773
    dec_vocab_size_ = 14144

    model = Transformer(enc_vocab_size=vocab_size, dec_vocab_size=dec_vocab_size_, enc_len_seq=80, dec_len_seq=10,
                        embedding_dim=embedding_dim_, enc_depth=n_layers_, dec_depth=n_layers_).to(device)
    model(torch.randint(0, vocab_size, (5, 80)).to(device), torch.randint(0, dec_vocab_size_, (5, 10)).to(device))
