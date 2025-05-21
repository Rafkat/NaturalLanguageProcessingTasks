import torch
from torch import nn


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
        k = self.to_q(self.norm(k))
        v = self.to_q(self.norm(v))

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
    def __init__(self, input_size, depth, head_dim=64, num_heads=8, dropout=0.):
        super(Encoder, self).__init__()
        self.norm = nn.LayerNorm(input_size)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(input_size, head_dim, num_heads, dropout),
                FeedForward(input_size, int(head_dim * num_heads), dropout)
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x, x, x) + x
            x = self.norm(x)
            x = ff(x) + x
            x = self.norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, depth, head_dim=64, num_heads=8, dropout=0.):
        super(Decoder, self).__init__()
        self.norm = nn.LayerNorm(input_size)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(input_size, head_dim, num_heads, dropout),
                FeedForward(input_size, int(head_dim * num_heads), dropout)
            ]) for _ in range(depth)
        ])

    def forward(self, enc_q, enc_k, x, mask):
        for attn, ff in self.layers:
            x = attn(x, x, x, mask) + x
            x = self.norm(x)
            x = attn(enc_q, enc_k, x) + x
            x = self.norm(x)
            x = ff(x) + x
            x = self.norm(x)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, max_len_seq, embedding_dim, enc_depth, head_dim=64, num_heads=8, dropout=0.,
                 nb_classes=10):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = Encoder(embedding_dim, enc_depth, head_dim, num_heads, dropout)
        self.fc = nn.Linear(embedding_dim, nb_classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pos_embed = torch.zeros(1, max_len_seq + 1, embedding_dim)
        for k in range(max_len_seq + 1):
            for i in torch.arange(int(embedding_dim / 2)):
                denominator = torch.pow(10000, 2 * i / embedding_dim)
                self.pos_embed[:, k, 2 * i] = torch.sin(k / denominator)
                self.pos_embed[:, k, 2 * i + 1] = torch.cos(k / denominator)
        self.pos_embed = nn.Parameter(self.pos_embed)

    def forward(self, x):
        x = self.embedding(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.fc(x[:, 0, :])
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 20
    output_size_ = 3
    embedding_dim_ = 400
    n_layers_ = 2
    vocab_size = 8507

    model = TransformerClassifier(vocab_size, 313, embedding_dim_, enc_depth=n_layers_, nb_classes=output_size_).to(
        device)
    model(torch.randint(0, vocab_size, (50, 313)).to(device))
