import torch
import torch.nn as nn


class DependencyAwareTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(DependencyAwareTransformerEncoderLayer, self).__init__()

        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.dependency_attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, input_seq, dependency_matrix):
        # Self-attention
        self_attn_output, _ = self.self_attention(input_seq, input_seq, input_seq)
        self_attn_output = self.layer_norm1(input_seq + self_attn_output)

        # Dependency attention
        dep_attn_output, _ = self.dependency_attention(input_seq, input_seq, input_seq, key_padding_mask=None,
                                                       attn_mask=dependency_matrix)
        dep_attn_output = self.layer_norm2(self_attn_output + dep_attn_output)

        # Feed-forward
        ff_output = self.feed_forward(dep_attn_output)
        enc_output = self.layer_norm3(dep_attn_output + ff_output)

        return enc_output


class DependencyAwareTransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(DependencyAwareTransformerEncoder, self).__init__()

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.transformer_layers = nn.ModuleList([
            DependencyAwareTransformerEncoderLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, input_seq, dependency_matrix):
        # Embedding
        embedded_seq = self.embedding(input_seq)

        # Transformer layers
        enc_output = embedded_seq
        for layer in self.transformer_layers:
            enc_output = layer(enc_output, dependency_matrix)

        return enc_output