import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        
        self.d_in = d_in
        self.d_out = d_out
        
        self.ff = nn.Linear(d_in, 3*d_out)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        ff_out = self.ff(x)                                                     # (batch_size, seq_len, 3 * d_out)
        ff_out = ff_out.view(batch_size, seq_len, 3, -1)                        # (batch_size, seq_len, 3, d_out)

        # Extract query, key, value
        query = ff_out[:,:,0,:].view(batch_size, seq_len, -1)                   # (batch_size, seq_len, d_out)
        key = ff_out[:,:,1,:].view(batch_size, seq_len, -1)                     # (batch_size, seq_len, d_out)
        value = ff_out[:,:,2,:].view(batch_size, seq_len, -1)                   # (batch_size, seq_len, d_out)   

        # Compute attention score
        attn_scores = query @ key.transpose(-2, -1)                             # (batch_size, seq_len, d_out) @ (batch_size, d_out, seq_len) -> (batch_size, seq_len, seq_len)
        
        # Normalize score to weights
        attn_weights = torch.softmax(attn_scores / (self.d_out ** 0.5), dim=-1) # (batch_size, seq_len, seq_len)

        # Drop some weights if dropout > 0
        attn_weights = self.dropout(attn_weights)                               # (batch_size, seq_len, seq_len)

        # Compute weighted sum of values
        output = attn_weights @ value                                           # (batch_size, seq_len, seq_len) @ (batch_size, seq_len, d_out) -> (batch_size, seq_len, d_out)
        
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads

        self.ff = nn.Linear(d_in, num_heads*3*d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        ff_out = self.ff(x)                                                         # (batch_size, seq_len, num_heads * 3 * d_out)
        ff_out = ff_out.view(batch_size, seq_len, self.num_heads, 3, -1)            # (batch_size, seq_len, num_heads, 3, d_out)
        ff_out = ff_out.permute(0, 2, 1, 3, 4)                                      # (batch_size, num_heads, seq_len, 3, d_out)

        # Extract query, key, value
        query = ff_out[:,:,:,0,:].view(batch_size, self.num_heads, seq_len, -1)     # (batch_size, num_heads, seq_len, d_out)
        key = ff_out[:,:,:,1,:].view(batch_size, self.num_heads, seq_len, -1)       # (batch_size, num_heads, seq_len, d_out)
        value = ff_out[:,:,:,2,:].view(batch_size, self.num_heads, seq_len, -1)     # (batch_size, num_heads, seq_len, d_out)

        # Compute attention score
        attn_scores = query @ key.transpose(-2, -1)                                 # (batch_size, num_heads, seq_len, d_out) @ (batch_size, num_heads, d_out, seq_len) -> (batch_size, num_heads, seq_len, seq_len)

        # Normalize score to weights
        attn_weights = torch.softmax(attn_scores / (self.d_out ** 0.5), dim=-1)     # (batch_size, num_heads, seq_len, seq_len)
        
        # Drop some weights if dropout > 0
        attn_weights = self.dropout(attn_weights)                                   # (batch_size, num_heads, seq_len, seq_len)

        # Compute weighted sum of values
        output = attn_weights @ value                                               # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, d_out) -> (batch_size, num_heads, seq_len, d_out)
        output = output.permute(0, 2, 1, 3)                                         # (batch_size, seq_len, num_heads, d_out)
        output = output.view(batch_size, seq_len, -1)                               # (batch_size, seq_len, num_heads * d_out)

        return output

class CausalSelfAttention():
    def __init__(self):
        pass

class CausalMultiHeadAttention():
    def __init__(self):
        pass

class GalerkinAttention():
    def __init__(self):
        pass