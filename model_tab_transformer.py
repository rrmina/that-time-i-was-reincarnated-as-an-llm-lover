import torch
import torch.nn as nn
import torch.optim as optim

class TabTransformer(nn.Module):
    def __init__(self, categories, num_continuous, d=32, n_layers=4, n_heads=4, mlp_hidden=[64, 32], task='classification'):
        super().__init__()
        
        # 1. Column Embedding Layer
        self.num_categories = len(categories)
        self.embeddings = nn.ModuleList([nn.Embedding(num_classes, d) for num_classes in categories])
        
        # Unique Identifier 'c' to distinguish classes in column i from other columns
        # Shape: (1, num_categories, d)
        self.column_id = nn.Parameter(torch.randn(1, self.num_categories, d))
        
        # 2. Stack of N Transformer Layers (Multi-head self-attention + Feed-forward)
        # Using TransformerEncoderLayer as it provides the Full (Non-masked) Attention Matrix
        self.transformer_stack = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d, 
                nhead=n_heads, 
                dim_feedforward=d*4, 
                dropout=0.1, 
                batch_first=True,
                norm_first=True # Better stability for tabular data
            ) for _ in range(n_layers)
        ])
        
        # 3. Concatenation & MLP (g)
        # Contextual embeddings (m * d) + continuous features (c)
        input_dim_mlp = (self.num_categories * d) + num_continuous
        
        mlp_layers = []
        for h_dim in mlp_hidden:
            mlp_layers.append(nn.Linear(input_dim_mlp, h_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.LayerNorm(h_dim))
            input_dim_mlp = h_dim
        
        # Output layer (v)
        output_dim = 2 if task == 'classification' else 1
        mlp_layers.append(nn.Linear(mlp_hidden[-1], output_dim))
        
        self.mlp = nn.Sequential(*mlp_layers)
        self.task = task

    def forward(self, x_cat, x_cont):
        # Embed each categorical feature into dimension d
        # x_cat: (batch, num_categories)
        embeddings = [embed(x_cat[:, i]) for i, embed in enumerate(self.embeddings)]
        x = torch.stack(embeddings, dim=1) # (batch, m, d)
        
        # Add Unique Identifier (c) - this replaces positional encoding
        x = x + self.column_id 
        
        # Pass through N Transformer layers to get contextual embeddings {h1, h2, ...}
        for layer in self.transformer_stack:
            x = layer(x)
        
        # Flatten {h} and concatenate with x_cont
        h_flat = x.flatten(1) 
        combined = torch.cat([h_flat, x_cont], dim=1)
        
        # Prediction via top MLP g
        return self.mlp(combined)
