import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn

class TabTransformer(nn.Module):
    def __init__(self, categories, num_continuous, d=32, n_layers=6, n_heads=8):
        super().__init__()
        # Based on paper specs: d=32, layers=6, heads=8 
        self.num_categories = len(categories)
        self.embeddings = nn.ModuleList([nn.Embedding(num_classes, d) for num_classes in categories])
        
        # Unique identifier 'c' to distinguish column classes 
        self.column_id = nn.Parameter(torch.randn(1, self.num_categories, d))
        
        # Transformer Stack for Contextual Embeddings 
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, batch_first=True) 
            for _ in range(n_layers)
        ])
        
        # MLP Top Layer (g) 
        # Input dim = (Number of Categories * d) + Number of Continuous 
        input_dim = (self.num_categories * d) + num_continuous
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128), # Layer size 4xl as suggested 
            nn.ReLU(),
            nn.Linear(128, 64),  # Layer size 2xl as suggested 
            nn.ReLU(),
            nn.Linear(64, 1),    # Output: Probability of Target
            nn.Sigmoid()
        )

    def forward(self, x_cat, x_cont):
        # 1. Column Embedding
        x = [embed(x_cat[:, i]) for i, embed in enumerate(self.embeddings)]
        x = torch.stack(x, dim=1) + self.column_id
        
        # 2. Transformer Contextualization
        for layer in self.transformer:
            x = layer(x)
        
        # 3. Concatenation with Continuous Features 
        x_contextual = x.flatten(1)
        combined = torch.cat([x_contextual, x_cont], dim=1)
        
        # 4. Final Prediction
        return self.mlp(combined)