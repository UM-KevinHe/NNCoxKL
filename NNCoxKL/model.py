import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from .train_utils import set_seeds

class MLPVanilla(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_dr):
        super(MLPVanilla, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=num_dr)
        self.layer_2 = nn.Linear(hidden_dim, output_dim,bias=False)
        self.use_gpu = False
       
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        return x
    

class MLPVanillaThree(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_dr):
        super(MLPVanillaThree, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(p=num_dr)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="relu")
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(p=num_dr)
        self.layer_3 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.use_gpu = False
       
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = F.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = self.layer_3(x)
        return x

    

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_rate):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout_rate)
        self.transformer = nn.Transformer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers, 
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate
        )
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.use_gpu = False

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embedding(src) * math.sqrt(self.hidden_dim)
        src = self.positional_encoding(src)
        tgt = self.embedding(tgt) * math.sqrt(self.hidden_dim)
        tgt = self.positional_encoding(tgt)
        
        output = self.transformer(src, tgt, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
        output = self.output_layer(output)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, 1, hidden_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class MLPDynamic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_dr, seed_value = 42):
        set_seeds(seed_value)

        super(MLPDynamic, self).__init__()
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        nn.init.kaiming_uniform_(self.layers[0].weight, nonlinearity='relu')
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            nn.init.kaiming_uniform_(self.layers[-1].weight, nonlinearity='relu')
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)
        
        # Batch normalization and Dropout
        self.batchnorms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(p=num_dr) for _ in range(num_layers)])
        
        self.use_gpu = False  # Flag to use GPU

    def forward(self, x):
        for layer, batchnorm, dropout in zip(self.layers, self.batchnorms, self.dropouts):
            x = F.selu(layer(x))
            x = batchnorm(x)
            x = dropout(x)
        
        x = self.output_layer(x)
        return x
    



################
# class MLPDynamicEmbedding(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_dr, 
#                  num_categories, embedding_dim, seed_value=42):
#         set_seeds(seed_value)

#         super(MLPDynamic, self).__init__()
#         self.layers = nn.ModuleList()
        
#         self.embedding = nn.Embedding(num_categories, embedding_dim)
        
#         input_dim += embedding_dim - 1 
        
#         self.layers.append(nn.Linear(input_dim, hidden_dim))
#         nn.init.kaiming_uniform_(self.layers[0].weight, nonlinearity='relu')
        
#         for _ in range(num_layers - 1):
#             self.layers.append(nn.Linear(hidden_dim, hidden_dim))
#             nn.init.kaiming_uniform_(self.layers[-1].weight, nonlinearity='relu')
        
#         self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)
        
#         self.batchnorms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
#         self.dropouts = nn.ModuleList([nn.Dropout(p=num_dr) for _ in range(num_layers)])
        
#         self.use_gpu = False  # Flag to use GPU

#     def forward(self, x_cat, x_cont):        
#         x_cat = self.embedding(x_cat).view(x_cat.size(0), -1)
#         x = torch.cat([x_cont, x_cat], dim=1)
        
#         for layer, batchnorm, dropout in zip(self.layers, self.batchnorms, self.dropouts):
#             x = F.relu(layer(x))
#             x = batchnorm(x)
#             x = dropout(x)
        
#         x = self.output_layer(x)
#         return x