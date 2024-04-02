import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, dModel, dropout=0.1, maxLen=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Positional Encoding Shape (maxLen, dModel)
        pe = torch.zeros(maxLen, dModel)
        
        position = torch.arange(0, maxLen).unsqueeze(1)
        
        # Just math here
        divTerm = torch.exp(
            torch.arange(0, dModel, 2) * -(math.log(10000.0) / dModel)
        )
        # PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * divTerm)
        # PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * divTerm)
        
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x):
        # x + positional encoding
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, vocabSize, embeddingDim, numClass, feedforwardDim=256, numHead=2, numLayers=3, dropout=0.1, maxLength=128):
        super(Transformer, self).__init__()
        # Embedding Layer
        self.embedding = nn.Embedding(vocabSize, embeddingDim)
        # Position Encoding Layer
        self.positionalEncoding = PositionalEncoding(embeddingDim, dropout, maxLength)
        # Encoding Layer
        self.encoderLayer = nn.TransformerEncoderLayer(embeddingDim, numHead, feedforwardDim, dropout)
        # Transformer
        self.transformer = nn.TransformerEncoder(self.encoderLayer, numLayers)
        # Output Layer
        self.fc = nn.Linear(embeddingDim, numClass)
    
    def forward(self, x):
        x = x.transpose(0, 1) 
        x = self.embedding(x)
        x = self.positionalEncoding(x)
        x = self.transformer(x)
        x = x.mean(axis=0)
        x = self.fc(x)
        return x