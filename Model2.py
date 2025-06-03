#!/usr/bin/env python

import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Custom Time Series Transformer class
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, dim_feedforward, 
                 num_encoder_layers, num_decoder_layers, output_dim, 
                 pred_len=1, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.pred_len = pred_len 
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.tgt_projection = nn.Linear(output_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=num_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                   nhead=num_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.output_projection = nn.Linear(d_model, output_dim)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt=None):
        src = self.input_projection(src)
        src = self.positional_encoding(src)

        memory = self.transformer_encoder(src)

        if tgt is None:
            tgt = torch.zeros((src.size(0), 1, self.tgt_projection.in_features), device=src.device)
            outputs = []

            for _ in range(self.pred_len):
                tgt_embed = self.tgt_projection(tgt)
                tgt_embed = self.positional_encoding(tgt_embed)

                tgt_mask = self.generate_square_subsequent_mask(tgt_embed.size(1)).to(src.device)

                output = self.transformer_decoder(tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask)

                next_step = self.output_projection(output[:, -1:, :])
                outputs.append(next_step)

                tgt = torch.cat([tgt, next_step], dim=1)

            return torch.cat(outputs, dim=1)

        else:
            tgt = self.tgt_projection(tgt)
            tgt = self.positional_encoding(tgt)

            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(src.device)

            output = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)

            return self.output_projection(output)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Custom dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len=1):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        src_start = idx
        src_end = src_start + self.seq_len
        
        tgt_start = src_end - 1
        tgt_end = tgt_start + self.pred_len
        
        target_start = src_end
        target_end = target_start + self.pred_len
        
        src_seq = self.data[src_start:src_end]  # [seq_len]
        tgt_seq = self.data[tgt_start:tgt_end]  # [pred_len]
        target_seq = self.data[target_start:target_end]  # [pred_len]
        
        return (torch.tensor(src_seq).unsqueeze(-1),  # [seq_len, 1]
                torch.tensor(tgt_seq).unsqueeze(-1),  # [pred_len, 1]
                torch.tensor(target_seq))             # [pred_len]

# Train method
def train(model, dataloader, criterion, optimizer, device, epochs):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for src_batch, tgt_batch, target_batch in dataloader:
            src_batch = src_batch.float().to(device)  # [batch, seq_len, 1]
            tgt_batch = tgt_batch.float().to(device)  # [batch, pred_len, 1]
            target_batch = target_batch.float().to(device)  # [batch, pred_len]
            
            output = model(src_batch, tgt_batch)  # [batch, pred_len, 1]
            
            loss = criterion(output.squeeze(-1), target_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.5f}")

# Parse method
def parse_args():
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/training_parameters2.yaml')
    return parser.parse_args()

def main():
    # Get args
    args = parse_args()

    # Open the file of training parameters
    with open(args.config, 'r') as f:
        parameters = yaml.safe_load(f)

    # Obtain necessary parameters
    input_dim = parameters['input_dim']
    d_model = parameters['d_model']
    dim_feedforward = parameters['dim_feedforward']
    num_heads = parameters['num_heads']
    num_layers = parameters['num_layers']
    output_dim = parameters['output_dim']
    seq_len = parameters['seq_len']
    pred_len = parameters['pred_len']
    batch_size = parameters['batch_size']
    epochs = parameters['epochs']
    model_save_path = parameters['model_save_path']

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device, "device")
    
    # Generate train data
    data_length = 1000
    data = torch.sin(torch.linspace(0.0, 70.0, data_length)) + 0.1 * torch.FloatTensor(data_length).uniform_(-1.0, 1.0)
    
    # Set custom dataset
    dataset = TimeSeriesDataset(data.numpy(), seq_len=seq_len, pred_len=pred_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Build the model
    model = TimeSeriesTransformer(input_dim=input_dim,
                                  d_model=d_model,
                                  num_heads=num_heads,
                                  dim_feedforward=dim_feedforward,
                                  num_encoder_layers=num_layers,
                                  num_decoder_layers=num_layers,
                                  output_dim=output_dim,
                                  pred_len=pred_len).to(device)

    # Set loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, amsgrad=True)

    # Тrain the model
    train(model=model,
          dataloader=dataloader,
          criterion=criterion,
          optimizer=optimizer,
          device=device,
          epochs=epochs)
    
    # Save model
    torch.save(model.state_dict(), model_save_path)
    print(f"The model was saved at {model_save_path}")


if __name__ == "__main__":
    main()