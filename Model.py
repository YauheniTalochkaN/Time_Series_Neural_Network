#!/usr/bin/env python

import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Custom CNN-LSTM class
class TimeSeriesCNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, kernel_size=3, dropout=0.1):
        super(TimeSeriesCNNLSTM, self).__init__()
        
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv_out = self.conv(x)
        
        conv_out = conv_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(conv_out)
        
        return self.lin(lstm_out[:, -1, :])

# Custom dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        src_seq = self.data[idx:idx + self.seq_len]
        tgt_seq = self.data[idx + self.seq_len]
        
        return torch.tensor(src_seq).unsqueeze(-1), torch.tensor(tgt_seq)

# Train method
def train(model, dataloader, criterion, optimizer, device, epochs):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for src_batch, tgt_batch in dataloader:
            src_batch = src_batch.float().to(device)
            tgt_batch = tgt_batch.float().to(device)
            
            output = model(src_batch)
            
            loss = criterion(output.squeeze(), tgt_batch)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.5f}")

# Parse method
def parse_args():
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/training_parameters.yaml')
    return parser.parse_args()

def main():
    # Get args
    args = parse_args()

    # Open the file of training parameters
    with open(args.config, 'r') as f:
        parameters = yaml.safe_load(f)

    # Obtain necessary parameters
    input_dim = parameters['input_dim']
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layers']
    output_dim = parameters['output_dim']
    seq_len = parameters['seq_len']
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
    dataset = TimeSeriesDataset(data.numpy(), seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Build the model
    model = TimeSeriesCNNLSTM(input_dim=input_dim,
                              hidden_dim=hidden_dim,
                              num_layers=num_layers,
                              output_dim=output_dim).to(device)

    # Set loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

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