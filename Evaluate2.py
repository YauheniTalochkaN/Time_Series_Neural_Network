#!/usr/bin/env python

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from Model2 import TimeSeriesTransformer, parse_args

# Prediction method
def predict_future(model, initial_sequence, n_steps, device, pred_len):
    model.eval()
    
    predictions = []
    
    current_sequence = initial_sequence.clone().detach().to(device)  # [seq_len]
    current_sequence = current_sequence.unsqueeze(-1).float()  # [seq_len, 1]
    current_sequence = current_sequence.unsqueeze(0)  # [1, seq_len, 1]
    
    with torch.no_grad():
        num_iterations = (n_steps + pred_len - 1) // pred_len
        
        for _ in range(num_iterations):
            prediction = model(current_sequence)  # [1, pred_len, 1]
            
            predictions.extend(prediction.squeeze(0).squeeze(-1).cpu().numpy())

            current_sequence = torch.cat((current_sequence.squeeze(0)[pred_len:], prediction.squeeze(0)), dim=0).unsqueeze(0)  # [1, seq_len, 1]
    
    return predictions[:n_steps]

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
    model_load_path = parameters['model_save_path']

    # Generate test data
    data_length_test = 400
    data_test = torch.sin(torch.linspace(0., 28.0, data_length_test)) + 0.1 * torch.FloatTensor(data_length_test).uniform_(-1.0, 1.0)
    smooth_data = np.sin(np.linspace(0., 28.0, data_length_test))

    # How many steps do you want to predict
    n_steps_ahead = data_length_test - seq_len

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device, "device")
    
    # Load the model
    model = TimeSeriesTransformer(input_dim=input_dim,
                                  d_model=d_model,
                                  num_heads=num_heads,
                                  dim_feedforward=dim_feedforward,
                                  num_encoder_layers=num_layers,
                                  num_decoder_layers=num_layers,
                                  output_dim=output_dim,
                                  pred_len=pred_len).to(device)
    
    model.load_state_dict(torch.load(model_load_path, map_location=device, weights_only=True))
    
    print(f"The model was uploaded from {model_load_path}")
    
    # Initial sequence
    initial_sequence_test = data_test[:seq_len]
    
    # Make predictions
    predicted_values = predict_future(model=model,
                                      initial_sequence=initial_sequence_test.to(device),
                                      n_steps=n_steps_ahead,
                                      device=device,
                                      pred_len=pred_len)
    
    # Smooth data for the prediction range
    true_values_future = smooth_data[seq_len:seq_len + n_steps_ahead]

    # Plot the results 
    plt.close('all')
    
    plt.figure(figsize=(10,6))
    
    plt.plot(range(seq_len), initial_sequence_test.numpy(), label="Initial Sequence", color="blue")
    
    plt.plot(range(seq_len, seq_len + n_steps_ahead), true_values_future,
             label="True Future Values (smooth function)", color="green")
    
    plt.plot(range(seq_len, seq_len + n_steps_ahead), predicted_values,
             label="Predicted Future Values", color="red", linestyle="dashed")
    
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    
    plt.legend(loc='lower left')
    
    plt.title("True vs Predicted Future Values")
    
    plt.show()


if __name__ == "__main__":
    main()