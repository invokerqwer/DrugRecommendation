import torch
import torch.nn as nn
import torch.nn.functional as F

class VQVAE(nn.Module):
    def __init__(self, input_key_dim, input_value_dim, codebook_size, hidden_size,alpha):
        super(VQVAE, self).__init__()
        self.input_key_dim = input_key_dim
        self.input_value_dim = input_value_dim
        self.hidden_key_size = hidden_size
        self.hidden_value_size = input_value_dim
        self.codebook_size = codebook_size
        self.alpha = alpha
        # Encoder and Decoder for `history_keys`
        self.encoder_keys = nn.Linear(input_key_dim, hidden_size)
        self.decoder_keys = nn.Linear(hidden_size, input_key_dim)
        
        # Encoder and Decoder for `history_values`
        self.encoder_values = nn.Linear(input_value_dim, input_value_dim)
        self.decoder_values = nn.Linear(input_value_dim, input_value_dim)
        
        # Codebooks for both `history_keys` and `history_values`
        self.codebook_keys = nn.Embedding(codebook_size, hidden_size)
        self.codebook_values = nn.Embedding(codebook_size, input_value_dim)
        
        # Codebook initialization
        nn.init.uniform_(self.codebook_keys.weight, -1.0 / codebook_size, 1.0 / codebook_size)
        nn.init.uniform_(self.codebook_values.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, keys, values):
        # Encode keys and values
        encoded_keys = self.encoder_keys(keys)
        encoded_values = self.encoder_values(values)
        
        # Vector Quantization for keys
        keys_flat = encoded_keys.view(-1, self.hidden_key_size)
        key_embeddings = self.codebook_keys.weight
        key_indices = torch.argmin(torch.cdist(keys_flat, key_embeddings), dim=1)
        quantized_keys = self.codebook_keys(key_indices).view_as(encoded_keys)
        
        # Vector Quantization for values
        values_flat = encoded_values.view(-1, self.hidden_value_size)
        value_embeddings = self.codebook_values.weight
        value_indices = torch.argmin(torch.cdist(values_flat, value_embeddings), dim=1)
        quantized_values = self.codebook_values(value_indices).view_as(encoded_values)
        
        # Straight-Through Estimator for backpropagation
        quantized_keys = encoded_keys + (quantized_keys - encoded_keys).detach()
        quantized_values = encoded_values + (quantized_values - encoded_values).detach()
        
        # Decode quantized keys and values
        decoded_keys = self.decoder_keys(quantized_keys)
        decoded_values = self.decoder_values(quantized_values)
        
        return decoded_keys, decoded_values, encoded_keys, encoded_values, quantized_keys, quantized_values

    def compute_loss(self, keys, values, decoded_keys, decoded_values, quantized_keys, quantized_values):
        reconstruction_loss = F.mse_loss(decoded_keys, keys) + F.mse_loss(decoded_values, values)
        vq_loss = F.mse_loss(keys.detach(), quantized_keys) + F.mse_loss(values.detach(), quantized_values)
        commitment_loss = F.mse_loss(quantized_keys.detach(), keys) + F.mse_loss(quantized_values.detach(), values)
        total_loss = reconstruction_loss + vq_loss + self.alpha * commitment_loss
        return total_loss