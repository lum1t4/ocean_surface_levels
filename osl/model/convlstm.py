"""
ConvLSTM-based video prediction model for ocean surface prediction.
Based on "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
"""

import torch
import torch.nn as nn
from typing import List
from dataclasses import dataclass
from pydantic import BaseModel

class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell."""
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )


class ConvLSTM(nn.Module):
    """Multi-layer Convolutional LSTM."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        kernel_sizes: List[int],
        num_layers: int,
        batch_first: bool = True,
        bias: bool = True,
        return_all_layers: bool = False
    ):
        super().__init__()
        
        self._check_consistency(hidden_dims, kernel_sizes, num_layers)
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dims[i],
                    kernel_size=self.kernel_sizes[i],
                    bias=self.bias
                )
            )
        
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if self.batch_first:
            # (b, t, c, h, w) -> (t, b, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        seq_len, b, _, h, w = input_tensor.size()
        
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, height=h, width=w, device=input_tensor.device)
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[t, :, :, :, :], cur_state=[h, c])
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=0)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, height, width, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, height, width, device))
        return init_states

    @staticmethod
    def _check_consistency(hidden_dims, kernel_sizes, num_layers):
        if len(hidden_dims) != num_layers:
            raise ValueError("Inconsistent list length.")
        if len(kernel_sizes) != num_layers:
            raise ValueError("Inconsistent list length.")




class OSPConfig(BaseModel):
    input_channels: int = 3  # sla, ugos, vgos
    hidden_dims: List[int] = [64, 128, 128, 64]
    kernel_sizes: List[int] = [3, 3, 3, 3]
    num_layers: int = 4
    seq_length: int = 3


class OceanSurfacePredictorConvLSTM(nn.Module):
    """
    Ocean surface predictor using ConvLSTM architecture.
    Takes a sequence of ocean states and predicts future states.
    """
    
    def __init__(self, config: OSPConfig):
        super().__init__()
        
        self.input_channels = config.input_channels
        self.seq_length = config.seq_length
        
        # Encoder ConvLSTM
        self.encoder = ConvLSTM(
            input_dim=config.input_channels,
            hidden_dims=config.hidden_dims,
            kernel_sizes=config.kernel_sizes,
            num_layers=config.num_layers,
            batch_first=True,
            return_all_layers=False
        )
        
        # Decoder ConvLSTM for prediction
        self.decoder = ConvLSTM(
            input_dim=config.hidden_dims[-1],  # Use last hidden dim from encoder
            hidden_dims=config.hidden_dims[::-1],  # Reverse hidden dims for decoder
            kernel_sizes=config.kernel_sizes[::-1],
            num_layers=config.num_layers,
            batch_first=True,
            return_all_layers=False
        )
        
        # Output projection
        self.output_conv = nn.Conv2d(
            config.hidden_dims[0],  # First element of reversed hidden_dims
            config.input_channels,
            kernel_size=1
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_length, channels, height, width)

        Returns:
            Predicted tensor of shape (batch, seq_length, channels, height, width)
        """
        batch_size, seq_len, _, height, width = x.shape

        # Encode input sequence
        _, _ = self.encoder(x)

        # Initialize decoder input with zeros (predict same length as input)
        decoder_input = torch.zeros(
            batch_size, seq_len, self.encoder.hidden_dims[-1], height, width,
            device=x.device, dtype=x.dtype
        )

        # Decode to generate predictions
        decoder_output, _ = self.decoder(decoder_input, None)

        # Reshape for output projection
        decoder_output = decoder_output[0]  # Get last layer output
        # decoder_output is (seq_len, batch, channels, h, w) due to ConvLSTM internals
        decoder_output = decoder_output.permute(1, 0, 2, 3, 4)  # (batch, seq_len, channels, h, w)

        # Project to output channels
        output = []
        for t in range(seq_len):
            out_t = self.output_conv(decoder_output[:, t])  # Apply conv to each timestep
            output.append(out_t)

        output = torch.stack(output, dim=1)  # (batch, seq_length, channels, height, width)

        return output


if __name__ == "__main__":
    # Test the model
    config = OSPConfig(seq_length=3)
    model = OceanSurfacePredictorConvLSTM(config)

    # Test input
    x = torch.randn(2, 3, 3, 224, 224)  # (batch, seq_len, channels, height, width)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")  # Should be (2, 3, 3, 224, 224)