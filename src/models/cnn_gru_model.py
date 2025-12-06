"""
CNN-GRU Hybrid Model for Academic Performance Prediction.
Combines 1D CNN for local pattern extraction with GRU for temporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNGRU(nn.Module):
    """
    Hybrid CNN-GRU model for student performance prediction.

    Architecture:
        1. 1D CNN layers for local pattern extraction
        2. GRU layers for temporal sequence modeling
        3. Fully connected classifier head
    """

    def __init__(self, input_dim: int, cnn_channels: list, kernel_sizes: list,
                 gru_hidden_dim: int, gru_num_layers: int, num_classes: int,
                 dropout: float = 0.3, bidirectional: bool = True):
        """
        Initialize the CNN-GRU model.

        Args:
            input_dim: Number of input features
            cnn_channels: List of CNN output channels for each layer
            kernel_sizes: List of kernel sizes for each CNN layer
            gru_hidden_dim: Hidden dimension of GRU
            gru_num_layers: Number of GRU layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
        """
        super(CNNGRU, self).__init__()

        self.input_dim = input_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_num_layers = gru_num_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes

        # CNN layers for local pattern extraction
        self.cnn_layers = nn.ModuleList()
        in_channels = 1  # Start with 1 channel (treating features as single channel)

        for out_channels, kernel_size in zip(cnn_channels, kernel_sizes):
            self.cnn_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels

        # The CNN will transform input_dim features into cnn_channels[-1] features
        self.cnn_output_dim = cnn_channels[-1]

        # Projection layer to combine features after CNN
        self.feature_projection = nn.Linear(input_dim, self.cnn_output_dim)

        # GRU for temporal modeling
        gru_input_dim = self.cnn_output_dim * 2  # CNN output + original features projected
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Calculate GRU output dimension
        gru_output_dim = gru_hidden_dim * (2 if bidirectional else 1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim, gru_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_output_dim // 2, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, input_dim = x.size()

        # CNN expects (batch_size, channels, sequence_length)
        # We'll process each feature dimension through CNN
        # Reshape: (batch_size, sequence_length, input_dim) -> (batch_size, 1, sequence_length, input_dim)
        # Then process through 1D CNN along sequence dimension

        # Transpose for CNN: (batch_size, input_dim, sequence_length)
        x_cnn = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)

        # Apply 1D convolutions across the sequence for each feature
        # Treat input_dim as channels
        cnn_out = x_cnn.unsqueeze(1)  # (batch_size, 1, input_dim, seq_len)

        # Process through CNN layers along the sequence dimension
        # Reshape to apply 1D CNN: (batch_size * input_dim, 1, seq_len)
        cnn_input = x_cnn.unsqueeze(2)  # (batch_size, input_dim, 1, seq_len)

        # Alternative approach: apply CNN to each sample's features across time
        # Permute to (batch_size, 1, seq_len, input_dim), then process
        x_for_cnn = x.unsqueeze(1)  # (batch_size, 1, seq_len, input_dim)

        # Process sequence dimension with 1D CNN
        # Reshape to (batch_size, input_dim, seq_len) for 1D CNN
        cnn_input = x.transpose(1, 2).unsqueeze(1)  # (batch_size, 1, input_dim, seq_len)

        # Better approach: treat each time step's features as a channel
        # Input shape: (batch_size, seq_len, input_dim)
        # For 1D CNN along time: (batch_size, input_dim, seq_len)
        x_transposed = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)

        # Treat all features as a single channel for simplicity
        # Reshape to (batch_size, 1, seq_len * input_dim) - not ideal

        # Better: apply CNN along sequence dimension with input_dim channels
        # But our CNN expects 1 input channel. Let's use a different approach.

        # Use multi-channel CNN approach
        # Input: (batch_size, seq_len, input_dim)
        # Reshape to (batch_size, input_dim, seq_len) for Conv1d along sequence
        x_cnn_input = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)

        # Apply CNN (first layer expects input_dim channels)
        # Modify architecture: first conv takes input_dim channels
        # For this, we'll apply CNN along sequence dimension treating features as channels

        # Simpler approach: Use projection + standard CNN
        # Apply CNN to learn patterns across the sequence dimension
        # Each time step has input_dim features

        # Let's use: (batch_size, 1, seq_len) where we process one feature at a time
        # Or: process all features together

        # Final approach: Apply 1D CNN along sequence, treating input_dim as initial channels
        # This requires modifying CNN to accept input_dim input channels

        # For the pretrained architecture with 1 input channel:
        # We'll apply CNN to each feature separately and aggregate

        # Simpler solution: Average features first or use embedding
        # Let's use mean pooling across features for CNN input
        x_mean = x.mean(dim=2, keepdim=True)  # (batch_size, seq_len, 1)
        x_mean = x_mean.transpose(1, 2)  # (batch_size, 1, seq_len)

        # Apply CNN layers
        cnn_out = x_mean
        for cnn_layer in self.cnn_layers:
            cnn_out = cnn_layer(cnn_out)  # (batch_size, channels, seq_len)

        # Transpose back: (batch_size, seq_len, channels)
        cnn_out = cnn_out.transpose(1, 2)  # (batch_size, seq_len, cnn_output_dim)

        # Project original features to same dimension
        x_projected = self.feature_projection(x)  # (batch_size, seq_len, cnn_output_dim)

        # Concatenate CNN output with projected features
        gru_input = torch.cat([cnn_out, x_projected], dim=2)  # (batch_size, seq_len, gru_input_dim)

        # GRU forward pass
        gru_out, _ = self.gru(gru_input)  # (batch_size, seq_len, gru_output_dim)

        # Use the last time step output
        last_output = gru_out[:, -1, :]  # (batch_size, gru_output_dim)

        # Classify
        logits = self.classifier(last_output)  # (batch_size, num_classes)

        return logits

    def get_feature_maps(self, x):
        """
        Extract intermediate feature maps for visualization/analysis.

        Args:
            x: Input tensor

        Returns:
            Dictionary with CNN and GRU outputs
        """
        # This is similar to forward but returns intermediate outputs
        x_mean = x.mean(dim=2, keepdim=True).transpose(1, 2)

        cnn_out = x_mean
        for cnn_layer in self.cnn_layers:
            cnn_out = cnn_layer(cnn_out)

        cnn_out = cnn_out.transpose(1, 2)
        x_projected = self.feature_projection(x)
        gru_input = torch.cat([cnn_out, x_projected], dim=2)

        gru_out, hidden = self.gru(gru_input)

        return {
            'cnn_output': cnn_out,
            'gru_output': gru_out,
            'gru_hidden': hidden
        }


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
