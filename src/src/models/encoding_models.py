```python
"""
Neural encoding models for retinal ganglion cells.
Implements Linear-Nonlinear (LN), Generalized Linear Model (GLM), and CNN approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class LinearNonlinearModel(nn.Module):
    """
    Linear-Nonlinear (LN) model for neural encoding.
    
    Architecture:
    1. Linear receptive field convolution
    2. Nonlinear activation (exponential for Poisson)
    3. Optional temporal dynamics
    """
    
    def __init__(
        self,
        stimulus_shape: Tuple[int, int, int],  # (time, height, width)
        nonlinearity: str = 'exp',
        temporal_dynamics: bool = False,
        regularization: float = 1e-4
    ):
        super().__init__()
        
        self.stimulus_shape = stimulus_shape
        self.n_time_bins, self.height, self.width = stimulus_shape
        self.nonlinearity = nonlinearity
        self.temporal_dynamics = temporal_dynamics
        self.regularization = regularization
        
        # Spatial receptive field
        self.spatial_rf = nn.Conv2d(
            in_channels=1,
            out_channels=1, 
            kernel_size=(self.height, self.width),
            bias=True
        )
        
        # Temporal filter
        self.temporal_filter = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.n_time_bins,
            bias=False
        )
        
        # Optional temporal dynamics (feedback)
        if temporal_dynamics:
            self.history_filter = nn.Conv1d(1, 1, kernel_size=10, bias=False)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with appropriate distributions."""
        # Initialize spatial RF with Gaussian
        with torch.no_grad():
            center_h, center_w = self.height // 2, self.width // 2
            y, x = torch.meshgrid(torch.arange(self.height), torch.arange(self.width), indexing='ij')
            gaussian = torch.exp(-((x - center_w)**2 + (y - center_h)**2) / (2 * (min(self.height, self.width) / 4)**2))
            self.spatial_rf.weight.data[0, 0] = gaussian * (torch.randn_like(gaussian) * 0.1 + 1)
            
        # Initialize temporal filter
        nn.init.normal_(self.temporal_filter.weight, std=0.1)
        
        if self.temporal_dynamics:
            nn.init.normal_(self.history_filter.weight, std=0.01)
    
    def forward(self, stimulus: torch.Tensor, history: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through LN model.
        
        Args:
            stimulus: Input stimulus (batch, time, height, width)
            history: Previous spike history for temporal dynamics
            
        Returns:
            Predicted firing rate (batch, 1)
        """
        batch_size = stimulus.shape[0]
        
        # Apply spatial receptive field to each time frame
        spatial_responses = []
        for t in range(self.n_time_bins):
            frame = stimulus[:, t:t+1, :, :]  # (batch, 1, height, width)
            response = self.spatial_rf(frame)  # (batch, 1, 1, 1)
            spatial_responses.append(response.squeeze(-1).squeeze(-1))  # (batch, 1)
        
        spatial_output = torch.stack(spatial_responses, dim=-1)  # (batch, 1, time)
        
        # Apply temporal filter
        linear_response = self.temporal_filter(spatial_output)  # (batch, 1, 1)
        linear_response = linear_response.squeeze(-1)  # (batch, 1)
        
        # Add history term if using temporal dynamics
        if self.temporal_dynamics and history is not None:
            history_term = self.history_filter(history.unsqueeze(1)).squeeze(-1)
            linear_response = linear_response + history_term
        
        # Apply nonlinearity
        if self.nonlinearity == 'exp':
            firing_rate = torch.exp(linear_response)
        elif self.nonlinearity == 'softplus':
            firing_rate = F.softplus(linear_response)
        elif self.nonlinearity == 'relu':
            firing_rate = F.relu(linear_response)
        else:
            firing_rate = linear_response
            
        return firing_rate
    
    def get_receptive_field(self) -> torch.Tensor:
        """Extract the learned spatial receptive field."""
        return self.spatial_rf.weight.data[0, 0]
    
    def get_temporal_filter(self) -> torch.Tensor:
        """Extract the learned temporal filter."""
        return self.temporal_filter.weight.data[0, 0]


class GeneralizedLinearModel(nn.Module):
    """
    Generalized Linear Model (GLM) for neural encoding.
    Includes spike history terms and more flexible architectures.
    """
    
    def __init__(
        self,
        stimulus_shape: Tuple[int, int, int],
        history_length: int = 20,
        coupling_cells: Optional[int] = None,
        hidden_dims: Optional[list] = None
    ):
        super().__init__()
        
        self.stimulus_shape = stimulus_shape
        self.n_time_bins, self.height, self.width = stimulus_shape
        self.history_length = history_length
        self.coupling_cells = coupling_cells
        
        # Stimulus filter
        self.stimulus_filter = nn.Conv3d(
            in_channels=1,
            out_channels=8,
            kernel_size=(self.n_time_bins, 5, 5),
            padding=(0, 2, 2)
        )
        
        # Calculate flattened size after convolution
        conv_output_size = 8 * self.height * self.width
        
        # Spike history filter
        self.history_filter = nn.Linear(history_length, 32)
        
        # Cell coupling (if specified)
        if coupling_cells:
            self.