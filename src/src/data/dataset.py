```python
"""
Data loading and preprocessing for retinal ganglion cell recordings.
Handles CRCNS ret-1 dataset format and creates PyTorch datasets.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict
import warnings
from pathlib import Path


class RetinalDataset(Dataset):
    """
    PyTorch dataset for retinal ganglion cell spike data.
    
    Args:
        data_path: Path to HDF5 file containing stimulus and response data
        cell_ids: List of cell IDs to include (None for all cells)
        time_window: Duration of stimulus window in seconds
        dt: Time bin size in seconds
        normalize_stimulus: Whether to normalize stimulus to [-1, 1]
        train: Whether this is training data (affects data augmentation)
    """
    
    def __init__(
        self,
        data_path: str,
        cell_ids: Optional[List[int]] = None,
        time_window: float = 0.5,
        dt: float = 0.01,
        normalize_stimulus: bool = True,
        train: bool = True
    ):
        self.data_path = Path(data_path)
        self.time_window = time_window
        self.dt = dt
        self.normalize_stimulus = normalize_stimulus
        self.train = train
        
        # Load data
        self.stimulus, self.spikes, self.cell_info = self._load_data()
        
        # Filter cells if specified
        if cell_ids is not None:
            self._filter_cells(cell_ids)
            
        # Create time bins
        self.n_time_bins = int(time_window / dt)
        
        # Prepare stimulus-response pairs
        self._prepare_data()
        
    def _load_data(self) -> Tuple[np.ndarray, Dict, Dict]:
        """Load stimulus and spike data from HDF5 file."""
        with h5py.File(self.data_path, 'r') as f:
            # Load stimulus (assuming white noise checkerboard)
            stimulus = np.array(f['stimulus/data'])  # Shape: (time, height, width)
            
            # Load spike data for all cells
            spikes = {}
            cell_info = {}
            
            for cell_id in f['spikes'].keys():
                spike_times = np.array(f[f'spikes/{cell_id}/times'])
                cell_type = f[f'spikes/{cell_id}'].attrs.get('cell_type', 'unknown')
                
                spikes[int(cell_id)] = spike_times
                cell_info[int(cell_id)] = {'cell_type': cell_type}
                
        return stimulus, spikes, cell_info
    
    def _filter_cells(self, cell_ids: List[int]):
        """Keep only specified cells."""
        filtered_spikes = {cid: self.spikes[cid] for cid in cell_ids if cid in self.spikes}
        filtered_info = {cid: self.cell_info[cid] for cid in cell_ids if cid in self.cell_info}
        
        self.spikes = filtered_spikes
        self.cell_info = filtered_info
        
    def _prepare_data(self):
        """Create stimulus-response pairs with temporal windows."""
        self.data_pairs = []
        
        # Get stimulus dimensions
        T, H, W = self.stimulus.shape
        
        # Convert spike times to binned responses
        max_time = T * self.dt  # Assuming stimulus dt matches our dt
        time_bins = np.arange(0, max_time, self.dt)
        
        for cell_id, spike_times in self.spikes.items():
            # Bin spikes
            spike_counts, _ = np.histogram(spike_times, bins=time_bins)
            
            # Create sliding windows
            for t in range(self.n_time_bins, len(spike_counts)):
                # Stimulus window (past frames leading to current response)
                stim_start = max(0, t - self.n_time_bins)
                stim_window = self.stimulus[stim_start:t]  # Shape: (n_time_bins, H, W)
                
                # Pad if necessary
                if stim_window.shape[0] < self.n_time_bins:
                    pad_size = self.n_time_bins - stim_window.shape[0]
                    padding = np.zeros((pad_size, H, W))
                    stim_window = np.concatenate([padding, stim_window], axis=0)
                
                # Response (spike count in current bin)
                response = spike_counts[t]
                
                self.data_pairs.append({
                    'stimulus': stim_window.astype(np.float32),
                    'response': response.astype(np.float32),
                    'cell_id': cell_id,
                    'time_idx': t
                })
        
        if self.normalize_stimulus:
            self._normalize_stimuli()
            
    def _normalize_stimuli(self):
        """Normalize stimuli to [-1, 1] range."""
        all_stimuli = np.array([pair['stimulus'] for pair in self.data_pairs])
        self.stim_mean = np.mean(all_stimuli)
        self.stim_std = np.std(all_stimuli)
        
        for pair in self.data_pairs:
            pair['stimulus'] = (pair['stimulus'] - self.stim_mean) / self.stim_std
    
    def __len__(self) -> int:
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.data_pairs[idx]
        
        return {
            'stimulus': torch.FloatTensor(pair['stimulus']),
            'response': torch.FloatTensor([pair['response']]),
            'cell_id': pair['cell_id'],
            'time_idx': pair['time_idx']
        }
    
    @property
    def stimulus_shape(self) -> Tuple[int, int, int]:
        """Returns (n_time_bins, height, width)."""
        if len(self.data_pairs) > 0:
            return self.data_pairs[0]['stimulus'].shape
        return (0, 0, 0)
    
    @property 
    def n_cells(self) -> int:
        """Number of cells in dataset."""
        return len(self.spikes)


def create_data_loaders(
    data_path: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    time_window: float = 0.5,
    dt: float = 0.01,
    num_workers: int = 4,
    cell_ids: Optional[List[int]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_path: Path to HDF5 data file
        batch_size: Batch size for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        time_window: Duration of stimulus window
        dt: Time bin size
        num_workers: Number of data loading workers
        cell_ids: Specific cells to include
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create full dataset
    full_dataset = RetinalDataset(
        data_path=data_path,
        cell_ids=cell_ids,
        time_window=time_window,
        dt=dt,
        normalize_stimulus=True,
        train=True
    )
    
    # Split dataset
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
```
