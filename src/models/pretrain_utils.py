"""
Pre-training Utilities for FaultFormer
======================================

Self-supervised learning utilities including:
- Masked Signal Modeling (MSM)
- Data augmentation for vibration signals
"""

import numpy as np
import tensorflow as tf


class MaskedSignalModeling(tf.keras.utils.Sequence):
    """
    Custom data generator for Masked Signal Modeling (MSM).
    
    Randomly masks portions of the input signal, and the model
    learns to reconstruct the original signal.
    """
    def __init__(self, data, batch_size=32, mask_ratio=0.75, shuffle=True):
        """
        Args:
            data: Input signals (N, seq_len, 1)
            batch_size: Batch size
            mask_ratio: Proportion of signal to mask (0.0 to 1.0)
            shuffle: Whether to shuffle data each epoch
        """
        self.data = data.astype(np.float32)
        self.batch_size = batch_size
        self.mask_ratio = mask_ratio
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = self.data[batch_indices].copy()
        
        # Create masked input
        masked_data = batch_data.copy()
        seq_len = batch_data.shape[1]
        num_mask = int(seq_len * self.mask_ratio)
        
        for i in range(len(batch_data)):
            mask_indices = np.random.choice(seq_len, size=num_mask, replace=False)
            masked_data[i, mask_indices, :] = 0
            
        # Input: masked signal, Target: original signal
        return masked_data, batch_data
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def get_pretraining_dataset(data, batch_size=32, mask_ratio=0.75):
    """
    Create a TensorFlow Dataset for pre-training with Masked Signal Modeling.
    
    Args:
        data: Input signals (N, seq_len, 1)
        batch_size: Batch size for training
        mask_ratio: Proportion of signal to mask
        
    Returns:
        tf.data.Dataset for training
    """
    data = data.astype(np.float32)
    seq_len = data.shape[1]
    num_mask = int(seq_len * mask_ratio)
    
    def mask_signal(signal):
        """Apply random masking to a single signal."""
        masked = tf.identity(signal)
        mask_indices = tf.random.shuffle(tf.range(seq_len))[:num_mask]
        
        # Create mask tensor
        mask = tf.ones_like(signal)
        indices = tf.expand_dims(mask_indices, 1)
        updates = tf.zeros((num_mask, 1))
        mask = tf.tensor_scatter_nd_update(mask, indices, updates)
        
        return masked * mask, signal
    
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(buffer_size=len(data))
    dataset = dataset.map(mask_signal, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def add_gaussian_noise(signal, noise_factor=0.05):
    """
    Add Gaussian noise to signal for data augmentation.
    
    Args:
        signal: Input signal tensor
        noise_factor: Standard deviation of noise
        
    Returns:
        Noisy signal
    """
    noise = tf.random.normal(shape=tf.shape(signal), stddev=noise_factor)
    return signal + noise


def time_shift(signal, shift_max=100):
    """
    Randomly shift signal in time for data augmentation.
    
    Args:
        signal: Input signal (seq_len, 1)
        shift_max: Maximum shift amount
        
    Returns:
        Shifted signal
    """
    shift = tf.random.uniform([], -shift_max, shift_max, dtype=tf.int32)
    return tf.roll(signal, shift, axis=0)
