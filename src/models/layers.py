"""
Custom Layers for FaultFormer
=============================

Contains:
- RotaryEmbedding (RoPE): Rotary Positional Embeddings
- TransformerEncoderBlock: Pre-norm Transformer with attention
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class RotaryEmbedding(layers.Layer):
    """
    Rotary Positional Embedding (RoPE) layer.
    Rotates Query and Key vectors to encode relative position information.
    
    Reference: RoFormer (Su et al., 2021)
    """
    def __init__(self, dim, max_wavelength=10000, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_wavelength = max_wavelength

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, start_index=0):
        # x shape: [batch, seq_len, heads, dim_head]
        seq_len = tf.shape(x)[1]
        
        # Create position indices
        positions = tf.range(start_index, start_index + seq_len, dtype=tf.float32)
        
        # Calculate frequencies
        dim = tf.cast(self.dim, tf.float32)
        indices = tf.range(0, self.dim, 2, dtype=tf.float32)
        theta = 1.0 / (self.max_wavelength ** (indices / dim))
        
        # Create angles: pos * theta
        freqs = tf.einsum('i,j->ij', positions, theta)
        
        # Create sin and cos embeddings
        emb = tf.concat([freqs, freqs], axis=-1)
        cos_emb = tf.cos(emb)
        sin_emb = tf.sin(emb)
        
        # Reshape for broadcasting
        cos_emb = cos_emb[tf.newaxis, :, tf.newaxis, :]
        sin_emb = sin_emb[tf.newaxis, :, tf.newaxis, :]
        
        # Apply rotation
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        x_out_part1 = x1 * cos_emb[..., :self.dim//2] - x2 * sin_emb[..., :self.dim//2]
        x_out_part2 = x1 * sin_emb[..., :self.dim//2] + x2 * cos_emb[..., :self.dim//2]
        
        # Interleave back
        x_out = tf.stack([x_out_part1, x_out_part2], axis=-1)
        x_out = tf.reshape(x_out, tf.shape(x))
        
        return x_out


class TransformerEncoderBlock(layers.Layer):
    """
    Transformer Encoder Block with Pre-LayerNorm.
    
    Architecture:
        Input -> LayerNorm -> MultiHeadAttention -> Residual
              -> LayerNorm -> FFN -> Residual -> Output
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.head_dim)
        self.rope = RotaryEmbedding(self.head_dim)
        
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # Pre-LayerNorm
        x_norm = self.layernorm1(inputs)
        
        # Multi-Head Self-Attention
        attn_output = self.att(x_norm, x_norm)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output  # Residual connection
        
        # Feed Forward Network
        x_norm2 = self.layernorm2(out1)
        ffn_output = self.ffn(x_norm2)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        return out1 + ffn_output  # Residual connection
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ffn.layers[0].units,
            "dropout_rate": self.dropout1.rate,
        })
        return config
