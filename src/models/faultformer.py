"""
FaultFormer Model Architecture
==============================

Transformer-based architecture for bearing fault diagnosis with:
- Hierarchical CNN Tokenizer
- Positional Embeddings
- Transformer Encoder Stack
- Classification/Reconstruction Heads
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.models.layers import TransformerEncoderBlock


def get_cnn_tokenizer(input_shape, embed_dim):
    """
    Hierarchical CNN Tokenizer.
    Converts raw signal (2048, 1) -> Tokens (Seq_Len, Embed_Dim)
    
    Args:
        input_shape: Input signal shape (window_size, channels)
        embed_dim: Embedding dimension for tokens
        
    Returns:
        Keras Model for tokenization
    """
    inputs = layers.Input(shape=input_shape)
    
    # Layer 1: Conv -> GELU -> BN
    # Stride 4 reduces sequence length by 4x
    x = layers.Conv1D(filters=embed_dim // 2, kernel_size=10, strides=4, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    
    # Layer 2: Conv -> GELU -> BN
    # Another Stride 4 reduction
    # Total reduction = 16x. (2048 -> 128 tokens)
    x = layers.Conv1D(filters=embed_dim, kernel_size=3, strides=4, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    
    return keras.Model(inputs, x, name="CNN_Tokenizer")


def build_faultformer(input_shape=(2048, 1), 
                      num_classes=10, 
                      embed_dim=64, 
                      num_heads=4, 
                      ff_dim=128, 
                      num_layers=4,
                      pretraining=False):
    """
    Build the FaultFormer model.
    
    Args:
        input_shape: Input signal shape (window_size, channels)
        num_classes: Number of fault classes
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward network dimension
        num_layers: Number of transformer layers
        pretraining: If True, use reconstruction head; else classification
        
    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=input_shape)
    
    # 1. Tokenization
    tokenizer = get_cnn_tokenizer(input_shape, embed_dim)
    tokens = tokenizer(inputs)  # Shape: (Batch, 128, 64)
    
    # 2. Add Absolute Position Embeddings
    seq_len = tokens.shape[1]
    pos_emb = layers.Embedding(input_dim=seq_len, output_dim=embed_dim)(tf.range(seq_len))
    x = tokens + pos_emb
    
    # 3. Transformer Encoder Stack
    for _ in range(num_layers):
        x = TransformerEncoderBlock(embed_dim, num_heads, ff_dim)(x)
        
    # 4. Output Head
    if pretraining:
        # --- Reconstruction Head (Masked Signal Modeling) ---
        # Upsample back to original length (2048)
        # (Batch, 128, 64) -> (Batch, 2048, 1)
        x = layers.Conv1DTranspose(filters=embed_dim // 2, kernel_size=3, strides=4, padding='same')(x)
        x = layers.Activation('gelu')(x)
        x = layers.Conv1DTranspose(filters=1, kernel_size=10, strides=4, padding='same')(x)
        outputs = x
        model_name = "FaultFormer_Pretrainer"
    else:
        # --- Classification Head (Diagnosis) ---
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model_name = "FaultFormer_Classifier"
        
    return keras.Model(inputs=inputs, outputs=outputs, name=model_name)


if __name__ == "__main__":
    # Test the build
    print("Building FaultFormer model...")
    model = build_faultformer(pretraining=False)
    model.summary()
