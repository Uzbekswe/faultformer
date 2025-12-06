"""
FaultFormer model architecture and custom layers.

Classes:
    - TransformerEncoderBlock: Transformer encoder with RoPE support
    - RotaryEmbedding: Rotary Positional Embedding layer

Functions:
    - build_faultformer: Build the complete FaultFormer model
    - get_cnn_tokenizer: Build the hierarchical CNN tokenizer
"""

from src.models.faultformer import build_faultformer, get_cnn_tokenizer
from src.models.layers import TransformerEncoderBlock, RotaryEmbedding
from src.models.pretrain_utils import get_pretraining_dataset, MaskedSignalModeling

__all__ = [
    "build_faultformer",
    "get_cnn_tokenizer", 
    "TransformerEncoderBlock",
    "RotaryEmbedding",
    "get_pretraining_dataset",
    "MaskedSignalModeling"
]
