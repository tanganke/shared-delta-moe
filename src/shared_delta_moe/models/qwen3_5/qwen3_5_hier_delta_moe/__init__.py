"""transformers' dense Qwen3.5 (Qwen3.8) with every MLP a HierDeltaMoE. Importing it patches Qwen3_5DecoderLayer
(see modeling.py); convert.py upscales a base model and its domain fine-tunes into that layout."""

from .modeling import Qwen3_5HierDeltaMoeBlock

__all__ = ["Qwen3_5HierDeltaMoeBlock"]
