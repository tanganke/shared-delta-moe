"""transformers' Qwen3.5 MoE with DeltaMoE blocks. Importing it patches Qwen3_5MoeDecoderLayer (see modeling.py)."""

from .modeling import Qwen3_5BitDeltaMoeBlock, Qwen3_5DeltaMoeBlock, to_bitdelta

__all__ = ["Qwen3_5BitDeltaMoeBlock", "Qwen3_5DeltaMoeBlock", "to_bitdelta"]
