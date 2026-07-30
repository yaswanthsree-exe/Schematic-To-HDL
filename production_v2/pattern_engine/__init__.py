"""Functional pattern recognition engine.

Recognizes circuit motifs (XOR, SR latch, ...) in the gate graph produced by
predict.py and compresses them into macro nodes before HDL generation.

    from pattern_engine import compress
    result = compress(circuit_result.graph)
    hdl = generate_all(result.graph, inputs, outputs, name)
"""
from .compressor import MAX_PASSES, CompressionResult, MatchRecord, compress

__all__ = ["compress", "CompressionResult", "MatchRecord", "MAX_PASSES"]
