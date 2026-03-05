from __future__ import annotations


def packed_bytes(num_values: int, bits: int) -> int:
    if bits <= 0:
        raise ValueError("bits must be positive")
    total_bits = num_values * bits
    return (total_bits + 7) // 8
