from __future__ import annotations

def create_quantizer(method: str, bits: int, block_size: int):
    method = method.upper()
    if method == "BF16":
        return None
    if method == "RTN":
        from .rtn import RTNQuantizer
        return RTNQuantizer(bits=bits, block_size=block_size)
    if method == "KIVI":
        from .kivi import KIVIQuantizer
        return KIVIQuantizer(bits=bits, block_size=block_size)
    if method == "QUAROT_KV":
        from .quarot_kv import QuaRotKVQuantizer
        return QuaRotKVQuantizer(bits=bits, block_size=block_size)
    raise ValueError(f"Unsupported method={method}")
