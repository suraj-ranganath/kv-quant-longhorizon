from __future__ import annotations


def create_quantizer(method: str, bits: int | None, block_size: int, **kwargs):
    method = method.upper()
    if method == "BF16":
        return None
    if method == "SPATIAL_MIXED":
        from .spatial_mixed import SpatialMixedQuantizer

        return SpatialMixedQuantizer(block_size=block_size, **kwargs)
    if method == "FLOWCACHE_PROFILE":
        from .flowcache_profile import FlowCacheProfileQuantizer

        return FlowCacheProfileQuantizer(block_size=block_size, **kwargs)
    if bits is None:
        raise ValueError(f"bits must be provided for method={method}")
    if method == "RTN":
        from .rtn import RTNQuantizer
        return RTNQuantizer(bits=bits, block_size=block_size, key_bits=key_bits, value_bits=value_bits, name=name)
    if method == "KIVI":
        from .kivi import KIVIQuantizer
        return KIVIQuantizer(bits=bits, block_size=block_size, key_bits=key_bits, value_bits=value_bits, name=name)
    if method == "QUAROT_KV":
        from .quarot_kv import QuaRotKVQuantizer
        return QuaRotKVQuantizer(bits=bits, block_size=block_size)
    if method == "PRQ":
        from .prq import PRQQuantizer
        return PRQQuantizer(bits=bits, block_size=block_size, **kwargs)
    if method == "QAQ":
        from .qaq import QAQQuantizer
        return QAQQuantizer(bits=bits, block_size=block_size, **kwargs)
    if method == "AGE_TIER":
        from .age_tier import AgeTierQuantizer
        return AgeTierQuantizer(bits=bits, block_size=block_size, **kwargs)
    if method == "TPTQ":
        from .tptq import TPTQQuantizer
        return TPTQQuantizer(bits=bits, block_size=block_size, **kwargs)
    if method == "FLOWCACHE_HYBRID":
        from .flowcache_hybrid import FlowCacheHybridQuantizer
        return FlowCacheHybridQuantizer(bits=bits, block_size=block_size, **kwargs)
    if method == "FLOWCACHE_ADAPTIVE":
        from .flowcache_adaptive import FlowCacheAdaptiveQuantizer

        return FlowCacheAdaptiveQuantizer(bits=bits, block_size=block_size, **kwargs)
    if method == "FLOWCACHE_PRUNE":
        from .flowcache_prune import FlowCachePruneQuantizer

        return FlowCachePruneQuantizer(bits=bits, block_size=block_size, **kwargs)
    if method == "FLOWCACHE_SOFT_PRUNE":
        from .flowcache_soft_prune import FlowCacheSoftPruneQuantizer

        return FlowCacheSoftPruneQuantizer(bits=bits, block_size=block_size, **kwargs)
    raise ValueError(f"Unsupported method={method}")
