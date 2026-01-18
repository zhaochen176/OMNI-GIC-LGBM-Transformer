from typing import Dict, Callable

_REGISTRY: Dict[str, Callable] = {}

def register(name: str):
    def deco(fn: Callable):
        _REGISTRY[name] = fn
        return fn
    return deco

def get_baseline(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown baseline '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]

def available_baselines():
    return sorted(_REGISTRY.keys())
