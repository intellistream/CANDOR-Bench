"""Pluggable buffer implementations for the modular router.

Two implementations:
  FlatBuffer    — single pre-alloc array. What router.py uses inline.
  ClusterBuffer — IVF-style K-bucket. Each bucket is itself a FlatBuffer.

The router (router_modular.py) talks only to the Buffer interface
(see base.py). This lets one router cover both "flat" and "structured"
buffer designs and plug into different maintenance strategies.
"""
from .base import Buffer
from .flat import FlatBuffer
from .cluster import ClusterBuffer

__all__ = ["Buffer", "FlatBuffer", "ClusterBuffer"]
