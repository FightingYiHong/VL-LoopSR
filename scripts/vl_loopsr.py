#!/usr/bin/env python3
"""Compatibility entry point for archived VL-LoopSR/V11 manifests.

New experiments should load :mod:`vega_sr`. Executing the canonical source in
this module namespace preserves the runtime-global behavior expected by older
dynamic benchmark loaders.
"""

from pathlib import Path


_CANONICAL_SOURCE = Path(__file__).with_name("vega_sr.py")
exec(
    compile(_CANONICAL_SOURCE.read_bytes(), str(_CANONICAL_SOURCE), "exec"),
    globals(),
    globals(),
)
