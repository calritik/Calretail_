"""
Bounds how many capabilities keep their state in memory at once.

Each capability builds derived frames on first use and would otherwise hold
them for the life of the process. That is the right trade on a machine with
memory to spare and the wrong one on a 512 MiB host, where working through the
console grows the process past the cap and gets it OOM-killed.

Modules register here once they finish building. When more than
``CALRETAIL_WARM_CAPABILITIES`` are warm the least recently used one is reset:
its frames are dropped and it rebuilds on its next call. A rebuild is a few
seconds against an already-open SQLite database, which beats being killed
mid-request.

Raise the limit (or set it to 16) wherever memory is not the constraint.
"""
from __future__ import annotations

import gc
import os
import sys
from collections import OrderedDict

LIMIT = max(1, int(os.environ.get("CALRETAIL_WARM_CAPABILITIES", "3")))

_warm: "OrderedDict[str, None]" = OrderedDict()


def touch(module_name: str) -> None:
    """Mark a capability as just-used, evicting the coldest ones over the limit."""
    _warm[module_name] = None
    _warm.move_to_end(module_name)

    while len(_warm) > LIMIT:
        victim, _ = _warm.popitem(last=False)
        # Never evict the module that just registered — it is about to read the
        # state it built.
        if victim == module_name:
            _warm[victim] = None
            _warm.move_to_end(victim)
            break
        reset = getattr(sys.modules.get(victim), "reset", None)
        if callable(reset):
            reset()
        gc.collect()


def warm() -> list[str]:
    """Capabilities currently holding state, coldest first."""
    return [m.rsplit(".", 1)[-1] for m in _warm]


def reset_all() -> None:
    """Drop every capability's state. Used by tests and after a rebuild."""
    for name in list(_warm):
        reset = getattr(sys.modules.get(name), "reset", None)
        if callable(reset):
            reset()
    _warm.clear()
    gc.collect()
