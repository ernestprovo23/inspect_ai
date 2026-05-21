"""Shared opt-in debug print for the checkpointing subsystem.

Off by default. Set ``INSPECT_CHECKPOINT_VALIDATE`` to any non-empty
value to enable verbose stdout output across hydrate, host egress,
fire, and resume — useful for diagnosing remote-destination behavior
without permanently muddying the normal log surface.
"""

from __future__ import annotations

import os
from typing import Any

_ENV_VAR = "INSPECT_CHECKPOINT_VALIDATE"


def debug_enabled() -> bool:
    return bool(os.environ.get(_ENV_VAR))


def debug(*args: Any, **kwargs: Any) -> None:
    if debug_enabled():
        print(*args, **kwargs)
