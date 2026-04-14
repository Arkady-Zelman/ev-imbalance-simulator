from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from typing import Any, Sequence

import pandas as pd


@lru_cache(maxsize=1)
def _matplotlib_available() -> bool:
    return find_spec("matplotlib") is not None


def with_optional_background_gradient(
    df: pd.DataFrame,
    *,
    subset: Sequence[str],
    cmap: str,
) -> Any:
    if not _matplotlib_available():
        return df
    return df.style.background_gradient(subset=list(subset), cmap=cmap)
