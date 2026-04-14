from __future__ import annotations

import pandas as pd

from src.ui import dataframes


def test_with_optional_background_gradient_returns_plain_dataframe_without_matplotlib(monkeypatch) -> None:
    df = pd.DataFrame({"Wholesale MW": [1.0, 2.0], "Balancing MW": [0.5, 1.5]})
    monkeypatch.setattr(dataframes, "_matplotlib_available", lambda: False)

    result = dataframes.with_optional_background_gradient(
        df,
        subset=["Wholesale MW", "Balancing MW"],
        cmap="Blues",
    )

    assert result is df


def test_with_optional_background_gradient_returns_styler_when_matplotlib_is_available(monkeypatch) -> None:
    df = pd.DataFrame({"Gap": [1.0, -0.5]})
    monkeypatch.setattr(dataframes, "_matplotlib_available", lambda: True)

    result = dataframes.with_optional_background_gradient(
        df,
        subset=["Gap"],
        cmap="RdYlGn_r",
    )

    assert result.__class__.__name__ == "Styler"
    assert result.data is df
