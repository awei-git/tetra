"""yfinance helpers for market data."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional dependency
    yf = None

UTC = timezone.utc


def yfinance_available() -> bool:
    return yf is not None


def _require_yfinance() -> None:
    if yf is None:
        raise RuntimeError("yfinance is not installed")


def _to_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _clean_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "iloc") and not isinstance(value, (str, bytes)):
        try:
            if getattr(value, "size", 0) == 0:
                return None
            value = value.iloc[0]
        except Exception:
            return None
    try:
        if value != value:
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def fetch_yfinance_asset_info(symbol: str) -> Optional[Dict[str, Any]]:
    _require_yfinance()
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    return info if info else None


def fetch_yfinance_ohlcv(symbol: str, start: date, end: date) -> List[Dict[str, Any]]:
    _require_yfinance()
    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
    end_dt = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    data = yf.download(
        symbol,
        start=start_dt,
        end=end_dt,
        interval="1d",
        progress=False,
        auto_adjust=False,
        actions=False,
    )
    if data is None or data.empty:
        return []

    if "Open" not in data.columns and getattr(data.columns, "nlevels", 1) > 1:
        try:
            data.columns = data.columns.droplevel(-1)
        except Exception:
            pass

    data = data.reset_index()
    rows: List[Dict[str, Any]] = []
    for _, row in data.iterrows():
        ts = row.get("Date")
        if ts is None:
            ts = row.get("Datetime")
        if ts is None:
            continue
        if hasattr(ts, "iloc") and not isinstance(ts, (str, bytes)):
            try:
                if getattr(ts, "size", 0) == 0:
                    continue
                ts = ts.iloc[0]
            except Exception:
                continue
        timestamp = _to_utc(ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts)
        rows.append(
            {
                "symbol": symbol.upper(),
                "timestamp": timestamp,
                "open": _clean_value(row.get("Open")),
                "high": _clean_value(row.get("High")),
                "low": _clean_value(row.get("Low")),
                "close": _clean_value(row.get("Close")),
                "volume": _clean_value(row.get("Volume")),
                "vwap": None,
                "turnover": None,
                "source": "yfinance",
                "ingested_at": datetime.now(tz=UTC),
            }
        )
    return rows
