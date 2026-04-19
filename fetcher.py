"""
Fetcher for SET100 Index daily prices.

Data source: Yahoo Finance quote snapshot for SET100.BK.
Yahoo Finance does not provide full historical OHLCV for SET100;
it only returns the current-day snapshot.  Run this daily (e.g. via cron)
to accumulate history in the local database.
"""

from typing import Optional
import yfinance as yf
from datetime import date

TICKER = "SET100.BK"


def fetch_today() -> Optional[dict]:
    """
    Return today's SET100 OHLCV snapshot, or None if unavailable.
    Keys: date, open, high, low, close, volume, prev_close
    """
    tk = yf.Ticker(TICKER)
    info = tk.info
    if not info:
        return None

    open_p  = info.get("regularMarketOpen")         or info.get("open")
    high    = info.get("regularMarketDayHigh")       or info.get("dayHigh")
    low     = info.get("regularMarketDayLow")        or info.get("dayLow")
    close   = info.get("regularMarketPrice")         or info.get("currentPrice")
    volume  = info.get("regularMarketVolume")        or info.get("volume") or 0
    prev_cl = info.get("regularMarketPreviousClose") or info.get("previousClose")

    if open_p is None or close is None:
        return None

    return {
        "date":       str(date.today()),
        "open":       round(float(open_p), 2),
        "high":       round(float(high),   2) if high   else round(float(close), 2),
        "low":        round(float(low),    2) if low    else round(float(close), 2),
        "close":      round(float(close),  2),
        "volume":     int(volume),
        "prev_close": round(float(prev_cl), 2) if prev_cl else None,
    }
