from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
from functools import lru_cache
import json
import os
import threading
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pathlib import Path
import time

try:
    from binance.client import Client
except Exception:  # pragma: no cover
    Client = None


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

_local = threading.local()
_balance_lock = threading.Lock()
_balance_cache: dict[str, float | None | int] = {"value": None, "ts": 0}
_RETRY_COUNT = max(1, int(os.getenv("BINANCE_RETRIES", "3")))
_RETRY_DELAY = float(os.getenv("BINANCE_RETRY_DELAY", "0.5"))
_BALANCE_CACHE_TTL = max(5, int(os.getenv("BINANCE_BALANCE_CACHE_TTL", "30")))
_BALANCE_CACHE_FILE = BASE_DIR / "balance_cache.json"
_DAILY_BALANCE_FILE = BASE_DIR / "daily_balance.json"


def _get_client():
    if not hasattr(_local, "client"):
        if Client is None:
            raise RuntimeError("python-binance is required")
        _local.client = Client(API_KEY, API_SECRET, testnet=False, requests_params={"timeout": 30})
        _local.client.FUTURES_URL = "https://demo-fapi.binance.com/fapi"
    return _local.client


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_balance_cache() -> Optional[float]:
    try:
        if _BALANCE_CACHE_FILE.exists():
            payload = json.loads(_BALANCE_CACHE_FILE.read_text(encoding="utf-8"))
            value = _safe_float(payload.get("balance"))
            if value is not None:
                return value
    except Exception:
        return None
    return None


def _write_balance_cache(balance: float) -> None:
    try:
        _BALANCE_CACHE_FILE.write_text(
            json.dumps(
                {
                    "balance": float(balance),
                    "ts": int(time.time()),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def _read_daily_balance_fallback() -> Optional[float]:
    try:
        if _DAILY_BALANCE_FILE.exists():
            payload = json.loads(_DAILY_BALANCE_FILE.read_text(encoding="utf-8"))
            value = _safe_float(payload.get("daily_start_balance"))
            if value is not None:
                return value
    except Exception:
        return None
    return None


def _with_retry(fn, *args, **kwargs):
    last_exc = None
    for attempt in range(_RETRY_COUNT):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            if attempt + 1 < _RETRY_COUNT:
                time.sleep(_RETRY_DELAY * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Binance call failed without exception")


@lru_cache(maxsize=1)
def _exchange_info() -> Dict[str, Any]:
    return _with_retry(_get_client().futures_exchange_info)


def get_symbol_info(symbol: str) -> Dict[str, Any]:
    target = str(symbol or "").upper().strip()
    if not target:
        return {}
    try:
        for row in _exchange_info().get("symbols", []):
            if str(row.get("symbol") or "").upper() == target:
                return row
    except Exception:
        return {}
    return {}


def is_symbol_trading(symbol: str) -> bool:
    info = get_symbol_info(symbol)
    return str(info.get("status") or "").upper() == "TRADING"


def get_quantity_step(symbol: str, order_type: str = "MARKET") -> Optional[float]:
    info = get_symbol_info(symbol)
    if not info:
        return None
    wanted = "MARKET_LOT_SIZE" if str(order_type or "").upper() == "MARKET" else "LOT_SIZE"
    for row in info.get("filters", []):
        if str(row.get("filterType") or "").upper() != wanted:
            continue
        step = _safe_float(row.get("stepSize"))
        if step is not None and step > 0:
            return step
    for row in info.get("filters", []):
        if str(row.get("filterType") or "").upper() != "LOT_SIZE":
            continue
        step = _safe_float(row.get("stepSize"))
        if step is not None and step > 0:
            return step
    quantity_precision = info.get("quantityPrecision")
    if quantity_precision is not None:
        try:
            return 10 ** (-int(quantity_precision))
        except Exception:
            pass
    return None


def get_min_quantity(symbol: str, order_type: str = "MARKET") -> float:
    info = get_symbol_info(symbol)
    wanted = "MARKET_LOT_SIZE" if str(order_type or "").upper() == "MARKET" else "LOT_SIZE"
    for row in info.get("filters", []):
        if str(row.get("filterType") or "").upper() != wanted:
            continue
        value = _safe_float(row.get("minQty"))
        if value is not None:
            return value
    for row in info.get("filters", []):
        if str(row.get("filterType") or "").upper() != "LOT_SIZE":
            continue
        value = _safe_float(row.get("minQty"))
        if value is not None:
            return value
    return 0.0


def normalize_quantity(symbol: str, qty: float, order_type: str = "MARKET") -> float:
    step = get_quantity_step(symbol, order_type)
    if not step or step <= 0:
        return max(float(qty), 0.0)
    qty_dec = Decimal(str(max(float(qty), 0.0)))
    step_dec = Decimal(str(step))
    normalized = (qty_dec / step_dec).to_integral_value(rounding=ROUND_DOWN) * step_dec
    min_qty = Decimal(str(get_min_quantity(symbol, order_type)))
    if min_qty > 0 and normalized < min_qty:
        return 0.0
    return float(normalized)


def get_price(symbol: str) -> float:
    client = _get_client()
    ticker = _with_retry(client.futures_symbol_ticker, symbol=symbol)
    price = _safe_float(ticker.get("price"))
    if price is None:
        ticker = _with_retry(client.futures_mark_price, symbol=symbol)
        price = _safe_float(ticker.get("markPrice"))
    if price is None:
        raise RuntimeError(f"Unable to fetch Binance price for {symbol}")
    return price


def get_position(symbol: str) -> Dict[str, Any] | None:
    client = _get_client()
    positions = _with_retry(client.futures_position_information, symbol=symbol)
    for row in positions:
        position_amt = _safe_float(row.get("positionAmt")) or 0.0
        if abs(position_amt) > 0:
            return {
                "side": "Buy" if position_amt > 0 else "Sell",
                "size": abs(position_amt),
                "entry_price": _safe_float(row.get("entryPrice")) or 0.0,
                "unrealised_pnl": _safe_float(row.get("unRealizedProfit")) or 0.0,
            }
    return None


def open_long(symbol: str, qty: float) -> Dict[str, Any]:
    qty = normalize_quantity(symbol, qty)
    if qty <= 0:
        raise RuntimeError(f"Quantity too small for {symbol}")
    return _with_retry(
        _get_client().futures_create_order,
        symbol=symbol,
        side="BUY",
        type="MARKET",
        quantity=qty,
    )


def open_short(symbol: str, qty: float) -> Dict[str, Any]:
    qty = normalize_quantity(symbol, qty)
    if qty <= 0:
        raise RuntimeError(f"Quantity too small for {symbol}")
    return _with_retry(
        _get_client().futures_create_order,
        symbol=symbol,
        side="SELL",
        type="MARKET",
        quantity=qty,
    )


def close_position(symbol: str, position: Dict[str, Any]) -> Dict[str, Any]:
    side = str(position.get("side") or "").lower()
    qty = normalize_quantity(symbol, float(position.get("size") or 0.0))
    if qty <= 0:
        return {}
    order_side = "SELL" if side == "buy" else "BUY"
    return _with_retry(
        _get_client().futures_create_order,
        symbol=symbol,
        side=order_side,
        type="MARKET",
        quantity=qty,
        reduceOnly=True,
    )


def get_balance(force_refresh: bool = False) -> float:
    now = time.time()
    with _balance_lock:
        cached_value = _balance_cache.get("value")
        cached_ts = float(_balance_cache.get("ts") or 0)
        if cached_value is not None and not force_refresh and now - cached_ts < _BALANCE_CACHE_TTL:
            return float(cached_value)

    try:
        balances = _with_retry(_get_client().futures_account_balance)
        for row in balances:
            if row.get("asset") == "USDT":
                for key in ("availableBalance", "balance", "crossWalletBalance"):
                    value = _safe_float(row.get(key))
                    if value is not None:
                        with _balance_lock:
                            _balance_cache["value"] = value
                            _balance_cache["ts"] = now
                        _write_balance_cache(value)
                        return value
    except Exception:
        pass

    cached = _read_balance_cache()
    if cached is not None:
        with _balance_lock:
            _balance_cache["value"] = cached
            _balance_cache["ts"] = now
        return cached

    daily_cached = _read_daily_balance_fallback()
    if daily_cached is not None:
        with _balance_lock:
            _balance_cache["value"] = daily_cached
            _balance_cache["ts"] = now
        return daily_cached

    with _balance_lock:
        cached_value = _balance_cache.get("value")
        if cached_value is not None:
            return float(cached_value)
    return 0.0


def set_leverage(symbol: str, leverage: int) -> None:
    _with_retry(_get_client().futures_change_leverage, symbol=symbol, leverage=int(leverage))


def get_kline(symbol: str, timeframe: int, limit: int = 250):
    interval = f"{timeframe}m"
    return _with_retry(_get_client().futures_klines, symbol=symbol, interval=interval, limit=limit)
