from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "trades.db"
_LAST_CANDLE_BODY_PCT: float = 0.0


def set_trade_context(candle_body_pct: float | None = None) -> None:
    global _LAST_CANDLE_BODY_PCT
    try:
        _LAST_CANDLE_BODY_PCT = float(candle_body_pct or 0.0)
    except (TypeError, ValueError):
        _LAST_CANDLE_BODY_PCT = 0.0


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _columns(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("PRAGMA table_info(trades)").fetchall()
    return {row[1] for row in rows}


def _ensure_column(conn: sqlite3.Connection, column: str, ddl: str) -> None:
    try:
        if column not in _columns(conn):
            conn.execute(ddl)
    except sqlite3.OperationalError:
        pass


def _init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                side TEXT NOT NULL,
                symbol TEXT NOT NULL DEFAULT 'BTCUSDT',
                entry_price REAL NOT NULL,
                exit_price REAL,
                pnl REAL,
                risk_usdt REAL,
                r_multiple REAL,
                exit_reason TEXT,
                ema9 REAL,
                ema21 REAL,
                rsi REAL,
                ema200 REAL,
                candle_body_pct REAL,
                adx REAL,
                supertrend_dir INTEGER,
                volume_ratio REAL,
                atr_pct REAL,
                return_3_pct REAL,
                return_5_pct REAL,
                range_5_pct REAL,
                body_avg_3_pct REAL,
                volume_trend_3_ratio REAL,
                close_pos_5 REAL,
                ema9_slope_3_pct REAL,
                qty REAL,
                notional_usdt REAL,
                result TEXT
            )
            """
        )
        _ensure_column(conn, "symbol", "ALTER TABLE trades ADD COLUMN symbol TEXT DEFAULT 'BTCUSDT'")
        _ensure_column(conn, "candle_body_pct", "ALTER TABLE trades ADD COLUMN candle_body_pct REAL")
        _ensure_column(conn, "adx", "ALTER TABLE trades ADD COLUMN adx REAL")
        _ensure_column(conn, "supertrend_dir", "ALTER TABLE trades ADD COLUMN supertrend_dir INTEGER")
        _ensure_column(conn, "volume_ratio", "ALTER TABLE trades ADD COLUMN volume_ratio REAL")
        _ensure_column(conn, "atr_pct", "ALTER TABLE trades ADD COLUMN atr_pct REAL")
        _ensure_column(conn, "return_3_pct", "ALTER TABLE trades ADD COLUMN return_3_pct REAL")
        _ensure_column(conn, "return_5_pct", "ALTER TABLE trades ADD COLUMN return_5_pct REAL")
        _ensure_column(conn, "range_5_pct", "ALTER TABLE trades ADD COLUMN range_5_pct REAL")
        _ensure_column(conn, "body_avg_3_pct", "ALTER TABLE trades ADD COLUMN body_avg_3_pct REAL")
        _ensure_column(conn, "volume_trend_3_ratio", "ALTER TABLE trades ADD COLUMN volume_trend_3_ratio REAL")
        _ensure_column(conn, "close_pos_5", "ALTER TABLE trades ADD COLUMN close_pos_5 REAL")
        _ensure_column(conn, "ema9_slope_3_pct", "ALTER TABLE trades ADD COLUMN ema9_slope_3_pct REAL")
        _ensure_column(conn, "qty", "ALTER TABLE trades ADD COLUMN qty REAL")
        _ensure_column(conn, "notional_usdt", "ALTER TABLE trades ADD COLUMN notional_usdt REAL")
        _ensure_column(conn, "risk_usdt", "ALTER TABLE trades ADD COLUMN risk_usdt REAL")
        _ensure_column(conn, "r_multiple", "ALTER TABLE trades ADD COLUMN r_multiple REAL")
        _ensure_column(conn, "exit_reason", "ALTER TABLE trades ADD COLUMN exit_reason TEXT")
        conn.execute("UPDATE trades SET symbol='BTCUSDT' WHERE symbol IS NULL OR symbol=''")
        conn.commit()


_init_db()


def _is_reconciled_trade(trade: Dict[str, Any]) -> bool:
    return str(trade.get("result") or "").upper() == "RECONCILED"


def _is_real_closed_trade(trade: Dict[str, Any]) -> bool:
    return trade.get("pnl") is not None and not _is_reconciled_trade(trade)


def log_trade(
    side: str,
    entry_price: float,
    ema9: float,
    ema21: float,
    rsi: float,
    ema200: float,
    adx: float = 0.0,
    supertrend_dir: int = 0,
    volume_ratio: float = 0.0,
    atr_pct: float = 0.0,
    return_3_pct: float = 0.0,
    return_5_pct: float = 0.0,
    range_5_pct: float = 0.0,
    body_avg_3_pct: float = 0.0,
    volume_trend_3_ratio: float = 0.0,
    close_pos_5: float = 0.0,
    ema9_slope_3_pct: float = 0.0,
    qty: float = 0.0,
    notional_usdt: float = 0.0,
    risk_usdt: float = 0.0,
    symbol: str = "BTCUSDT",
) -> int:
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO trades (
                ts, side, symbol, entry_price, exit_price, pnl, ema9, ema21, rsi,
                ema200, candle_body_pct, adx, supertrend_dir, volume_ratio, atr_pct,
                return_3_pct, return_5_pct, range_5_pct, body_avg_3_pct,
                volume_trend_3_ratio, close_pos_5, ema9_slope_3_pct,
                qty, notional_usdt, risk_usdt, r_multiple, exit_reason, result
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                side,
                symbol,
                float(entry_price),
                None,
                None,
                float(ema9),
                float(ema21),
                float(rsi),
                float(ema200),
                float(_LAST_CANDLE_BODY_PCT or 0.0),
                float(adx or 0.0),
                int(supertrend_dir or 0),
                float(volume_ratio or 0.0),
                float(atr_pct or 0.0),
                float(return_3_pct or 0.0),
                float(return_5_pct or 0.0),
                float(range_5_pct or 0.0),
                float(body_avg_3_pct or 0.0),
                float(volume_trend_3_ratio or 0.0),
                float(close_pos_5 or 0.0),
                float(ema9_slope_3_pct or 0.0),
                float(qty or 0.0),
                float(notional_usdt or 0.0),
                float(risk_usdt or 0.0),
                None,
                None,
                "OPEN",
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def close_trade(trade_id: int, exit_price: float, pnl: float, risk_usdt: float = 0.0, exit_reason: str = "") -> None:
    result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"
    r_multiple = (float(pnl) / float(risk_usdt)) if risk_usdt else None
    with _connect() as conn:
        conn.execute(
            """
            UPDATE trades
            SET exit_price = ?, pnl = ?, risk_usdt = COALESCE(risk_usdt, ?), r_multiple = ?, exit_reason = ?, result = ?
            WHERE id = ?
            """,
            (
                float(exit_price),
                float(pnl),
                float(risk_usdt or 0.0),
                r_multiple,
                exit_reason,
                result,
                int(trade_id),
            ),
        )
        conn.commit()


def reconcile_trade(trade_id: int, exit_price: float, exit_reason: str = "RECONCILED_ORPHAN") -> None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT entry_price, risk_usdt FROM trades WHERE id = ?",
            (int(trade_id),),
        ).fetchone()
        if row is None:
            return
        entry_price = float(row["entry_price"] or 0.0)
        risk_usdt = float(row["risk_usdt"] or 0.0)
        pnl = float(exit_price) - entry_price
        result = "RECONCILED"
        if risk_usdt > 0:
            r_multiple = pnl / risk_usdt
        else:
            r_multiple = None
        conn.execute(
            """
            UPDATE trades
            SET exit_price = ?, pnl = ?, risk_usdt = COALESCE(risk_usdt, ?), r_multiple = ?, exit_reason = ?, result = ?
            WHERE id = ?
            """,
            (
                float(exit_price),
                float(pnl),
                float(risk_usdt or 0.0),
                r_multiple,
                exit_reason,
                result,
                int(trade_id),
            ),
        )
        conn.commit()


def get_all_trades(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    query = """
        SELECT id, ts, side, symbol, entry_price, exit_price, pnl, ema9, ema21, rsi,
               ema200, candle_body_pct, adx, supertrend_dir, volume_ratio, atr_pct,
               return_3_pct, return_5_pct, range_5_pct, body_avg_3_pct,
               volume_trend_3_ratio, close_pos_5, ema9_slope_3_pct,
               qty, notional_usdt, risk_usdt, r_multiple, exit_reason, result
        FROM trades
    """
    params: List[Any] = []
    if symbol is not None:
        query += " WHERE symbol = ?"
        params.append(symbol)
    query += " ORDER BY id ASC"
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_stats(symbol: Optional[str] = None) -> Dict[str, Any]:
    trades = get_all_trades(symbol=symbol)
    closed = [trade for trade in trades if _is_real_closed_trade(trade)]
    reconciled = [trade for trade in trades if trade.get("pnl") is not None and _is_reconciled_trade(trade)]
    total = len(closed)
    wins = sum(1 for trade in closed if float(trade.get("pnl") or 0.0) > 0)
    losses = sum(1 for trade in closed if float(trade.get("pnl") or 0.0) < 0)
    pnls = [float(trade.get("pnl") or 0.0) for trade in closed]
    r_values = [float(trade.get("r_multiple") or 0.0) for trade in closed if trade.get("r_multiple") is not None]
    positive = [p for p in pnls if p > 0]
    negative = [p for p in pnls if p < 0]
    avg_pnl = sum(pnls) / total if total else 0.0
    avg_win = sum(positive) / len(positive) if positive else 0.0
    avg_loss = abs(sum(negative) / len(negative)) if negative else 0.0
    win_rate = (wins / total * 100.0) if total else 0.0
    best = max(pnls) if pnls else 0.0
    worst = min(pnls) if pnls else 0.0
    avg_r = sum(r_values) / len(r_values) if r_values else 0.0
    best_r = max(r_values) if r_values else 0.0
    worst_r = min(r_values) if r_values else 0.0

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best": best,
        "worst": worst,
        "avg_r": avg_r,
        "best_r": best_r,
        "worst_r": worst_r,
        "reconciled": len(reconciled),
        "all_closed": len(closed) + len(reconciled),
    }
