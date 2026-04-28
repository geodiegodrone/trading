"""Binance dashboard entrypoint.

This keeps the historical `dashboard.py` name working, but the actual app is
the Binance BTC-only dashboard.
"""

from dashboard_multi import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("dashboard_multi:app", host="0.0.0.0", port=8000, reload=False)
