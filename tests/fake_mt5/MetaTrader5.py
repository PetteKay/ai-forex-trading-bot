"""
Fake MetaTrader5 package for offline tests (the real one is Windows-only).
Stateful: an account, a few symbols, synthetic H1/H4 candles, positions, and
order_send that fills market orders. Put this folder first on PYTHONPATH.
"""
import time as _time
from types import SimpleNamespace as _NS

import numpy as _np

TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_M30 = 1, 5, 15, 30
TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1, TIMEFRAME_W1 = 16385, 16388, 16408, 32769
ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
ORDER_TIME_GTC = 0
TRADE_ACTION_DEAL, TRADE_ACTION_SLTP = 1, 6
TRADE_RETCODE_DONE = 10009

_SYMBOLS = {"EURUSD": 1.0850, "GBPUSD": 1.2700, "USDJPY": 150.20, "AUDUSD": 0.6600}
_positions: list = []
_next_ticket = [5000]
orders_sent: list = []           # tests can inspect this


def __getattr__(name):           # any other constant
    return 0


def initialize(*a, **k): return True
def shutdown(): return None
def last_error(): return (0, "ok")
def terminal_info(): return _NS(name="FakeTerminal", connected=True)
def account_info():
    return _NS(login=1, company="FakeBroker", balance=10_000.0, equity=10_000.0, currency="USD")
def symbols_get(*a, **k): return [_NS(name=s) for s in _SYMBOLS]
def symbol_select(*a, **k): return True


def _point(s): return 0.001 if "JPY" in s else 0.00001


def symbol_info(symbol):
    if symbol not in _SYMBOLS:
        return None
    pt = _point(symbol)
    return _NS(name=symbol, point=pt, digits=3 if pt == 0.001 else 5,
               trade_tick_value=1.0 if pt == 0.00001 else 0.66, trade_tick_size=pt,
               volume_min=0.01, volume_step=0.01, volume_max=100.0, filling_mode=2,
               visible=True, trade_mode=4)


def symbol_info_tick(symbol):
    p = _SYMBOLS.get(symbol)
    if p is None:
        return None
    return _NS(bid=p, ask=p + 10 * _point(symbol), time=int(_time.time()))


def copy_rates_from_pos(symbol, tf, start, count):
    base = _SYMBOLS.get(symbol, 1.0)
    rng = _np.random.default_rng(abs(hash((symbol, tf))) % 2**32)
    c = base * _np.exp(_np.cumsum(rng.normal(0.0002, 0.0015, count)))
    c *= base / c[-1]
    o = _np.r_[c[0], c[:-1]]
    h = _np.maximum(o, c) * (1 + abs(rng.normal(0, .0008, count)))
    lo = _np.minimum(o, c) * (1 - abs(rng.normal(0, .0008, count)))
    step = 14400 if tf == TIMEFRAME_H4 else 3600
    now = (int(_time.time()) // step) * step - start * step
    t = _np.arange(now - (count - 1) * step, now + 1, step)
    dt = _np.dtype([("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
                    ("close", "f8"), ("tick_volume", "i8")])
    return _np.array(list(zip(t, o, h, lo, c, rng.integers(100, 1000, count))), dtype=dt)


def positions_get(symbol=None, ticket=None, group=None):
    out = [p for p in _positions
           if (symbol is None or p.symbol == symbol) and (ticket is None or p.ticket == ticket)]
    return tuple(out)


def order_send(req):
    orders_sent.append(dict(req))
    if req.get("action") != TRADE_ACTION_DEAL:
        return _NS(retcode=TRADE_RETCODE_DONE, order=0)
    if "position" in req:                     # close
        _positions[:] = [p for p in _positions if p.ticket != req["position"]]
        return _NS(retcode=TRADE_RETCODE_DONE, order=req["position"])
    _next_ticket[0] += 1
    t = _next_ticket[0]
    _positions.append(_NS(ticket=t, symbol=req["symbol"], type=req["type"], volume=req["volume"],
                          price_open=req["price"], sl=req["sl"], tp=req["tp"], profit=0.0,
                          magic=req.get("magic", 0), time=int(_time.time())))
    return _NS(retcode=TRADE_RETCODE_DONE, order=t)
