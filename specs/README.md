# TLA+ specifications

Each spec is model-checked with TLC to no counterexample. The tests named below
check the same properties on the Python code.

| Spec | Models | Code | Tests |
|---|---|---|---|
| `RiskGovernor.tla` | daily loss cap, kill-switch latch, max positions / orders, equity never negative | `risk_governor.py` | `tests/test_risk_governor*.py`, `tests/test_phase1_fixes.py` |
| `OrderStateMachine.tla` | one order: signal → governor → fill → close | `bot.py`, `mt5_bridge.py` | `tests/test_critical_fixes.py` |
| `PositionLifecycle.tla` | stop loss never widens, break-even, partial closes | `position_manager.py` | `tests/test_sl_never_widens.py` |
| `BotControl.tla` | `bot.py` controlled from an MCP client: one trader per account, booked once, stop by instance, confirmed pause, limits only tighten, kill switch untouched; liveness of stop / booking / close requests | `bot_control.py`, `bot.py`, `mcp_server.py`, `trade_booking.py` | `tests/test_bot_control.py` |

## Run TLC

Download `tla2tools.jar` from https://github.com/tlaplus/tlaplus/releases and run
TLC in this folder. You need Java 11 or later.

```
java -cp tla2tools.jar tlc2.TLC -workers auto RiskGovernor.tla      -config RiskGovernor.cfg
java -cp tla2tools.jar tlc2.TLC -workers auto OrderStateMachine.tla -config OrderStateMachine.cfg
java -cp tla2tools.jar tlc2.TLC -workers auto PositionLifecycle.tla -config PositionLifecycle.cfg
java -cp tla2tools.jar tlc2.TLC -workers auto BotControl.tla        -config BotControl.cfg
java -cp tla2tools.jar tlc2.TLC -workers auto BotControl.tla        -config BotControlLive.cfg
```

`BotControl.cfg` checks every safety invariant and action property.
`BotControlLive.cfg` checks the liveness properties under fairness, on a smaller
model, because liveness checking costs more.

## Last results (TLC 2.19, 2 workers)

| Model | Distinct states | Result |
|---|---|---|
| RiskGovernor | 46,136 | no error |
| OrderStateMachine | see the PR / commit message | no error |
| PositionLifecycle | 5,550 | no error |
| BotControl (safety) | 10,029,140 | no error |
| BotControlLive (liveness) | 751,896 | no error |

## The spec catches these bugs (mutation check)

Each change below was made to a copy of `BotControl.tla`, and TLC reported a
counterexample. The same change to the code makes `tests/test_bot_control.py` fail.

| Mutation | Violated |
|---|---|
| M1: book a trade without holding the account lock | INV-BC-3 BookedOnce (two processes book the same close) |
| M2: a stop request is a plain on/off flag, not addressed to an instance | INV-BC-4 StopHitsTarget (a stale stop kills the next bot) |
| M3: use the AI's override directly instead of the tighter of it and config.py | INV-BC-6 LimitsOnlyTighten (after you lower config.py) |
| M4: the bot confirms a pause while an order is in flight | INV-BC-5 PauseConfirmed |

A fifth mutation, skipping the pause re-check right before an order, does **not**
violate safety. The guarantee comes from the bot confirming the pause; the
re-check only makes a pause take effect sooner. The code keeps it for that reason.

## PositionLifecycle: a bug TLC found in the code

`INV_PL1_SLNeverWidens` failed on this trace: the trailing stop moves the SL past
entry, then "move to break-even" pulls it back to entry. `position_manager.py` had
the same flaw. Tier 1 of the TP ladder moved the SL to break-even unconditionally,
which widened a trailed SL (reproduced: BUY SL 1.10250 → 1.10010). Both the spec
and the code now move it only if that tightens the stop.
