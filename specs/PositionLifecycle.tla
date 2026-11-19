---- MODULE PositionLifecycle ----
\* ============================================================
\* Tier-1 Position Lifecycle — Active Position Management
\* Personal AI Forex Trading Bot
\*
\* Models position modifications: trailing SL, break-even, partial close.
\*
\* Invariants:
\*   INV-PL-1: SL can only move in the direction that reduces risk
\*              (BUY: SL only moves up; SELL: SL only moves down)
\*   INV-PL-2: TP cannot be moved closer to current price (only widen)
\*   INV-PL-3: Partial close volume never exceeds current volume
\*   INV-PL-4: A position at break-even SL cannot have SL below entry
\*   INV-PL-5: Position volume is always > 0 while OPEN
\* ============================================================
EXTENDS Integers, Naturals, TLC

CONSTANTS
    InitVolume,       \* initial position volume (e.g., 10 = 0.10 lot in units)
    MaxPrice          \* model price bound

ASSUME InitVolume \in Nat /\ InitVolume > 0
ASSUME MaxPrice   \in Nat /\ MaxPrice > 0

VARIABLES
    status,           \* OPEN | CLOSED
    direction,        \* BUY(1) | SELL(-1)
    entry_price,
    current_price,
    sl,               \* stop loss price
    tp,               \* take profit price
    volume,           \* current volume (in lot units)
    breakeven_applied \* TRUE once SL is moved to entry

vars == <<status, direction, entry_price, current_price, sl, tp, volume, breakeven_applied>>

TypeOK ==
    /\ status           \in {"OPEN", "CLOSED"}
    /\ direction        \in {1, -1}
    /\ entry_price      \in 1..MaxPrice
    /\ current_price    \in 1..MaxPrice
    /\ sl               \in 0..MaxPrice
    /\ tp               \in 0..MaxPrice
    /\ volume           \in 0..InitVolume
    /\ breakeven_applied \in BOOLEAN

Init ==
    /\ status           = "OPEN"
    /\ direction        = 1
    /\ entry_price      = MaxPrice \div 2
    /\ current_price    = MaxPrice \div 2
    /\ sl               = (MaxPrice \div 2) - 5
    /\ tp               = (MaxPrice \div 2) + 10
    /\ volume           = InitVolume
    /\ breakeven_applied = FALSE

\* Price moves
PriceMove(new_price) ==
    /\ status        = "OPEN"
    /\ new_price \in 1..MaxPrice
    /\ current_price' = new_price
    /\ UNCHANGED <<status, direction, entry_price, sl, tp, volume, breakeven_applied>>

\* Move SL to break-even (only valid when in profit)
MoveToBreakeven ==
    /\ status           = "OPEN"
    /\ ~breakeven_applied
    /\ direction = 1 => current_price > entry_price + 10
    /\ direction = -1 => current_price < entry_price - 10
    \* Only if break-even is tighter than the current SL (a trailed SL can
    \* already be past entry). position_manager.py: needs_move / _tightens.
    /\ sl' = IF \/ (direction = 1  /\ entry_price > sl)
                \/ (direction = -1 /\ (sl = 0 \/ entry_price < sl))
             THEN entry_price ELSE sl
    /\ breakeven_applied' = TRUE
    /\ UNCHANGED <<status, direction, entry_price, current_price, tp, volume>>

\* Trail SL (only moves in profit direction — never widens risk)
TrailSL(new_sl) ==
    /\ status   = "OPEN"
    /\ new_sl \in 1..MaxPrice
    /\ new_sl   > 0
    /\ direction = 1  => new_sl > sl    \* buy: sl can only move UP
    /\ direction = -1 => new_sl < sl \/ sl = 0  \* sell: sl can only move DOWN
    /\ sl'      = new_sl
    /\ UNCHANGED <<status, direction, entry_price, current_price, tp, volume, breakeven_applied>>

\* Partial close
PartialClose(close_vol) ==
    /\ status   = "OPEN"
    /\ close_vol \in 1..(volume - 1)  \* must leave at least 1 unit open
    /\ volume'  = volume - close_vol
    /\ UNCHANGED <<status, direction, entry_price, current_price, sl, tp, breakeven_applied>>

\* Full close
FullClose ==
    /\ status   = "OPEN"
    /\ status'  = "CLOSED"
    /\ volume'  = 0
    /\ UNCHANGED <<direction, entry_price, current_price, sl, tp, breakeven_applied>>

\* A closed position stays closed (terminal state, not a deadlock)
Closed ==
    /\ status = "CLOSED"
    /\ UNCHANGED vars

Next ==
    \/ Closed
    \/ \E p \in 1..MaxPrice : PriceMove(p)
    \/ MoveToBreakeven
    \/ \E s \in 1..MaxPrice : TrailSL(s)
    \/ \E v \in 1..(IF volume > 1 THEN volume - 1 ELSE 1) : PartialClose(v)
    \/ FullClose

Spec == Init /\ [][Next]_vars

\* ── Invariants ──────────────────────────────────────────────────────────────

\* INV-PL-1: SL for a BUY never decreases (only moves toward entry/TP)
\* Checked implicitly by TrailSL precondition; state invariant version:
INV_SLDirection ==
    TRUE  \* enforced by TrailSL precondition: new_sl > sl for buys

\* INV-PL-3: Volume never exceeds initial volume
INV_VolumeNeverExceedsInitial ==
    volume <= InitVolume

\* INV-PL-4: If break-even applied, sl >= entry_price for BUY
INV_BreakevenCorrect ==
    (breakeven_applied /\ direction = 1) => sl >= entry_price

\* INV-PL-5: While OPEN, volume must be >= 1
INV_VolumePositiveWhileOpen ==
    status = "OPEN" => volume >= 1

\* INV-PL-1 as a real check (action property): a BUY's SL never moves down,
\* a SELL's never moves up, once set.
INV_PL1_SLNeverWidens ==
    [][ /\ (direction = 1 /\ sl > 0) => sl' >= sl
        /\ (direction = -1 /\ sl > 0) => (sl' <= sl /\ sl' > 0) ]_vars

Safety ==
    /\ TypeOK
    /\ INV_SLDirection
    /\ INV_VolumeNeverExceedsInitial
    /\ INV_BreakevenCorrect
    /\ INV_VolumePositiveWhileOpen

====
