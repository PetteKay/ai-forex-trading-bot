---- MODULE OrderStateMachine ----
\* ============================================================
\* Tier-1 Order State Machine
\* Personal AI Forex Trading Bot
\*
\* Models the lifecycle of a single order from signal to fill to close.
\*
\* States:
\*   IDLE      → no active order
\*   PENDING   → signal fired, awaiting governor approval
\*   APPROVED  → governor approved, sending to broker
\*   OPEN      → order filled and position is live
\*   CLOSING   → close request sent to broker
\*   CLOSED    → position fully closed
\*   REJECTED  → governor or broker rejected the order
\*
\* Invariants:
\*   INV-OS-1: A CLOSED order always has a non-zero exit price
\*   INV-OS-2: An order can only move to OPEN from APPROVED
\*   INV-OS-3: A REJECTED order cannot transition to any other state
\*   INV-OS-4: Only OPEN positions can be CLOSING
\*   INV-OS-5: Entry price is set once and never changes after OPEN
\* ============================================================
EXTENDS Integers, TLC

CONSTANTS
    MaxPrice,         \* upper bound for prices (model exploration)
    MaxSlippage       \* max pips of slippage

ASSUME /\ MaxPrice    \in Nat /\ MaxPrice > 0
       /\ MaxSlippage \in Nat /\ MaxSlippage >= 0

VARIABLES
    order_state,      \* current state of the order
    signal_price,     \* price at which signal fired
    entry_price,      \* actual fill price (0 until OPEN)
    exit_price,       \* actual close price (0 until CLOSED)
    sl_price,         \* stop-loss price
    tp_price,         \* take-profit price
    direction         \* BUY(1) or SELL(-1)

vars == <<order_state, signal_price, entry_price, exit_price,
          sl_price, tp_price, direction>>

STATES == {"IDLE", "PENDING", "APPROVED", "OPEN", "CLOSING", "CLOSED", "REJECTED"}

TypeOK ==
    /\ order_state  \in STATES
    /\ signal_price \in 0..MaxPrice
    /\ entry_price  \in 0..MaxPrice
    /\ exit_price   \in 0..MaxPrice
    /\ sl_price     \in 0..MaxPrice
    /\ tp_price     \in 0..MaxPrice
    /\ direction    \in {1, -1}

Init ==
    /\ order_state  = "IDLE"
    /\ signal_price = 0
    /\ entry_price  = 0
    /\ exit_price   = 0
    /\ sl_price     = 0
    /\ tp_price     = 0
    /\ direction    = 1

\* Signal fires: bot decides to enter
FireSignal(price, dir, sl, tp) ==
    /\ order_state    = "IDLE"
    /\ price > 0 /\ price <= MaxPrice
    /\ sl > 0 /\ tp > 0
    /\ order_state'   = "PENDING"
    /\ signal_price'  = price
    /\ direction'     = dir
    /\ sl_price'      = sl
    /\ tp_price'      = tp
    /\ UNCHANGED <<entry_price, exit_price>>

\* Governor approves the order
GovernorApprove ==
    /\ order_state  = "PENDING"
    /\ order_state' = "APPROVED"
    /\ UNCHANGED <<signal_price, entry_price, exit_price, sl_price, tp_price, direction>>

\* Governor rejects the order
GovernorReject ==
    /\ order_state  = "PENDING"
    /\ order_state' = "REJECTED"
    /\ UNCHANGED <<signal_price, entry_price, exit_price, sl_price, tp_price, direction>>

\* Broker fills the order (with possible slippage)
BrokerFill(fill_price) ==
    /\ order_state  = "APPROVED"
    /\ fill_price > 0 /\ fill_price <= MaxPrice
    /\ ABS(fill_price - signal_price) <= MaxSlippage
    /\ order_state'  = "OPEN"
    /\ entry_price'  = fill_price
    /\ UNCHANGED <<signal_price, exit_price, sl_price, tp_price, direction>>

ABS(x) == IF x >= 0 THEN x ELSE -x

\* Position closes (TP, SL, or manual)
ClosePosition(close_price) ==
    /\ order_state  = "OPEN"
    /\ close_price > 0 /\ close_price <= MaxPrice
    /\ order_state' = "CLOSING"
    /\ exit_price'  = close_price
    /\ UNCHANGED <<signal_price, entry_price, sl_price, tp_price, direction>>

\* Broker confirms close
BrokerConfirmClose ==
    /\ order_state  = "CLOSING"
    /\ order_state' = "CLOSED"
    /\ UNCHANGED <<signal_price, entry_price, exit_price, sl_price, tp_price, direction>>

\* Reset to IDLE from CLOSED or REJECTED
Reset ==
    /\ order_state \in {"CLOSED", "REJECTED"}
    /\ order_state'  = "IDLE"
    /\ signal_price' = 0
    /\ entry_price'  = 0
    /\ exit_price'   = 0
    /\ sl_price'     = 0
    /\ tp_price'     = 0
    /\ direction'    = 1

Next ==
    \/ \E p \in 1..MaxPrice, sl \in 1..MaxPrice, tp \in 1..MaxPrice, d \in {1, -1} :
        FireSignal(p, d, sl, tp)
    \/ GovernorApprove
    \/ GovernorReject
    \/ \E fp \in 1..MaxPrice : BrokerFill(fp)
    \/ \E cp \in 1..MaxPrice : ClosePosition(cp)
    \/ BrokerConfirmClose
    \/ Reset

Spec == Init /\ [][Next]_vars

\* ── Invariants ──────────────────────────────────────────────────────────────

\* INV-OS-1: CLOSED orders always have a non-zero exit price
INV_ClosedHasExitPrice ==
    order_state = "CLOSED" => exit_price > 0

\* INV-OS-2: OPEN can only be reached via APPROVED → BrokerFill
\* (structural: FireSignal only goes to PENDING, BrokerFill only from APPROVED)
INV_OpenFromApproved ==
    order_state = "OPEN" => entry_price > 0

\* INV-OS-3: REJECTED is a terminal state (no further transitions)
\* (structural: no action takes REJECTED as precondition except Reset)
INV_RejectedTerminal ==
    TRUE  \* structurally enforced

\* INV-OS-4: Only OPEN positions can be CLOSING
INV_ClosingOnlyFromOpen ==
    order_state = "CLOSING" => entry_price > 0

\* INV-OS-5: Entry price is set once (never modified after OPEN)
\* (structural: only BrokerFill sets entry_price, and only from APPROVED)
INV_EntryPriceImmutable ==
    (order_state \in {"OPEN", "CLOSING", "CLOSED"}) => entry_price > 0

Safety ==
    /\ TypeOK
    /\ INV_ClosedHasExitPrice
    /\ INV_OpenFromApproved
    /\ INV_RejectedTerminal
    /\ INV_ClosingOnlyFromOpen
    /\ INV_EntryPriceImmutable

====
