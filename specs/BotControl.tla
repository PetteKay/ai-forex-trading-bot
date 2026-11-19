---- MODULE BotControl ----
\* ============================================================
\* Tier-0 Bot Control — bot.py controlled from an MCP client
\* (Claude Code, Claude Desktop, Codex, ChatGPT ...)
\*
\* Actors on ONE trading account:
\*   * bot.py processes   — each start is a new instance id (1..MaxStarts)
\*   * the MCP server     — places / closes / books trades itself only
\*                          while no bot is running
\* Shared:
\*   * account lock       — an OS file lock (observability/bot.lock);
\*                          the OS releases it when its holder dies
\*   * control file       — pause flag, stop requests (by instance id),
\*                          close requests (by ticket); written by MCP,
\*                          read by the bot before every order
\*   * risk governor      — the daily-loss kill switch (RiskGovernor.tla)
\*
\* Code: bot_control.py, bot.py, mcp_server.py, trade_booking.py
\*
\* Invariants:
\*   INV-BC-1  SingleTrader       at most one actor can trade the account
\*   INV-BC-2  RunnerHoldsLock    a running bot holds the account lock
\*   INV-BC-3  BookedOnce         every closed trade is booked at most once
\*                                (governor P&L, journal, ML sample, lesson)
\*   INV-BC-4  StopHitsTarget     a stop request only stops the instance it
\*                                names — never a later instance
\*   INV-BC-5  PauseConfirmed     once the bot confirms a pause, it has no
\*                                order in flight and sends none until resumed
\*   INV-BC-6  LimitsOnlyTighten  AI-set limits never exceed the human's
\*                                config.py value, even after a later edit
\*   ACT-BC-7  LiveNeedsHuman     a live bot is only started when the human
\*                                has set live mode
\*   ACT-BC-8  KillLatchKept      no control action clears the kill switch;
\*                                only the day reset does
\*   ACT-BC-9  NoEntryWhilePaused no MCP order while paused; no bot order
\*                                after the bot has seen the pause
\* Liveness (weak fairness on the bot and on MCP booking):
\*   LIVE-BC-1 StopTakesEffect    a requested stop eventually stops it
\*   LIVE-BC-2 EveryCloseBooked   every closed trade is eventually booked
\*   LIVE-BC-3 CloseRequestServed a close request to a running bot is
\*                                eventually carried out (or the bot exits)
\* ============================================================
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    MaxStarts,     \* bot starts to explore (instance ids 1..MaxStarts)
    Tickets,       \* trade tickets to explore
    MaxLimit       \* largest value of a risk limit (e.g. max open positions)

ASSUME /\ MaxStarts \in Nat /\ MaxStarts > 0
       /\ IsFiniteSet(Tickets) /\ Tickets # {}
       /\ MaxLimit \in Nat /\ MaxLimit > 0

Instances == 1..MaxStarts
NoOne     == 0                        \* lock is free
MCP       == MaxStarts + 1            \* lock held by the MCP server

VARIABLES
    proc,        \* proc[i] \in {"none","waiting","running","exited"}
    exitCause,   \* exitCause[i] \in {"none","stop","force","crash","lostlock"}
    live,        \* live[i]: instance i was started in live mode
    lock,        \* NoOne, an instance id, or MCP
    paused,      \* control file: pause flag
    seen,        \* the running bot's last read of the pause flag
    acked,       \* TRUE once the bot has read the latest control change
    inFlight,    \* the bot passed its pause check and is sending an order
    stopReq,     \* control file: instance ids asked to stop
    closeReq,    \* control file: <<instance, ticket>> close requests
    trade,       \* trade[t] \in {"none","open","closed","booked"}
    bookCount,   \* bookCount[t]: how many times trade t was booked
    bookers,     \* bookers[t]: actors that read trade t as "closed, not
                 \* booked" and are about to write the booking
    kill,        \* governor kill switch (daily loss cap hit)
    humanLive,   \* config.py: MCP_TRADING_MODE = "live"
    humanLimit,  \* config.py value of a limit (the ceiling)
    aiLimit      \* override set through MCP (0 = none)

vars == <<proc, exitCause, live, lock, paused, seen, acked, inFlight, stopReq,
          closeReq, trade, bookCount, bookers, kill, humanLive, humanLimit, aiLimit>>

Actors == Instances \cup {MCP}

Running == {i \in Instances : proc[i] = "running"}

\* The limit the bot actually uses (bot_control.effective_limit)
EffectiveLimit == IF aiLimit = 0 THEN humanLimit ELSE IF aiLimit < humanLimit THEN aiLimit ELSE humanLimit

TypeOK ==
    /\ proc      \in [Instances -> {"none", "waiting", "running", "exited"}]
    /\ exitCause \in [Instances -> {"none", "stop", "force", "crash", "lostlock"}]
    /\ live      \in [Instances -> BOOLEAN]
    /\ lock      \in {NoOne, MCP} \cup Instances
    /\ paused \in BOOLEAN /\ seen \in BOOLEAN /\ acked \in BOOLEAN /\ inFlight \in BOOLEAN
    /\ stopReq   \subseteq Instances
    /\ closeReq  \subseteq (Instances \X Tickets)
    /\ trade     \in [Tickets -> {"none", "open", "closed", "booked"}]
    /\ bookCount \in [Tickets -> 0..(MaxStarts + 1)]
    /\ bookers   \in [Tickets -> SUBSET Actors]
    /\ kill \in BOOLEAN /\ humanLive \in BOOLEAN
    /\ humanLimit \in 1..MaxLimit
    /\ aiLimit   \in 0..MaxLimit

Init ==
    /\ proc = [i \in Instances |-> "none"]
    /\ exitCause = [i \in Instances |-> "none"]
    /\ live = [i \in Instances |-> FALSE]
    /\ lock = NoOne
    /\ paused = FALSE /\ seen = FALSE /\ acked = TRUE /\ inFlight = FALSE
    /\ stopReq = {} /\ closeReq = {}
    /\ trade = [t \in Tickets |-> "none"]
    /\ bookCount = [t \in Tickets |-> 0]
    /\ bookers = [t \in Tickets |-> {}]
    /\ kill = FALSE
    /\ humanLive \in BOOLEAN
    /\ humanLimit \in 1..MaxLimit
    /\ aiLimit = 0

\* ------------------------------------------------------------
\* Bot process lifecycle (bot.py + bot_control.BotLock)
\* ------------------------------------------------------------
\* start_bot launches the next instance (live only if the human chose live)
Launch(i, isLive) ==
    /\ proc[i] = "none"
    /\ \A j \in Instances : j < i => proc[j] # "none"
    /\ isLive => humanLive
    /\ proc' = [proc EXCEPT ![i] = "waiting"]
    /\ live' = [live EXCEPT ![i] = isLive]
    /\ UNCHANGED <<exitCause, lock, paused, seen, acked, inFlight, stopReq, closeReq,
                   trade, bookCount, bookers, kill, humanLive, humanLimit, aiLimit>>

\* Takes the free account lock and starts trading. Reads control on start.
Acquire(i) ==
    /\ proc[i] = "waiting" /\ lock = NoOne
    /\ lock' = i
    /\ proc' = [proc EXCEPT ![i] = "running"]
    /\ seen' = paused /\ acked' = TRUE
    /\ UNCHANGED <<exitCause, live, paused, inFlight, stopReq, closeReq, trade,
                   bookCount, bookers, kill, humanLive, humanLimit, aiLimit>>

\* Another bot already holds the lock: this one exits ("already running").
\* (If the MCP server holds it, the bot keeps waiting: MCP holds it briefly.)
LoseRace(i) ==
    /\ proc[i] = "waiting" /\ lock \in Instances
    /\ proc' = [proc EXCEPT ![i] = "exited"]
    /\ exitCause' = [exitCause EXCEPT ![i] = "lostlock"]
    /\ UNCHANGED <<live, lock, paused, seen, acked, inFlight, stopReq, closeReq, trade,
                   bookCount, bookers, kill, humanLive, humanLimit, aiLimit>>

Exit(i, cause) ==
    /\ proc' = [proc EXCEPT ![i] = "exited"]
    /\ exitCause' = [exitCause EXCEPT ![i] = cause]
    /\ lock' = NoOne                         \* the OS frees the file lock
    /\ inFlight' = FALSE
    /\ bookers' = [t \in Tickets |-> bookers[t] \ {i}]   \* booking abandoned
    /\ UNCHANGED <<live, paused, seen, acked, stopReq, closeReq, trade, bookCount,
                   kill, humanLive, humanLimit, aiLimit>>

\* Graceful stop: the bot sees a stop request naming ITS instance id
GracefulStop(i) == proc[i] = "running" /\ i \in stopReq /\ ~inFlight /\ Exit(i, "stop")

\* stop_bot(force=True) after the grace period: only for the named instance
ForceKill(i) == proc[i] = "running" /\ i \in stopReq /\ Exit(i, "force")

Crash(i) == proc[i] = "running" /\ Exit(i, "crash")

\* ------------------------------------------------------------
\* Control file (written by MCP tools)
\* ------------------------------------------------------------
\* stop_bot names the instance it read from the status file
RequestStop(i) ==
    /\ proc[i] = "running"
    /\ stopReq' = stopReq \cup {i}
    /\ UNCHANGED <<proc, exitCause, live, lock, paused, seen, acked, inFlight, closeReq,
                   trade, bookCount, bookers, kill, humanLive, humanLimit, aiLimit>>

SetPause(p) ==
    /\ paused' = p /\ acked' = FALSE
    /\ UNCHANGED <<proc, exitCause, live, lock, seen, inFlight, stopReq, closeReq,
                   trade, bookCount, bookers, kill, humanLive, humanLimit, aiLimit>>

\* The running bot reads the control file (each poll, and before every order)
BotReadControl(i) ==
    /\ proc[i] = "running" /\ ~inFlight
    /\ seen' = paused /\ acked' = TRUE
    /\ UNCHANGED <<proc, exitCause, live, lock, paused, inFlight, stopReq, closeReq,
                   trade, bookCount, bookers, kill, humanLive, humanLimit, aiLimit>>

\* update_settings: the AI may only set a value at or below the human's
SetAiLimit(v) ==
    /\ v \in 1..humanLimit
    /\ aiLimit' = v
    /\ UNCHANGED <<proc, exitCause, live, lock, paused, seen, acked, inFlight, stopReq,
                   closeReq, trade, bookCount, bookers, kill, humanLive, humanLimit>>

ClearAiLimit ==
    /\ aiLimit' = 0
    /\ UNCHANGED <<proc, exitCause, live, lock, paused, seen, acked, inFlight, stopReq,
                   closeReq, trade, bookCount, bookers, kill, humanLive, humanLimit>>

\* The human edits config.py (may lower OR raise the ceiling, or switch live)
HumanEdit(h, l) ==
    /\ humanLimit' = h /\ humanLive' = l
    /\ UNCHANGED <<proc, exitCause, live, lock, paused, seen, acked, inFlight, stopReq,
                   closeReq, trade, bookCount, bookers, kill, aiLimit>>

\* close_position while a bot runs: a request addressed to that instance
RequestClose(i, t) ==
    /\ trade[t] = "open" /\ proc[i] = "running"
    /\ closeReq' = closeReq \cup {<<i, t>>}
    /\ UNCHANGED <<proc, exitCause, live, lock, paused, seen, acked, inFlight, stopReq,
                   trade, bookCount, bookers, kill, humanLive, humanLimit, aiLimit>>

\* ------------------------------------------------------------
\* Trading by the running bot (holds the lock)
\* ------------------------------------------------------------
\* Pause check right before the order: re-read control, go only if not paused.
\* A pending stop for this instance wins: no new order is started.
BotBeginOrder(i) ==
    /\ proc[i] = "running" /\ ~inFlight /\ ~kill /\ i \notin stopReq
    /\ seen' = paused /\ acked' = TRUE
    /\ inFlight' = ~paused
    /\ UNCHANGED <<proc, exitCause, live, lock, paused, stopReq, closeReq, trade,
                   bookCount, bookers, kill, humanLive, humanLimit, aiLimit>>

BotFill(i, t) ==
    /\ proc[i] = "running" /\ inFlight /\ trade[t] = "none"
    /\ trade' = [trade EXCEPT ![t] = "open"]
    /\ inFlight' = FALSE
    /\ UNCHANGED <<proc, exitCause, live, lock, paused, seen, acked, stopReq, closeReq,
                   bookCount, bookers, kill, humanLive, humanLimit, aiLimit>>

BotOrderFails(i) ==
    /\ proc[i] = "running" /\ inFlight
    /\ inFlight' = FALSE
    /\ UNCHANGED <<proc, exitCause, live, lock, paused, seen, acked, stopReq, closeReq,
                   trade, bookCount, bookers, kill, humanLive, humanLimit, aiLimit>>

BotClose(i, t) ==
    /\ proc[i] = "running" /\ <<i, t>> \in closeReq /\ trade[t] = "open"
    /\ trade' = [trade EXCEPT ![t] = "closed"]
    /\ closeReq' = closeReq \ {<<i, t>>}
    /\ UNCHANGED <<proc, exitCause, live, lock, paused, seen, acked, inFlight, stopReq,
                   bookCount, bookers, kill, humanLive, humanLimit, aiLimit>>

\* A close request for a trade that is no longer open is dropped
BotDropClose(i, t) ==
    /\ proc[i] = "running" /\ <<i, t>> \in closeReq /\ trade[t] # "open"
    /\ closeReq' = closeReq \ {<<i, t>>}
    /\ UNCHANGED <<proc, exitCause, live, lock, paused, seen, acked, inFlight, stopReq,
                   trade, bookCount, bookers, kill, humanLive, humanLimit, aiLimit>>

\* Booking = read the journal (trade closed, row still OPEN) then write it
\* (journal CLOSED, governor P&L, blacklist, ML sample, lesson). Only the lock
\* holder may book. The bot books its own closes AND trades that closed while
\* no bot ran (orphans); the MCP server books only while it holds the lock.
BeginBook(a, t) ==
    /\ lock = a /\ trade[t] = "closed" /\ a \notin bookers[t]
    /\ bookers' = [bookers EXCEPT ![t] = @ \cup {a}]
    /\ UNCHANGED <<proc, exitCause, live, lock, paused, seen, acked, inFlight, stopReq,
                   closeReq, trade, bookCount, kill, humanLive, humanLimit, aiLimit>>

\* The write does not re-read: it acts on what BeginBook saw
EndBook(a, t) ==
    /\ lock = a /\ a \in bookers[t]
    /\ trade' = [trade EXCEPT ![t] = "booked"]
    /\ bookCount' = [bookCount EXCEPT ![t] = @ + 1]
    /\ bookers' = [bookers EXCEPT ![t] = @ \ {a}]
    /\ UNCHANGED <<proc, exitCause, live, lock, paused, seen, acked, inFlight, stopReq,
                   closeReq, kill, humanLive, humanLimit, aiLimit>>

BotBook(i, t) == proc[i] = "running" /\ (BeginBook(i, t) \/ EndBook(i, t))

\* ------------------------------------------------------------
\* MCP server: trades and books only while it holds the lock
\* ------------------------------------------------------------
McpAcquire ==
    /\ lock = NoOne
    /\ lock' = MCP
    /\ UNCHANGED <<proc, exitCause, live, paused, seen, acked, inFlight, stopReq,
                   closeReq, trade, bookCount, bookers, kill, humanLive, humanLimit, aiLimit>>

McpRelease ==
    /\ lock = MCP
    /\ \A t \in Tickets : MCP \notin bookers[t]     \* held for the whole booking
    /\ lock' = NoOne
    /\ UNCHANGED <<proc, exitCause, live, paused, seen, acked, inFlight, stopReq,
                   closeReq, trade, bookCount, bookers, kill, humanLive, humanLimit, aiLimit>>

McpPlace(t) ==
    /\ lock = MCP /\ ~paused /\ ~kill /\ humanLive /\ trade[t] = "none"
    /\ trade' = [trade EXCEPT ![t] = "open"]
    /\ UNCHANGED <<proc, exitCause, live, lock, paused, seen, acked, inFlight, stopReq,
                   closeReq, bookCount, bookers, kill, humanLive, humanLimit, aiLimit>>

McpClose(t) ==
    /\ lock = MCP /\ trade[t] = "open"
    /\ trade' = [trade EXCEPT ![t] = "closed"]
    /\ UNCHANGED <<proc, exitCause, live, lock, paused, seen, acked, inFlight, stopReq,
                   closeReq, bookCount, bookers, kill, humanLive, humanLimit, aiLimit>>

McpBook(t) == BeginBook(MCP, t) \/ EndBook(MCP, t)

\* ------------------------------------------------------------
\* Market and calendar
\* ------------------------------------------------------------
StopOrTargetHit(t) ==
    /\ trade[t] = "open"
    /\ trade' = [trade EXCEPT ![t] = "closed"]
    /\ UNCHANGED <<proc, exitCause, live, lock, paused, seen, acked, inFlight, stopReq,
                   closeReq, bookCount, bookers, kill, humanLive, humanLimit, aiLimit>>

LossCapHit ==
    /\ kill' = TRUE
    /\ UNCHANGED <<proc, exitCause, live, lock, paused, seen, acked, inFlight, stopReq,
                   closeReq, trade, bookCount, bookers, humanLive, humanLimit, aiLimit>>

DayReset ==
    /\ kill' = FALSE
    /\ UNCHANGED <<proc, exitCause, live, lock, paused, seen, acked, inFlight, stopReq,
                   closeReq, trade, bookCount, bookers, humanLive, humanLimit, aiLimit>>

\* ------------------------------------------------------------
BotSteps(i) ==
    \/ Acquire(i) \/ LoseRace(i) \/ GracefulStop(i) \/ BotReadControl(i)
    \/ BotBeginOrder(i) \/ BotOrderFails(i)
    \/ \E t \in Tickets : BotFill(i, t) \/ BotClose(i, t) \/ BotDropClose(i, t) \/ BotBook(i, t)

Next ==
    \/ \E i \in Instances : \E l \in BOOLEAN : Launch(i, l)
    \/ \E i \in Instances : BotSteps(i) \/ ForceKill(i) \/ Crash(i) \/ RequestStop(i)
    \/ \E p \in BOOLEAN : SetPause(p)
    \/ \E v \in 1..MaxLimit : SetAiLimit(v)
    \/ ClearAiLimit
    \/ \E h \in 1..MaxLimit, l \in BOOLEAN : HumanEdit(h, l)
    \/ McpAcquire \/ McpRelease
    \/ \E i \in Instances, t \in Tickets : RequestClose(i, t)
    \/ \E t \in Tickets : McpPlace(t) \/ McpClose(t) \/ McpBook(t)
                          \/ StopOrTargetHit(t)
    \/ LossCapHit \/ DayReset

\* Fairness: a running bot keeps working; the MCP server is eventually used again
\* and books what it can. Neither crashes nor launches are forced.
Fairness ==
    /\ \A i \in Instances : WF_vars(GracefulStop(i))
    /\ \A i \in Instances : WF_vars(BotReadControl(i)) /\ WF_vars(BotOrderFails(i))
    /\ \A i \in Instances, t \in Tickets :
          WF_vars(BotBook(i, t)) /\ WF_vars(BotClose(i, t)) /\ WF_vars(BotDropClose(i, t))
    /\ \A i \in Instances : WF_vars(Acquire(i)) /\ WF_vars(LoseRace(i))
    /\ WF_vars(McpRelease)
    /\ SF_vars(McpAcquire)
    /\ \A t \in Tickets : SF_vars(BeginBook(MCP, t)) /\ WF_vars(EndBook(MCP, t))

Spec == Init /\ [][Next]_vars
LiveSpec == Spec /\ Fairness

\* ============================================================
\* SAFETY
\* ============================================================
INV_BC1_SingleTrader ==
    Cardinality(Running) + (IF lock = MCP THEN 1 ELSE 0) <= 1

INV_BC2_RunnerHoldsLock ==
    \A i \in Instances : proc[i] = "running" => lock = i

INV_BC3_BookedOnce ==
    \A t \in Tickets : /\ bookCount[t] <= 1
                       /\ (trade[t] = "booked") <=> (bookCount[t] = 1)
                       /\ Cardinality(bookers[t]) <= 1

INV_BC4_StopHitsTarget ==
    \A i \in Instances : exitCause[i] \in {"stop", "force"} => i \in stopReq

\* After the bot confirms it has read a pause, nothing of its is in flight
INV_BC5_PauseConfirmed ==
    \A i \in Running : (paused /\ acked) => (seen /\ ~inFlight)

INV_BC6_LimitsOnlyTighten ==
    EffectiveLimit <= humanLimit

Safety ==
    /\ TypeOK
    /\ INV_BC1_SingleTrader
    /\ INV_BC2_RunnerHoldsLock
    /\ INV_BC3_BookedOnce
    /\ INV_BC4_StopHitsTarget
    /\ INV_BC5_PauseConfirmed
    /\ INV_BC6_LimitsOnlyTighten

\* ACT-BC-8: only DayReset turns the kill switch off
ACT_BC8_KillLatchKept ==
    [][ (kill /\ ~kill') => (UNCHANGED <<bookers, proc, lock, paused, stopReq, trade, aiLimit, humanLimit>>) ]_vars

\* ACT-BC-9: no MCP order while paused; no bot fill unless its last read saw "not paused"
ACT_BC9_NoEntryWhilePaused ==
    [][ \A t \in Tickets :
          (trade[t] = "none" /\ trade'[t] = "open") =>
             \/ (lock = MCP /\ ~paused)
             \/ (\E i \in Running : inFlight /\ ~seen) ]_vars

\* ACT-BC-7: launching live requires the human's live switch at that moment
ACT_BC7_LiveNeedsHuman ==
    [][ \A i \in Instances : (proc[i] = "none" /\ proc'[i] = "waiting" /\ live'[i]) => humanLive ]_vars

\* ============================================================
\* LIVENESS
\* ============================================================
LIVE_BC1_StopTakesEffect ==
    \A i \in Instances : (proc[i] = "running" /\ i \in stopReq) ~> (proc[i] = "exited")

LIVE_BC2_EveryCloseBooked ==
    \A t \in Tickets : (trade[t] = "closed") ~> (trade[t] = "booked")

LIVE_BC3_CloseRequestServed ==
    \A i \in Instances, t \in Tickets :
        (<<i, t>> \in closeReq) ~> (<<i, t>> \notin closeReq \/ proc[i] = "exited")

====
