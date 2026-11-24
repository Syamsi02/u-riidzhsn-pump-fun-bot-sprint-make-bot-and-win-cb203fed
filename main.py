"""Pump.fun Bot Sprint entry: buy the first feasible moment, exit fast.

Idea (from the dev window's own shape, not from any one coin): a fresh
pump.fun coin's first minutes decide it, and the size cap is what decides
when a bot can act at all -- a buy on one coin in one batch is capped at 1%
of its real quote reserves, so a meaningful buy only becomes possible once
the coin has about 1.5 SOL of real reserves. Six earlier versions of this
bot were graded on the dev window: buying late, near the top, lost the
most; scaling into flat coins lost more; and chasing only the coins that had
already doubled lost the most of all, because the big stakes the cap allows
on those coins ride the fade down. The least-bad shape is this one: one
small buy per coin at the first feasible batch, then out fast -- a close
take-profit, a close stop-loss, and a short trade-count exit, so no position
sits long enough to ride the fade. No clock, no randomness, no network:
every decision is a pure function of the batches seen so far, so two runs
agree byte for byte.

Mechanics (agent-fun/replay/v2, via vendor/makefun_bot.py):
- price of a coin = virtual_quote_reserves / virtual_token_reserves.
- a buy on one coin in one batch is capped at 1% of its real quote reserves;
  buys below MIN_BUY never count toward the 20-coin round-trip guardrail.
- at most 5 open positions; each coin is bought at most once.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import makefun_bot  # noqa: E402  (after the sys.path line: -P leaves the script dir off sys.path)

LAMPORTS_PER_SOL = makefun_bot.LAMPORTS_PER_SOL

# --- sizing -----------------------------------------------------------------
STAKE_LAMPORTS = 300_000_000          # 0.30 SOL target stake per coin (cap usually binds first)
MIN_BUY_LAMPORTS = 15_000_000         # 0.015 SOL: safely above the 0.01 SOL round-trip bar
MAX_SLIPPAGE_BPS = 10_000             # wide open, as the kit's example does

# --- entry ------------------------------------------------------------------
MAX_ENTRY_MULT = 1.60                 # never chase a coin already up 60%+
MAX_AGE_SLOTS = 8_400                 # only coins first seen within ~30 min

# --- exits ------------------------------------------------------------------
TAKE_PROFIT_MULT = 1.15               # sell all at +15% vs average entry cost
STOP_LOSS_MULT = 0.92                 # sell all at -8% vs average entry cost
MAX_TRADES_HELD = 10                  # exit once the coin has had 10 trades to move
MAX_HOLD_SLOTS = 8_400                # backstop for coins that go quiet (~30 min)


def _price(curve):
    """SOL per token from a curve's virtual reserves, or None when unknown."""
    if curve is None or curve.virtual_token_reserves <= 0:
        return None
    return curve.virtual_quote_reserves / curve.virtual_token_reserves


class Coin:
    """What the bot remembers about one mint, in first-seen order."""

    __slots__ = (
        "first_price", "first_slot", "trades",
        "decided", "entry_slot", "entry_trades", "peak_price",
    )

    def __init__(self, first_price, first_slot):
        self.first_price = first_price
        self.first_slot = first_slot
        self.trades = 0               # TradeEvents seen on this coin
        self.decided = False          # the buy was sent (or the coin was passed over)
        self.entry_slot = None        # slot the buy was decided on
        self.entry_trades = None      # trades counted when the buy was decided
        self.peak_price = None        # highest price seen since the buy


# Mint -> Coin, insertion ordered (never iterated as a set: determinism).
coins = {}
# Mints with a sell already in flight, so the same exit is not sent twice.
selling = {}


def on_hello(state):
    # Nothing to seed: the strategy is purely reactive to the replay.
    return None


def on_batch(state, batch):
    orders = []
    caps = state.hello["caps"]
    max_positions = caps["max_open_positions"]
    tx = state.tx_fee_lamports()
    free = state.free_cash_lamports()
    slots_free = max_positions - state.committed_position_count()

    # --- fold this batch's events into per-coin memory ----------------------
    for e in batch["events"]:
        mint = e["mint"]
        kind = e["event"]
        if kind == "CreateEvent":
            if mint not in coins:
                price = _price(state.curves.get(mint))
                if price is not None:
                    coins[mint] = Coin(price, e["slot"])
        elif kind == "TradeEvent":
            coin = coins.get(mint)
            if coin is None:
                price = _price(state.curves.get(mint))
                if price is None:
                    continue
                coin = Coin(price, e["slot"])
                coins[mint] = coin
            coin.trades += 1
        # CompleteEvent: the curve is done; the harness closes positions.

    # A sell that filled (or was rejected) is no longer in flight: the mint
    # leaves `selling` when no sell for it is pending and it is not held.
    pending_sells = {}
    for p in state.pending.values():
        if p.order["side"] == "sell":
            pending_sells[p.order["mint"]] = True
    for mint in list(selling.keys()):
        if mint not in state.positions and mint not in pending_sells:
            del selling[mint]

    # --- exits first: free slots and cash before considering new buys -------
    for mint in list(state.positions.keys()):
        holding = state.positions[mint]
        if holding is None or holding.token_amount <= 0:
            continue
        if mint in selling:
            continue
        coin = coins.get(mint)
        curve = state.curves.get(mint)
        price = _price(curve)
        if price is None or coin is None:
            continue
        if curve.complete:
            continue  # harness closes completed curves itself
        # True average entry cost from the filled position itself: the buy
        # was decided two slots earlier, so the decision-time price is stale.
        entry = holding.cost_lamports / holding.token_amount
        reason = None
        if price >= entry * TAKE_PROFIT_MULT:
            reason = "take-profit"
        elif price <= entry * STOP_LOSS_MULT:
            reason = "stop-loss"
        elif (coin.entry_trades is not None
                and coin.trades - coin.entry_trades >= MAX_TRADES_HELD):
            reason = "max-trades"
        elif coin.entry_slot is not None and batch["slot"] - coin.entry_slot >= MAX_HOLD_SLOTS:
            reason = "max-hold"
        if reason is None:
            continue
        if len(orders) >= caps["max_orders_per_batch"]:
            break
        orders.append({
            "side": "sell", "mint": mint,
            "token_amount": holding.token_amount,
            "max_slippage_bps": MAX_SLIPPAGE_BPS,
        })
        selling[mint] = True
        slots_free += 1  # the slot frees once this sell fills

    # --- entries: first batch where the cap allows a real stake -------------
    for e in batch["events"]:
        if e["event"] not in ("CreateEvent", "TradeEvent"):
            continue
        mint = e["mint"]
        coin = coins.get(mint)
        if coin is None or coin.decided:
            continue
        if mint in state.positions or mint in selling:
            continue
        if batch["slot"] - coin.first_slot > MAX_AGE_SLOTS:
            coin.decided = True
            continue
        curve = state.curves.get(mint)
        if curve is None or curve.complete:
            coin.decided = True
            continue
        price = _price(curve)
        if price is None:
            continue
        if price > coin.first_price * MAX_ENTRY_MULT:
            # Already ran away: chasing it is how the earlier versions lost.
            coin.decided = True
            continue
        cap = state.max_buy_lamports(mint)
        if cap < MIN_BUY_LAMPORTS:
            continue  # not feasible yet; retry on a later batch
        amount = min(STAKE_LAMPORTS, cap)
        if slots_free <= 0 or free < amount + tx:
            continue  # no slot / no cash; retry on a later batch
        if len(orders) >= caps["max_orders_per_batch"]:
            break
        orders.append({
            "side": "buy", "mint": mint,
            "quote_lamports": amount,
            "max_slippage_bps": MAX_SLIPPAGE_BPS,
        })
        coin.decided = True
        coin.entry_slot = batch["slot"]
        coin.entry_trades = coin.trades
        coin.peak_price = price
        free -= amount + tx
        slots_free -= 1

    return orders


def on_done(state, done):
    # Report the run to stderr (stdout carries replies only): final cash and
    # how many coins were traded, for the local grade log.
    print(
        "done: cash=%d lamports positions=%d coins_seen=%d batches=%d"
        % (state.cash, len(state.positions), len(coins), state.seq),
        file=sys.stderr,
    )


if __name__ == "__main__":
    makefun_bot.run(on_batch, on_hello=on_hello, on_done=on_done)
