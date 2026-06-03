# Funding-Carry Engine — v1 Design of Record

**Status:** scoped 2026-06-03 (not yet built). The first strategy candidate with a
*stationary, market-neutral, validated* edge. Read alongside `RESEARCH_FINDINGS_2026-06-03.md`
(why the majors are tapped) and `SIGNAL_POOL_DESIGN.md` (where this signal goes).

## The thesis (validated, stationary)
BTC perpetual funding carry — delta-neutral long-spot / short-perp, collect funding:

| 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| +30.6% | +4.2% | +7.9% | +12.0% | +5.1% |

**Positive every year** — stationary (unlike FCARRY's 2022-only, XS_MOM's decay), and
**market-neutral** (the spot hedge cancels price risk). **Low-turnover** (enter, hold,
collect, exit) → the ~60 bps round-trip cost is negligible over a multi-month hold, so
net ≈ gross ≈ **~+8%/yr**. Sub-10% standalone, but a **near-ideal pool member**: a smooth,
uncorrelated, low-vol stream → high marginal-Sharpe contribution.

## Key discovery — most of the machinery already exists
- **`BacktestFundingCostService`** — the backtest already accrues funding (paid *and*
  received) per position over the hold window (`notional × Σ funding_rate`, sign by side).
- **`BacktestTradeExecutorService:527`** — `realizedPnl = priceP&L − fundingCost`. **Funding
  is a separable term**, not baked into the price P&L.
- **`XsBook`** — a dollar-neutral multi-leg book (built for cross-sectional).

So the carry is **not** a from-scratch two-leg engine.

## v1 — delta-neutral carry via price-neutralization (NO spot data needed)
A delta-neutral carry's P&L = `funding − Δ(basis)`, and the basis (perp−spot) is tiny and
funding-pinned, so **Δbasis ≈ 0** over a hold. Therefore model the position as: hold the
perp on the funding-**receiving** side, **accrue funding** (exists), and **neutralize the
directional P&L** (the spot hedge offsets it — approximated by zeroing the price term).

**Implementation (small, bounded):**
1. **Carry-mode flag** — a spec param / `account_strategy` column, e.g. `delta_neutral_carry`.
2. **`BacktestTradeExecutorService` (~line 527)** — in carry mode, `realizedPnl := −fundingCost`
   (drop the price-P&L term). *This one branch is the perfect-hedge approximation.*
3. **Carry strategy** — each period, position on the funding-**receiving** side
   (short perp when funding > 0, long when funding < 0; or stand aside when |funding| is
   below a floor). Low turnover: re-evaluate only on funding-sign change or a slow cadence.
   Reads `funding_rate_8h` (already in `feature_store`). Could be a new minimal engine OR a
   "carry mode" on the existing `FCARRY` executor (which already reads funding).

Equity curve = `Σ funding − (entry/exit cost)`. Validation of the *level* is done (+8%/yr,
stationary); the backtest confirms the **net** after the (low) turnover cost — the XS_MOM
lesson applied: confirm net, don't assume.

## v2 (precise, deferred) — true two-leg basis carry
Plumb **spot** klines (a new Binance feed) + a true two-leg book (long spot + short perp via
`XsBook`-style legs) → captures `Δbasis` exactly + the spot-leg fees/borrow. Bigger (new data
feed + two-leg accounting). Only needed if the v1 approximation proves too coarse.

## Build scope (v1)
| Item | Where | Size |
|---|---|---|
| Carry-mode P&L branch (`realizedPnl := −fundingCost`) | `BacktestTradeExecutorService` | small |
| Carry-mode flag | spec param or `account_strategy` (Flyway if column) | small |
| Carry strategy (receiving-side, low-turnover) | new engine OR `FCARRY` carry mode | small–med |
| Funding data coverage | `funding_rate_history` 2022→now (BTC/ETH present ✓) | done |
| Escalate | SIGNIFICANT_EDGE → walk-forward → `/pool/evaluate` | exists |

## Risks / honesty notes
- **Approximation:** v1 zeros the price P&L (perfect hedge). Real carry has basis noise + the
  spot leg isn't free (fees, borrow). v1 slightly *over*-states; v2 (spot) is precise. Treat a
  v1 pass as "promising, confirm with v2" — same discipline that caught XS_MOM.
- **Don't leave directional exposure:** the price-neutralization must be exact, or the
  "carry" secretly carries market beta (→ a disguised directional bet, the FCARRY trap).
- **Sub-10% by design** — this is a *pool* candidate, not a standalone graduation. Its value
  is being uncorrelated + market-neutral (high Sharpe), which the marginal-Sharpe admission
  rewards.
