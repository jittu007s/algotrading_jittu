"""Smart-Money-Concept detection primitives, kept pure and broker-free so
every function is unit-testable on synthetic candles (tests/test_smc.py).

These extend the sweep -> MSS -> FVG scanner in structure.py with the
explicit objects the SMC brief asks for:

  find_fvgs        - all bullish/bearish 3-candle imbalances, size-filtered,
                     freshness-checked against later price (unmitigated only),
                     ranked largest-first.
  find_order_blocks- last opposite-colour candle before an impulsive move
                     that broke the prior swing (BOS), displacement-validated,
                     mitigation-flagged.
  find_equal_levels- clustered swing highs/lows (equal highs / equal lows)
                     that mark resting-liquidity pools.

"No look-ahead" note: every object is only *created* using candles at or
before its own index; the freshness/mitigation flags are computed from
candles strictly AFTER creation, which is the correct causal direction for
"has this been filled yet as of now".
"""

from __future__ import annotations

from typing import List

from .models import FVG, Candle, EqualLevel, OrderBlock, SwingKind
from .structure import find_swings


def find_fvgs(candles: List[Candle], min_size: float = 0.0,
              require_unmitigated: bool = True) -> List[FVG]:
    """Classic 3-candle imbalance.

    Bullish FVG at i: low[i] > high[i-2]  -> gap [high[i-2], low[i]].
    Bearish FVG at i: high[i] < low[i-2]  -> gap [low[i], high[i-2]]  (stored
    low<high). `min_size` filters insignificant gaps. When
    `require_unmitigated`, a gap that any later candle has traded back
    through is dropped. Returned ranked by size, largest first.
    """
    out: List[FVG] = []
    for i in range(2, len(candles)):
        c1, c3 = candles[i - 2], candles[i]
        gap = None
        if c3.low > c1.high:                     # bullish imbalance
            gap = FVG(low=c1.high, high=c3.low, created_index=i, timestamp=c3.timestamp)
        elif c3.high < c1.low:                   # bearish imbalance
            gap = FVG(low=c3.high, high=c1.low, created_index=i, timestamp=c3.timestamp)
        if gap is None or (gap.high - gap.low) < min_size:
            continue
        if require_unmitigated:
            mid = gap.midpoint
            later = candles[i + 1:]
            # mitigated once a later candle's range covers the gap midpoint
            if any(c.low <= mid <= c.high for c in later):
                continue
        out.append(gap)
    out.sort(key=lambda g: g.high - g.low, reverse=True)
    return out


def find_order_blocks(candles: List[Candle], swing_k: int = 2,
                      displacement_mult: float = 1.5, avg_period: int = 10,
                      only_untested: bool = True) -> List[OrderBlock]:
    """Detect order blocks validated by a break of structure.

    For each confirmed swing, look for a displacement candle (body >=
    displacement_mult x average body) whose close breaks that swing; the
    order block is the last opposite-colour candle immediately before the
    displacement leg. `only_untested` drops blocks price has since revisited.
    """
    if len(candles) < swing_k * 2 + 3:
        return []
    swings = find_swings(candles, swing_k)
    bodies = [c.body for c in candles]

    def avg_body(i: int) -> float:
        lo = max(0, i - avg_period)
        window = bodies[lo:i] or [0.0]
        return sum(window) / len(window)

    blocks: List[OrderBlock] = []
    for i in range(swing_k + 1, len(candles)):
        c = candles[i]
        avg = avg_body(i)
        if avg <= 0 or c.body < displacement_mult * avg:
            continue

        # bullish displacement: strong up candle closing above a recent swing high
        recent_high = next((s for s in reversed(swings)
                            if s.kind == SwingKind.HIGH and s.index < i), None)
        recent_low = next((s for s in reversed(swings)
                           if s.kind == SwingKind.LOW and s.index < i), None)

        if c.bullish and recent_high is not None and c.close > recent_high.price:
            ob_idx = _last_opposite(candles, i, want_bearish=True)
            if ob_idx is not None:
                blocks.append(_make_ob(candles, ob_idx, SwingKind.LOW, c.body))
        elif (not c.bullish) and recent_low is not None and c.close < recent_low.price:
            ob_idx = _last_opposite(candles, i, want_bearish=False)
            if ob_idx is not None:
                blocks.append(_make_ob(candles, ob_idx, SwingKind.HIGH, c.body))

    if only_untested:
        blocks = [b for b in blocks if not _mitigated(candles, b)]
    # newest first (most relevant), de-duplicated by index
    seen, uniq = set(), []
    for b in sorted(blocks, key=lambda b: b.index, reverse=True):
        if b.index not in seen:
            seen.add(b.index)
            uniq.append(b)
    return uniq


def _last_opposite(candles: List[Candle], disp_i: int, want_bearish: bool) -> int | None:
    """Index of the last candle before disp_i with the opposite colour."""
    for j in range(disp_i - 1, -1, -1):
        if candles[j].bullish != want_bearish:   # want_bearish -> candle is bearish
            return j
    return None


def _make_ob(candles: List[Candle], idx: int, kind: SwingKind, displacement: float) -> OrderBlock:
    c = candles[idx]
    return OrderBlock(kind=kind, low=c.low, high=c.high, index=idx,
                      timestamp=c.timestamp, displacement=displacement)


def _mitigated(candles: List[Candle], ob: OrderBlock) -> bool:
    mid = ob.midpoint
    return any(c.low <= mid <= c.high for c in candles[ob.index + 2:])


def find_equal_levels(candles: List[Candle], swing_k: int = 2,
                      tolerance: float = 3.0, min_count: int = 2) -> List[EqualLevel]:
    """Cluster confirmed swing highs (equal highs) and swing lows (equal
    lows) that sit within `tolerance` points of each other - resting
    liquidity pools that price is drawn to sweep."""
    swings = find_swings(candles, swing_k)
    out: List[EqualLevel] = []
    for kind in (SwingKind.HIGH, SwingKind.LOW):
        pts = [s for s in swings if s.kind == kind]
        used = [False] * len(pts)
        for a in range(len(pts)):
            if used[a]:
                continue
            cluster = [pts[a]]
            used[a] = True
            for b in range(a + 1, len(pts)):
                if not used[b] and abs(pts[b].price - pts[a].price) <= tolerance:
                    cluster.append(pts[b])
                    used[b] = True
            if len(cluster) >= min_count:
                price = sum(s.price for s in cluster) / len(cluster)
                out.append(EqualLevel(kind=kind, price=round(price, 2),
                                      count=len(cluster),
                                      last_index=max(s.index for s in cluster)))
    return out
