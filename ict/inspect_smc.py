from datetime import datetime, timedelta
from ict.datafeed import PollingFeed
from ict.smc import find_fvgs, find_order_blocks, find_equal_levels

feed = PollingFeed()
now = datetime.now()
candles = feed.fetch("THREE_MINUTE", now - timedelta(days=2), now)
today = [c for c in candles if c.timestamp.date() == now.date()]

print(f"{len(today)} candles today")
fvgs = find_fvgs(today, min_size=5.0)
print(f"\nTop 5 fresh FVGs (largest first):")
for g in fvgs[:5]:
    print(f"  {g.timestamp:%H:%M}  {g.low:.1f}-{g.high:.1f}  ({g.high-g.low:.1f} pts)")
obs = find_order_blocks(today, displacement_mult=1.5)
print(f"\nUntested order blocks:")
for b in obs[:5]:
    print(f"  {b.timestamp:%H:%M}  {b.kind.value}  {b.low:.1f}-{b.high:.1f}")
eqs = find_equal_levels(today, tolerance=5.0)
print(f"\nEqual highs/lows (liquidity pools):")
for e in eqs:
    print(f"  {e.kind.value} @ {e.price:.1f}  (x{e.count})")
