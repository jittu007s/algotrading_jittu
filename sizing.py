"""Position sizing for option buys.

Rule (config-driven): deploy FUND_UTILISATION_PCT of available capital, but
never fewer than MIN_LOTS lots. Options trade in lots, so the cost of one lot
is premium x lot_size and the deployable number of lots is the floor of
budget / cost_per_lot.

The MIN_LOTS floor deliberately overrides the utilisation cap (2 lots may cost
more than 50% of a small account). It does NOT override affordability: if the
minimum position costs more than the whole account the trade is rejected
rather than sent to be bounced by the broker's margin check.
"""

import logging
import math

logger = logging.getLogger(__name__)


class Sizing:
    __slots__ = ("lots", "quantity", "cost", "reason", "ok")

    def __init__(self, lots, quantity, cost, reason, ok):
        self.lots = lots
        self.quantity = quantity
        self.cost = cost
        self.reason = reason
        self.ok = ok

    def __repr__(self):
        return (f"Sizing(lots={self.lots}, qty={self.quantity}, "
                f"cost={self.cost:.2f}, ok={self.ok}, reason={self.reason!r})")


def compute_size(capital, premium, lot_size, utilisation_pct=50.0, min_lots=2):
    """Size an option buy.

    capital   - cash available (None/<=0 means unknown -> fall back to min_lots)
    premium   - option price per unit
    lot_size  - units per lot (from the scrip master)
    Returns a Sizing; check `.ok` before ordering.
    """
    if not premium or premium <= 0 or not lot_size or lot_size <= 0:
        return Sizing(0, 0, 0.0, "invalid premium or lot size", False)

    cost_per_lot = premium * lot_size

    if capital is None or capital <= 0:
        cost = min_lots * cost_per_lot
        return Sizing(min_lots, min_lots * lot_size, cost,
                      "capital unknown - using minimum lots", True)

    budget = capital * utilisation_pct / 100.0
    lots = int(math.floor(budget / cost_per_lot))

    if lots >= min_lots:
        reason = f"{utilisation_pct:.0f}% of {capital:.0f}"
    else:
        lots = min_lots
        reason = (f"{utilisation_pct:.0f}% of {capital:.0f} affords "
                  f"{int(math.floor(budget / cost_per_lot))} lot(s) - raised to "
                  f"minimum {min_lots}")

    cost = lots * cost_per_lot
    if cost > capital:
        return Sizing(lots, lots * lot_size, cost,
                      f"minimum {min_lots} lot(s) cost {cost:.0f} > available "
                      f"capital {capital:.0f}", False)
    return Sizing(lots, lots * lot_size, cost, reason, True)
