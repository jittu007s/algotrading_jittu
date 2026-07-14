"""Typed, YAML-driven configuration for the ICT system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass(frozen=True)
class RiskCfg:
    risk_per_trade_pct: float
    max_daily_loss_pct: float
    equity_drawdown_stop_pct: float


@dataclass(frozen=True)
class StructureCfg:
    swing_k_bias: int
    swing_k_setup: int
    sweep_min_pierce: float
    sweep_needs_rejection: bool
    displacement_body_mult: float
    avg_body_period: int
    setup_validity_candles: int
    entry_point: str


@dataclass(frozen=True)
class SessionCfg:
    skip_first_minutes: int
    no_entry_after: dtime
    square_off: dtime


@dataclass(frozen=True)
class OptionsCfg:
    strike: str
    delta_assumed: float
    avoid_expiry_afternoon: bool


@dataclass(frozen=True)
class IctConfig:
    capital: float
    mode: str
    risk: RiskCfg
    structure: StructureCfg
    session: SessionCfg
    options: OptionsCfg
    bias_daily_tf: str
    bias_intraday_tf: str
    setup_tf: str
    telegram_enabled: bool
    journal_db: str


def _parse_time(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))


def load(path: Path = CONFIG_PATH) -> IctConfig:
    raw = yaml.safe_load(path.read_text())
    return IctConfig(
        capital=float(raw["capital"]),
        mode=str(raw.get("mode", "paper")),
        risk=RiskCfg(**{k: float(v) for k, v in raw["risk"].items()}),
        structure=StructureCfg(
            swing_k_bias=int(raw["structure"]["swing_k_bias"]),
            swing_k_setup=int(raw["structure"]["swing_k_setup"]),
            sweep_min_pierce=float(raw["structure"]["sweep_min_pierce"]),
            sweep_needs_rejection=bool(raw["structure"]["sweep_needs_rejection"]),
            displacement_body_mult=float(raw["structure"]["displacement_body_mult"]),
            avg_body_period=int(raw["structure"]["avg_body_period"]),
            setup_validity_candles=int(raw["structure"]["setup_validity_candles"]),
            entry_point=str(raw["structure"].get("entry_point", "midpoint")),
        ),
        session=SessionCfg(
            skip_first_minutes=int(raw["session"]["skip_first_minutes"]),
            no_entry_after=_parse_time(raw["session"]["no_entry_after"]),
            square_off=_parse_time(raw["session"]["square_off"]),
        ),
        options=OptionsCfg(
            strike=str(raw["options"]["strike"]),
            delta_assumed=float(raw["options"]["delta_assumed"]),
            avoid_expiry_afternoon=bool(raw["options"]["avoid_expiry_afternoon"]),
        ),
        bias_daily_tf=raw["timeframes"]["bias_daily"],
        bias_intraday_tf=raw["timeframes"]["bias_intraday"],
        setup_tf=raw["timeframes"]["setup"],
        telegram_enabled=bool(raw["telegram"]["enabled"]),
        journal_db=str(raw["journal"]["db_path"]),
    )
