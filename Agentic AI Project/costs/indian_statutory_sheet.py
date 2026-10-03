"""Friction audit for the institutional desk.

Equity delivery and intraday go through :class:`ChargeCalculator` -- the same
dated schedules the backtest and the order path use. F&O is not on those
schedules, so it is computed here from the published NSE/Zerodha rates, in
``Decimal``, and labelled as such.

The public report still serialises to ``float`` because the dashboard JSON
schema is float. Construction never does float arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

from trading_agent.costs.calculator import ChargeCalculator
from trading_agent.domain.enums import ProductType, Side

__all__ = [
    "IndianStatutoryAuditEngine",
    "SegmentType",
    "StatutoryBreakdown",
    "StatutoryFrictionReport",
]


def _charges_dir() -> Path:
    """Resolve the dated YAML directory in a checkout *or* a container.

    ``__file__`` is ``src/trading_agent/costs/...`` in a checkout and
    ``.venv/lib/.../site-packages/trading_agent/costs/...`` in the image, so
    walking parents of this file is not portable. Prefer the working
    directory (``/app/config/charges`` in Docker, the repo root in a checkout).
    """
    candidates = (
        Path("config/charges"),
        Path(__file__).resolve().parents[3] / "config" / "charges",
    )
    for path in candidates:
        if path.is_dir() and any(path.glob("*.yaml")):
            return path
    raise FileNotFoundError(
        "no charge schedules found; expected config/charges/*.yaml relative to cwd"
    )


#: F&O rates from zerodha.com/charges/ (retrieved 2026-09-11) and NSE/FA/73061.
_FUT_STT_SELL = Decimal("0.0005")  # 0.05% sell
_OPT_STT_SELL = Decimal("0.0015")  # 0.15% of premium, sell
_FUT_EXCHANGE = Decimal("0.0000183")  # 0.00183% (txn + IPFT, true-to-label)
_OPT_EXCHANGE = Decimal("0.0003553")  # 0.03553% of premium
_SEBI = Decimal("0.000001")
_FUT_STAMP_BUY = Decimal("0.00002")
_OPT_STAMP_BUY = Decimal("0.00003")
_GST_RATE = Decimal("0.18")
_STCG_RATE = Decimal("0.20")
_TWO_PLACES = Decimal("0.01")


class SegmentType(StrEnum):
    EQUITY_DELIVERY = "equity_delivery"
    EQUITY_INTRADAY = "equity_intraday"
    FUTURES = "futures"
    OPTIONS = "options"


@dataclass(frozen=True, slots=True)
class StatutoryBreakdown:
    brokerage: float
    stt: float
    exchange_charges: float
    sebi_charges: float
    stamp_duty: float
    gst: float
    total_statutory_charges: float
    stcg_tax: float
    total_friction: float


@dataclass(frozen=True, slots=True)
class StatutoryFrictionReport:
    symbol: str
    segment: SegmentType
    buy_turnover: float
    sell_turnover: float
    total_turnover: float
    gross_pnl: float
    net_pnl_after_charges: float
    net_pnl_after_tax: float
    statutory_drag_bps: float
    tax_drag_bps: float
    total_friction_bps: float
    break_even_alpha_pct: float
    breakdown: StatutoryBreakdown


def _money(value: Decimal) -> float:
    return float(value.quantize(_TWO_PLACES))


def _bps(part: Decimal, capital: Decimal) -> float:
    if capital <= 0:
        return 0.0
    return float(((part / capital) * Decimal("10000")).quantize(_TWO_PLACES))


class IndianStatutoryAuditEngine:
    """Round-trip friction: equity via the charge calculator, F&O via Decimal."""

    @staticmethod
    def audit(
        *,
        symbol: str,
        segment: SegmentType | str = SegmentType.EQUITY_DELIVERY,
        buy_price: float,
        sell_price: float,
        quantity: int,
        flat_brokerage_per_order: float = 20.0,
        is_profit_applicable_for_stcg: bool = True,
        on: date | None = None,
        calculator: ChargeCalculator | None = None,
    ) -> StatutoryFrictionReport:
        seg = SegmentType(segment)
        qty = abs(quantity)
        trade_date = on or date.today()  # noqa: DTZ011 -- audit of a notional, not a timestamp
        buy = Decimal(str(buy_price))
        sell = Decimal(str(sell_price))
        buy_val = buy * qty
        sell_val = sell * qty
        total_val = buy_val + sell_val
        gross_pnl = sell_val - buy_val

        if seg in (SegmentType.EQUITY_DELIVERY, SegmentType.EQUITY_INTRADAY):
            product = ProductType.CNC if seg is SegmentType.EQUITY_DELIVERY else ProductType.MIS
            calc = calculator or ChargeCalculator.from_directory(_charges_dir(), broker="zerodha")
            legs = calc.round_trip(
                entry_price=buy,
                exit_price=sell,
                quantity=qty,
                side=Side.BUY,
                product=product,
                entry_date=trade_date,
            )
            brokerage = legs.brokerage.value
            stt = legs.securities_transaction_tax.value
            exchange_charges = (
                legs.exchange_transaction_charge.value + legs.investor_protection_fund.value
            )
            sebi_charges = legs.sebi_turnover_fee.value
            stamp_duty = legs.stamp_duty.value
            gst = legs.goods_and_services_tax.value
            # DP is a broker/depository debit, not a statutory levy, but it is
            # friction the round trip actually pays. Fold it into brokerage so
            # the desk total matches the calculator's total.
            brokerage += legs.depository_participant_charge.value
        else:
            brokerage = Decimal(str(flat_brokerage_per_order))
            if buy_val > 0 and sell_val > 0:
                brokerage *= 2
            if seg is SegmentType.FUTURES:
                stt = sell_val * _FUT_STT_SELL
                exchange_charges = total_val * _FUT_EXCHANGE
                stamp_duty = buy_val * _FUT_STAMP_BUY
            else:
                stt = sell_val * _OPT_STT_SELL
                exchange_charges = total_val * _OPT_EXCHANGE
                stamp_duty = buy_val * _OPT_STAMP_BUY
            sebi_charges = total_val * _SEBI
            gst = (brokerage + exchange_charges + sebi_charges) * _GST_RATE

        total_statutory = brokerage + stt + exchange_charges + sebi_charges + stamp_duty + gst
        net_after_charges = gross_pnl - total_statutory

        stcg = Decimal("0")
        if (
            seg is SegmentType.EQUITY_DELIVERY
            and is_profit_applicable_for_stcg
            and net_after_charges > 0
        ):
            stcg = (net_after_charges * _STCG_RATE).quantize(_TWO_PLACES)

        net_after_tax = net_after_charges - stcg
        total_friction = total_statutory + stcg
        capital = buy_val if buy_val > 0 else total_val

        breakdown = StatutoryBreakdown(
            brokerage=_money(brokerage),
            stt=_money(stt),
            exchange_charges=_money(exchange_charges),
            sebi_charges=_money(sebi_charges),
            stamp_duty=_money(stamp_duty),
            gst=_money(gst),
            total_statutory_charges=_money(total_statutory),
            stcg_tax=_money(stcg),
            total_friction=_money(total_friction),
        )
        return StatutoryFrictionReport(
            symbol=symbol,
            segment=seg,
            buy_turnover=_money(buy_val),
            sell_turnover=_money(sell_val),
            total_turnover=_money(total_val),
            gross_pnl=_money(gross_pnl),
            net_pnl_after_charges=_money(net_after_charges),
            net_pnl_after_tax=_money(net_after_tax),
            statutory_drag_bps=_bps(total_statutory, capital),
            tax_drag_bps=_bps(stcg, capital),
            total_friction_bps=_bps(total_friction, capital),
            break_even_alpha_pct=float(
                ((total_statutory / capital) * Decimal("100")).quantize(Decimal("0.0001"))
                if capital > 0
                else Decimal("0")
            ),
            breakdown=breakdown,
        )
