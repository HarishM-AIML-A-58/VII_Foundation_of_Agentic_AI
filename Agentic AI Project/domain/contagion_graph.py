"""Supply Chain & Corporate Group Contagion Graph.

Pure domain network model of Indian conglomerates, cross-holdings, and customer-supplier
relationships. Simulates first-order and second-order price and revenue shock propagation
across equities during idiosyncratic or conglomerate-level distress events.

Zero external dependencies outside the standard library and trading_agent.domain.money.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Final

from trading_agent.domain.money import Rupees

__all__ = [
    "ContagionEdge",
    "ContagionGraph",
    "ContagionNode",
    "ContagionRelationship",
    "ContagionSimulationResult",
    "ShockPropagationItem",
    "create_default_indian_market_graph",
]

_DEFAULT_DAMPING_FACTOR: Final[float] = 0.70
_MIN_SHOCK_PROPAGATION_ABS_PCT: Final[float] = 0.05
_PERCENT_DIVISOR: Final[Decimal] = Decimal(100)
_FIRST_ORDER_HOP: Final[int] = 1
_SECOND_ORDER_HOP: Final[int] = 2


class ContagionRelationship(StrEnum):
    """Network edge typology."""

    CONGLOMERATE_AFFILIATE = "CONGLOMERATE_AFFILIATE"
    PRIMARY_CUSTOMER_TO_SUPPLIER = "PRIMARY_CUSTOMER_TO_SUPPLIER"
    KEY_SUPPLIER_TO_CUSTOMER = "KEY_SUPPLIER_TO_CUSTOMER"
    CROSS_HOLDING = "CROSS_HOLDING"


@dataclass(frozen=True, slots=True)
class ContagionNode:
    """Equity entity in network graph."""

    symbol: str
    company_name: str
    corporate_group: str | None = None
    sector: str = "GENERAL"


@dataclass(frozen=True, slots=True)
class ContagionEdge:
    """Directed transmission link between two equities."""

    source_symbol: str
    target_symbol: str
    relationship: ContagionRelationship
    coupling_weight: float  # (0.0, 1.0]


@dataclass(frozen=True, slots=True)
class ShockPropagationItem:
    """Estimated transmission impact on a single node."""

    symbol: str
    transmission_order: int
    relationship_path: str
    estimated_impact_pct: float
    stressed_rupee_impact: Rupees | None = None


@dataclass(frozen=True, slots=True)
class ContagionSimulationResult:
    """Aggregated network shock simulation."""

    source_symbol: str
    source_shock_pct: float
    impacted_nodes: list[ShockPropagationItem]
    max_contagion_symbol: str | None
    worst_impact_pct: float


@dataclass(slots=True)
class ContagionGraph:
    """Network model representing corporate and supply-chain linkages."""

    _nodes: dict[str, ContagionNode] = field(default_factory=dict)
    _adjacency: dict[str, list[ContagionEdge]] = field(default_factory=dict)

    def add_node(
        self,
        symbol: str,
        company_name: str,
        corporate_group: str | None = None,
        sector: str = "GENERAL",
    ) -> None:
        """Register a node in the graph."""
        sym = symbol.upper()
        self._nodes[sym] = ContagionNode(
            symbol=sym,
            company_name=company_name,
            corporate_group=corporate_group,
            sector=sector,
        )
        if sym not in self._adjacency:
            self._adjacency[sym] = []

    def add_edge(
        self,
        source: str,
        target: str,
        relationship: ContagionRelationship,
        coupling_weight: float,
    ) -> None:
        """Add a directed transmission edge from source to target."""
        src = source.upper()
        tgt = target.upper()
        if src not in self._adjacency:
            self._adjacency[src] = []

        edge = ContagionEdge(
            source_symbol=src,
            target_symbol=tgt,
            relationship=relationship,
            coupling_weight=max(0.0, min(1.0, coupling_weight)),
        )
        self._adjacency[src].append(edge)

    def get_node(self, symbol: str) -> ContagionNode | None:
        """Retrieve node info."""
        return self._nodes.get(symbol.upper())

    def get_outgoing_edges(self, symbol: str) -> list[ContagionEdge]:
        """List outgoing transmission edges for symbol."""
        return list(self._adjacency.get(symbol.upper(), []))

    def propagate_shock(
        self,
        source_symbol: str,
        shock_pct: float,
        damping: float = _DEFAULT_DAMPING_FACTOR,
        portfolio_positions: dict[str, Rupees] | None = None,
    ) -> ContagionSimulationResult:
        """Simulate shock contagion up to 2 hops across the graph."""
        src = source_symbol.upper()
        positions = portfolio_positions or {}
        impacts: dict[str, float] = {}
        paths: dict[str, tuple[int, str]] = {}

        # 1. First-order transmission
        first_order_edges = self.get_outgoing_edges(src)
        for edge in first_order_edges:
            tgt = edge.target_symbol
            impact = shock_pct * edge.coupling_weight * damping
            impacts[tgt] = impacts.get(tgt, 0.0) + impact
            paths[tgt] = (1, f"{src} -({edge.relationship.value})-> {tgt}")

        # 2. Second-order transmission
        for edge_1 in first_order_edges:
            mid = edge_1.target_symbol
            mid_impact = shock_pct * edge_1.coupling_weight * damping
            second_order_edges = self.get_outgoing_edges(mid)
            for edge_2 in second_order_edges:
                tgt_2 = edge_2.target_symbol
                if tgt_2 == src:
                    continue  # do not feedback into source
                impact_2 = mid_impact * edge_2.coupling_weight * damping
                impacts[tgt_2] = impacts.get(tgt_2, 0.0) + impact_2
                existing_order, _ = paths.get(tgt_2, (_SECOND_ORDER_HOP, ""))
                if existing_order >= _SECOND_ORDER_HOP:
                    paths[tgt_2] = (
                        _SECOND_ORDER_HOP,
                        f"{src} -> {mid} -({edge_2.relationship.value})-> {tgt_2}",
                    )

        # 3. Format results
        items: list[ShockPropagationItem] = []
        worst_symbol: str | None = None
        worst_impact: float = 0.0

        for symbol, total_impact in sorted(
            impacts.items(), key=lambda kv: kv[1] if shock_pct < 0 else -kv[1]
        ):
            if abs(total_impact) < _MIN_SHOCK_PROPAGATION_ABS_PCT:
                continue

            order, path_desc = paths[symbol]
            rupee_loss: Rupees | None = None
            if symbol in positions:
                pos_val = positions[symbol].value
                stressed_diff = pos_val * (Decimal(str(total_impact)) / _PERCENT_DIVISOR)
                rupee_loss = Rupees(stressed_diff)

            items.append(
                ShockPropagationItem(
                    symbol=symbol,
                    transmission_order=order,
                    relationship_path=path_desc,
                    estimated_impact_pct=round(total_impact, 2),
                    stressed_rupee_impact=rupee_loss,
                )
            )

            if abs(total_impact) > abs(worst_impact):
                worst_impact = total_impact
                worst_symbol = symbol

        return ContagionSimulationResult(
            source_symbol=src,
            source_shock_pct=shock_pct,
            impacted_nodes=items,
            max_contagion_symbol=worst_symbol,
            worst_impact_pct=round(worst_impact, 2),
        )


def create_default_indian_market_graph() -> ContagionGraph:
    """Pre-populate default Indian conglomerate and supply-chain network."""
    g = ContagionGraph()

    # --- TATA GROUP ---
    tata_group = "TATA"
    g.add_node("TCS", "Tata Consultancy Services", tata_group, "IT")
    g.add_node("TATAMOTORS", "Tata Motors Ltd", tata_group, "AUTO")
    g.add_node("TATASTEEL", "Tata Steel Ltd", tata_group, "METALS")
    g.add_node("TITAN", "Titan Company Ltd", tata_group, "CONSUMER")
    g.add_node("TATACONSUM", "Tata Consumer Products", tata_group, "FMCG")

    g.add_edge("TATAMOTORS", "TATASTEEL", ContagionRelationship.CONGLOMERATE_AFFILIATE, 0.30)
    g.add_edge("TATASTEEL", "TATAMOTORS", ContagionRelationship.CONGLOMERATE_AFFILIATE, 0.25)
    g.add_edge("TCS", "TATAMOTORS", ContagionRelationship.CONGLOMERATE_AFFILIATE, 0.15)

    # --- AUTO SUPPLY CHAIN ---
    g.add_node("MARUTI", "Maruti Suzuki India", None, "AUTO")
    g.add_node("MOTHERSON", "Samvardhana Motherson Int", None, "AUTO_ANCILLARY")
    g.add_node("SONACOMS", "Sona BLW Precision", None, "AUTO_ANCILLARY")
    g.add_node("BOSCHLTD", "Bosch Ltd", None, "AUTO_ANCILLARY")

    g.add_edge("MARUTI", "MOTHERSON", ContagionRelationship.PRIMARY_CUSTOMER_TO_SUPPLIER, 0.40)
    g.add_edge("MARUTI", "SONACOMS", ContagionRelationship.PRIMARY_CUSTOMER_TO_SUPPLIER, 0.25)
    g.add_edge("MARUTI", "BOSCHLTD", ContagionRelationship.PRIMARY_CUSTOMER_TO_SUPPLIER, 0.20)
    g.add_edge("TATAMOTORS", "MOTHERSON", ContagionRelationship.PRIMARY_CUSTOMER_TO_SUPPLIER, 0.30)

    # --- ADANI GROUP ---
    adani_group = "ADANI"
    g.add_node("ADANIENT", "Adani Enterprises Ltd", adani_group, "CONGLOMERATE")
    g.add_node("ADANIPORTS", "Adani Ports & SEZ", adani_group, "INFRASTRUCTURE")
    g.add_node("ADANIPOWER", "Adani Power Ltd", adani_group, "ENERGY")
    g.add_node("AMBUJACEM", "Ambuja Cements Ltd", adani_group, "CEMENT")

    g.add_edge("ADANIENT", "ADANIPORTS", ContagionRelationship.CONGLOMERATE_AFFILIATE, 0.60)
    g.add_edge("ADANIENT", "ADANIPOWER", ContagionRelationship.CONGLOMERATE_AFFILIATE, 0.55)
    g.add_edge("ADANIENT", "AMBUJACEM", ContagionRelationship.CONGLOMERATE_AFFILIATE, 0.45)
    g.add_edge("ADANIPORTS", "ADANIENT", ContagionRelationship.CONGLOMERATE_AFFILIATE, 0.50)

    # --- RELIANCE GROUP ---
    reliance_group = "RELIANCE"
    g.add_node("RELIANCE", "Reliance Industries Ltd", reliance_group, "OIL_TELECOM")
    g.add_node("JUSTDIAL", "Just Dial Ltd", reliance_group, "TECH")
    g.add_edge("RELIANCE", "JUSTDIAL", ContagionRelationship.CONGLOMERATE_AFFILIATE, 0.40)

    # --- CAPITAL GOODS & INFRASTRUCTURE ---
    g.add_node("LT", "Larsen & Toubro Ltd", None, "CAPITAL_GOODS")
    g.add_node("VOLTAS", "Voltas Ltd", tata_group, "CONSUMER_DURABLES")
    g.add_edge("LT", "VOLTAS", ContagionRelationship.PRIMARY_CUSTOMER_TO_SUPPLIER, 0.20)

    return g
