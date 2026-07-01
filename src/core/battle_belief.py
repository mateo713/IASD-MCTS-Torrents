from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class OpponentSetHypothesis:
    set_data: dict[str, Any]
    weight: float = 1.0
    violations: int = 0
    source: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def normalized_weight(self) -> float:
        return max(0.0, float(self.weight))


@dataclass(slots=True)
class BeliefPool:
    hypotheses: list[OpponentSetHypothesis] = field(default_factory=list)

    def add(
        self,
        set_data: dict[str, Any],
        weight: float = 1.0,
        violations: int = 0,
        source: str | None = None,
        notes: dict[str, Any] | None = None,
    ) -> None:
        self.hypotheses.append(
            OpponentSetHypothesis(
                set_data=set_data,
                weight=weight,
                violations=violations,
                source=source,
                notes=notes or {},
            )
        )

    def total_weight(self) -> float:
        return sum(hypothesis.normalized_weight() for hypothesis in self.hypotheses)

    def normalized_weights(self) -> list[tuple[dict[str, Any], float]]:
        total = self.total_weight()
        if total <= 0:
            return [(hypothesis.set_data, 1.0) for hypothesis in self.hypotheses]
        return [(hypothesis.set_data, hypothesis.normalized_weight() / total) for hypothesis in self.hypotheses]

    def sample(self, rng) -> dict[str, Any] | None:
        if not self.hypotheses:
            return None
        total = self.total_weight()
        if total <= 0:
            return self.hypotheses[0].set_data

        roll = rng.random() * total
        acc = 0.0
        for hypothesis in self.hypotheses:
            acc += hypothesis.normalized_weight()
            if roll <= acc:
                return hypothesis.set_data
        return self.hypotheses[-1].set_data

    def best_by_violations(self) -> list[dict[str, Any]]:
        if not self.hypotheses:
            return []
        best_score = min(hypothesis.violations for hypothesis in self.hypotheses)
        return [hypothesis.set_data for hypothesis in self.hypotheses if hypothesis.violations == best_score]
