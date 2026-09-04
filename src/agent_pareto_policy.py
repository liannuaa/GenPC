"""Small Pareto action policy for conflicting 2-D/3-D registration evidence."""

from __future__ import annotations

from typing import Mapping


def registration_objectives(visible: Mapping, *, deformation_ratio: float) -> tuple[float, float, float]:
    """Return minimization objectives: visible, 3-D surface, deformation size."""
    projection = visible["projection"]
    visible_loss = ((1. - float(projection["iou"]))
                    + .5 * (1. - float(projection["coverage"]))
                    + .5 * float(projection["leakage"]))
    return visible_loss, float(visible["geometric"]["objective"]), float(deformation_ratio)


def dominates(left: tuple[float, ...], right: tuple[float, ...], *, epsilon: float = 1e-9) -> bool:
    """True if left is no worse in every objective and strictly better in one."""
    return all(a <= b + epsilon for a, b in zip(left, right)) and any(a < b - epsilon for a, b in zip(left, right))


def pareto_archive(candidates: list[Mapping]) -> list[Mapping]:
    """Keep nondominated candidates; input entries must contain `objectives`."""
    result = []
    for candidate in candidates:
        value = tuple(map(float, candidate["objectives"]))
        if any(dominates(tuple(map(float, other["objectives"])), value) for other in candidates if other is not candidate):
            continue
        result.append(candidate)
    return result


def select_pareto_knee(candidates: list[Mapping]) -> Mapping:
    """Choose the archive point with smallest normalized worst-case regret."""
    archive = pareto_archive(candidates)
    if not archive:
        raise ValueError("at least one safe candidate is required")
    width = len(archive[0]["objectives"])
    low = [min(float(item["objectives"][axis]) for item in archive) for axis in range(width)]
    high = [max(float(item["objectives"][axis]) for item in archive) for axis in range(width)]
    def regret(item):
        return max((float(item["objectives"][axis]) - low[axis]) / max(high[axis] - low[axis], 1e-9)
                   for axis in range(width))
    selected = min(archive, key=lambda item: (regret(item), sum(map(float, item["objectives"]))))
    return {"selected": selected, "archive": archive, "selection": "normalized_minimax_pareto_knee",
            "regret": float(regret(selected))}
