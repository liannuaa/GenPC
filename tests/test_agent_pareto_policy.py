from src.agent_pareto_policy import dominates, pareto_archive, select_pareto_knee


def test_pareto_archive_keeps_tradeoffs_and_drops_dominated_action():
    candidates = [
        {"name": "small", "objectives": (0.1, 0.8, 0.0)},
        {"name": "balanced", "objectives": (0.2, 0.2, 0.2)},
        {"name": "large", "objectives": (0.4, 0.1, 0.8)},
        {"name": "bad", "objectives": (0.5, 0.9, 0.9)},
    ]
    assert dominates(candidates[1]["objectives"], candidates[3]["objectives"])
    archive = pareto_archive(candidates)
    assert {item["name"] for item in archive} == {"small", "balanced", "large"}
    result = select_pareto_knee(candidates)
    assert result["selected"]["name"] == "balanced"
