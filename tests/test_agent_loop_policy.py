from src.agent_loop_policy import (
    prefers_candidate,
    relative_geometric_gain,
    should_probe_generative_action,
)


def score(geometric: float, objective: float):
    return {"geometric": {"objective": geometric}, "objective": objective}


def test_relative_geometric_gain_is_positive_only_for_improvement():
    assert relative_geometric_gain(score(.2, .3), score(.1, .4)) == .5
    assert relative_geometric_gain(score(.2, .3), score(.22, .2)) < 0.


def test_prefer_observed_geometry_then_saved_view():
    assert prefers_candidate(score(.2, .3), score(.19, .5))
    assert prefers_candidate(score(.2, .3), score(.2, .29))
    assert not prefers_candidate(score(.2, .3), score(.21, .1))


def test_generative_probe_requires_an_accepted_reobserved_local_action():
    assert not should_probe_generative_action([])
    assert not should_probe_generative_action([
        {"route": "preserve_prior", "action_accepted": False},
        {"route": "accepted_global_qwen_nano3d", "action_accepted": True},
    ])
    assert should_probe_generative_action([
        {"route": "accepted_intrinsic_local_geometry", "action_accepted": True},
    ])
