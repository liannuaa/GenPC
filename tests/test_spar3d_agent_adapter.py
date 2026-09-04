import numpy as np

from src.prior_backend_contract import validate_proper_sim3
from src.spar3d_agent_adapter import prepare_spar3d_condition


def test_spar3d_conditioning_is_centered_isotropic_and_reversible():
    points = np.array([[2., 3., 5.], [6., 5., 7.], [4., 9., 6.]])
    conditioned, record = prepare_spar3d_condition(points)
    assert conditioned.shape == (3, 6)
    assert np.allclose(conditioned[:, 3:], .5)
    assert np.allclose(conditioned[:, :3].min(axis=0) + conditioned[:, :3].max(axis=0), 0.)
    assert np.isclose(np.ptp(conditioned[:, :3], axis=0).max(), 1.)
    recovered = conditioned[:, :3] @ record.condition_to_raw[:3, :3].T + record.condition_to_raw[:3, 3]
    assert np.allclose(recovered, points)
    assert validate_proper_sim3(record.mesh_output_rotation) == 1.


def test_spar3d_conditioning_bounds_attention_with_deterministic_sampling():
    points = np.stack([np.arange(10.), np.arange(10.), np.arange(10.)], axis=1)
    first, record = prepare_spar3d_condition(points, max_points=4, seed=7)
    second, _ = prepare_spar3d_condition(points, max_points=4, seed=7)
    assert first.shape == (4, 6)
    assert np.array_equal(first, second)
    assert record.source_point_count == 10
    assert record.condition_point_count == 4
