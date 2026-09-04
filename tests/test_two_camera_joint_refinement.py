import numpy as np

from src.two_camera_joint_refinement import compose_two_camera_transform, residual_proposals


def test_two_camera_composition_obeys_edge_order():
    q_to_m = np.eye(4); q_to_m[0, 3] = 1.
    m_to_p = np.eye(4); m_to_p[1, 3] = 2.
    delta_q = np.eye(4); delta_q[2, 3] = 3.
    delta_m = np.eye(4); delta_m[0, 3] = 4.
    total = compose_two_camera_transform(q_to_m, m_to_p, delta_q, delta_m)
    assert np.allclose(total, delta_m @ m_to_p @ delta_q @ q_to_m)


def test_residual_proposals_include_identity_and_center_preserving_rotation():
    centre = np.array((2., -1., .5))
    proposals = dict(residual_proposals(scale_delta=.01, rotation_deg=.5, translation=.1, centre=centre))
    assert "identity" in proposals and "rot_z_+1" in proposals
    rotated = proposals["rot_z_+1"]
    moved_centre = (np.r_[centre, 1.] @ rotated.T)[:3]
    assert np.allclose(moved_centre, centre)
