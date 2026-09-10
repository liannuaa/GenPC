import numpy as np
from scipy.sparse.csgraph import connected_components

from src.mesh_attached_gaussian_deformation import (
    _arap_surface_deformation,
    fit_carrier_affine,
    mesh_surface_graph,
)


def test_affine_carrier_transfer_is_slot_preserving():
    source = np.array([
        [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        [1., 1., 1.], [.2, .3, .4],
    ])
    linear = np.diag([1.25, .9, 1.1])
    target = source @ linear.T + np.array([.1, -.2, .3])
    transform, info = fit_carrier_affine(source, target)
    assert info["carrier_affine_max_residual"] < 1e-10
    assert np.allclose(transform[:3, :3], linear)


def test_mesh_graph_welds_coincident_uv_seam_vertices():
    # Two triangles meet along an edge but use duplicated positions, as can
    # happen across a texture seam in a generated GLB.
    vertices = np.array([
        [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
        [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
    ])
    graph = mesh_surface_graph(vertices, np.array([[0, 1, 2], [3, 4, 5]]))
    components, _ = connected_components(graph, directed=False)
    assert components == 1


def test_arap_keeps_fixed_vertices_and_moves_control_continuously():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
    graph = mesh_surface_graph(vertices, np.array([[0, 1, 2], [1, 3, 2]]))
    displacement, info = _arap_surface_deformation(
        vertices, graph, movable=np.array([1, 2, 3]), fixed=np.array([0]),
        control_vertices=np.array([3]), control_targets=np.array([[1.3, 1.2, 0.]]),
        iterations=2,
    )
    assert np.allclose(displacement[0], 0.)
    assert info["arap_control_improvement"] > .9
