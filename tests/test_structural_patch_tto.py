import numpy as np

from src.structural_patch_tto import estimate_normals_planarity


def test_planar_patch_has_consistent_normals_and_high_planarity():
    xy = np.stack(np.meshgrid(np.linspace(-1., 1., 12), np.linspace(-1., 1., 12)), axis=-1).reshape(-1, 2)
    points = np.c_[xy, np.zeros(len(xy))]
    normals, planarity = estimate_normals_planarity(points, neighbours=12)
    assert np.median(planarity) > .4
    assert np.median(np.abs(normals[:, 2])) > .95
