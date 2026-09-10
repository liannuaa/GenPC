import numpy as np

from src.two_camera_moge_bridge import (
    affine_mask_iou,
    apply_image_affine,
    bbox_affine,
    conjugate_native_residual_to_partial,
    conjugate_partial_residual_to_native,
    infer_point_uv_flip_y,
    partial_uv_to_image_pixels,
    transferred_partial_to_moge_matches,
)


def test_bbox_affine_transfers_foreground_support():
    source = np.zeros((40, 50), dtype=np.uint8)
    source[5:25, 10:30] = 255
    target = np.zeros((80, 100), dtype=np.uint8)
    target[10:50, 20:60] = 255
    affine = bbox_affine(source, target)
    # Nearest-neighbour rasterization of a non-integer endpoint-preserving
    # scale can differ by a one-pixel border; the support still transfers.
    assert affine_mask_iou(source, target, affine) > .90
    assert np.allclose(apply_image_affine(np.array([[10., 5.]]), affine), [[20., 10.]])


def test_partial_uv_flip_and_transferred_moge_matches():
    # The first UV is bottom-left canonical; with flip_y it becomes image y=9.
    uv = np.array(((0., 0.), (1., 1.), (.5, .5)))
    pixels, valid = partial_uv_to_image_pixels(uv, (10, 10), flip_y=True)
    assert valid.all()
    assert np.allclose(pixels, ((0., 9.), (9., 0.), (4.5, 4.5)))
    moge = pixels + np.array((3., -2.))
    affine = np.eye(3)
    affine[:2, 2] = (3., -2.)
    matches, info = transferred_partial_to_moge_matches(
        uv, (10, 10), affine, moge, max_pixel_distance=.01, flip_y=True,
    )
    assert info["matched_partial_pixels"] == 3
    assert np.array_equal(matches[:, 0], np.array((0., 1., 2.)))
    assert np.array_equal(matches[:, 1], np.array((0., 1., 2.)))


def test_uv_origin_is_inferred_from_observed_foreground_support():
    mask = np.zeros((101, 101), dtype=np.uint8)
    mask[70:91, 20:81] = 255
    top_left_uv = np.array(((.25, .75), (.50, .80), (.75, .85)))
    flip, scores = infer_point_uv_flip_y(top_left_uv, mask, dilation_pixels=0)
    assert not flip
    assert scores["top_left"] == 1.0
    assert scores["bottom_left"] == 0.0

    bottom_left_uv = top_left_uv.copy()
    bottom_left_uv[:, 1] = 1.0 - bottom_left_uv[:, 1]
    flip, scores = infer_point_uv_flip_y(bottom_left_uv, mask, dilation_pixels=0)
    assert flip
    assert scores["bottom_left"] == 1.0
    assert scores["top_left"] == 0.0


def test_native_residual_conjugation_is_equivalent_after_camera_bridge():
    bridge = np.eye(4)
    bridge[:3, :3] *= 2.
    bridge[:3, 3] = (1., -2., .5)
    partial_residual = np.eye(4)
    partial_residual[:3, 3] = (.1, .2, -.1)
    native_residual = conjugate_partial_residual_to_native(partial_residual, bridge)
    point = np.array([[.4, -.3, 1.2]])
    lhs = (np.c_[point, np.ones(1)] @ (partial_residual @ bridge).T)[:, :3]
    rhs = (np.c_[point, np.ones(1)] @ (bridge @ native_residual).T)[:, :3]
    assert np.allclose(lhs, rhs)
    assert np.allclose(conjugate_native_residual_to_partial(native_residual, bridge), partial_residual)
