from src.depth_semantic_prompt import build_depth_semantic_prompt


def test_depth_semantic_prompt_locks_camera_mask_and_scale() -> None:
    prompt = build_depth_semantic_prompt(
        "a blue wheeled rubbish bin",
        structural_constraints=("Keep the two wheels parallel and attached to one axle.",),
    )
    for phrase in (
        "hard ControlNet-like spatial constraint",
        "camera viewpoint",
        "image-plane center",
        "apparent width and height",
        "foreground mask",
        "two wheels parallel",
    ):
        assert phrase in prompt


def test_depth_semantic_prompt_requires_object_description() -> None:
    try:
        build_depth_semantic_prompt("  ")
    except ValueError as error:
        assert "object_description" in str(error)
    else:
        raise AssertionError("empty object description must be rejected")
