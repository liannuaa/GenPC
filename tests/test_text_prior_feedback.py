import numpy as np

from src.text_prior_feedback import build_text_edit_feedback


def _box(size):
    grid = np.linspace(-.5, .5, 9)
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    return np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1) * np.asarray(size)


def test_feedback_requests_shared_axis_extension_for_visible_residual():
    prior = _box((1., .4, .3))
    partial = prior.copy()
    partial[:, 0] *= 1.18
    feedback = build_text_edit_feedback(partial, prior)
    assert feedback.eligible
    assert feedback.correction == "extend"
    assert feedback.principal_axis == 0
    assert "Preserve the input asset" in feedback.prompt


def test_feedback_does_not_encode_identity_or_sample_id():
    points = _box((1., .8, .6))
    feedback = build_text_edit_feedback(points, points)
    assert "01184" not in feedback.prompt
    assert "chair" not in feedback.prompt.lower()
    assert "object" in feedback.prompt.lower()


def test_text_variant_runner_uses_shared_project_model_root():
    source = ("scripts/run_agent_text_prior_variant.py")
    text = open(source, encoding="utf-8").read()
    assert 'PROJECT_ROOT / "models" / "TRELLIS-text-xlarge"' in text
    assert 'ROOT.parents[1]' in text


def test_agent_backend_runners_resolve_models_from_canonical_project_root():
    for path in (
        "scripts/run_nano3d_agent_prior.py",
        "scripts/run_spar3d_agent_prior.py",
        "scripts/run_arbor_agent_prior.py",
        "scripts/run_hunyuan_omni_agent_prior.py",
    ):
        text = open(path, encoding="utf-8").read()
        assert "PROJECT_ROOT = ROOT.parents[1]" in text
        assert "ROOT.parent.parent / \"models\"" not in text


def test_nano_runner_pins_available_attention_backend():
    text = open("scripts/run_nano3d_agent_prior.py", encoding="utf-8").read()
    assert 'os.environ.setdefault("ATTN_BACKEND", "xformers")' in text
    assert 'os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")' in text
