from pathlib import Path


def test_hunyuan_runner_documents_text_and_coordinate_contract():
    source = (Path(__file__).resolve().parents[1] / "scripts" /
              "run_hunyuan21_agent_prior.py").read_text(encoding="utf-8")
    assert "caption_consumed_by_shape_model\": False" in source
    assert "upstream_semantic_image_only" in source
    assert "common_global_or_PCA_proper_Sim3_registration" in source
