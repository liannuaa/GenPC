"""Optional GPT vision verification for a bounded geometric edit proposal.

The VLM never generates 3-D instructions freely.  It sees a saved-view board
and may only approve or veto the numeric geometry module's one local edit.
This keeps the external semantic judgement reproducible and prevents visual
hallucinations from changing global pose, identity, or hidden geometry.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.text_prior_feedback import TextEditFeedback


VLM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "approve_edit": {"type": "boolean"},
        "visible_issue": {"type": "string", "enum": ["extent", "surface", "none"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reason": {"type": "string", "maxLength": 220},
    },
    "required": ["approve_edit", "visible_issue", "confidence", "reason"],
}


def _data_url(image_path: Path) -> str:
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def request_vlm_verdict(
    board_path: Path,
    feedback: TextEditFeedback,
    *,
    model: str,
    api_key: str | None = None,
    timeout_seconds: float = 90.0,
) -> dict:
    """Ask GPT to verify one already-bounded observed-surface edit.

    Raises only on missing credentials or a failed API request.  Callers are
    expected to fall back to the pure geometry policy; failure never grants an
    edit action.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for optional GPT vision feedback")
    instruction = (
        "You are a cautious geometric verifier. The image board has: left=partial "
        "scan silhouette, middle=current complete-prior silhouette, right=their "
        "overlay in the same saved camera view. Red is prior; gray is partial. "
        "Do not infer object category or hidden geometry. Decide only whether the "
        "following already-bounded local correction is visually supported: "
        f"{feedback.correction} along the {('long', 'middle', 'short')[feedback.principal_axis]} "
        f"principal axis. Approve only when the visible mismatch supports it."
    )
    payload = {
        "model": model,
        "store": False,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": instruction},
            {"type": "input_image", "image_url": _data_url(board_path), "detail": "high"},
        ]}],
        "text": {"format": {"type": "json_schema", "name": "geometric_edit_verdict",
                               "strict": True, "schema": VLM_SCHEMA}},
    }
    request = Request(
        "https://api.openai.com/v1/responses", data=json.dumps(payload).encode("utf-8"),
        method="POST", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=float(timeout_seconds)) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GPT vision request failed ({exc.code}): {body[:500]}") from exc
    output_text = result.get("output_text")
    if not output_text:
        raise RuntimeError("GPT vision response contained no structured output")
    verdict = json.loads(output_text)
    if set(verdict) != {"approve_edit", "visible_issue", "confidence", "reason"}:
        raise RuntimeError("GPT vision response did not satisfy the expected schema")
    verdict["model"] = model
    verdict["response_id"] = result.get("id")
    return verdict


def approved_by_vlm(verdict: dict | None) -> bool:
    """A missing/failed VLM is conservative: it never expands the action set."""
    return bool(verdict and verdict.get("approve_edit") and verdict.get("visible_issue") != "none")
