"""Scene wrapper for the frozen single-object GenPC+ mainline.

Nothing in this package changes the object-level reconstruction code.  It only
creates object partials in a shared Pixal-MoGe camera frame, dispatches the
fixed Pixal/registration stages per instance, and composes their registered
textured meshes.
"""

from .contracts import SceneInstance, SceneManifest, load_scene_manifest
from .scene_moge import SceneMoGeObservation, infer_scene_moge, load_scene_moge

__all__ = [
    "SceneInstance",
    "SceneManifest",
    "SceneMoGeObservation",
    "infer_scene_moge",
    "load_scene_manifest",
    "load_scene_moge",
]
