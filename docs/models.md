# Model assets

Model weights and upstream source trees are not distributed with GenPC++.

| Component | Purpose | Default path |
| --- | --- | --- |
| Direct GPT image generation | mask-locked depth-conditioned semantic completion | external; save image and exact prompt as run artifacts |
| Qwen-Image-Edit-2511 | optional local semantic-completion baseline | `models/Qwen-Image-Edit-2511` |
| Nunchaku Qwen INT4 | optional Qwen execution on a 24 GB GPU | `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors` |
| Pixal3D source | semantic image to complete textured prior | `models/Pixal3D` or `$PIXAL3D_SOURCE` |
| Pixal3D weights | Pixal inference | `models/Pixal3D-weights` |
| DINOv3 ViT-L/16 | Pixal image conditioning | `models/dinov3-vitl16-pretrain-lvd1689m` |
| MoGe-2 ViT-L | native camera observation and registration bridge | `models/moge-2-vitl/model.pt` |
| RMBG-2.0 | foreground masks | `models/RMBG-2.0` |

Use the official [Pixal3D](https://github.com/TencentARC/Pixal3D),
[MoGe](https://github.com/microsoft/MoGe), and
[RMBG](https://modelscope.cn/models/AI-ModelScope/RMBG-2.0) instructions and
their respective licenses.

Pixal3D depends on its upstream sparse 3D runtime. Keep that checkout isolated
from this repository and expose it through `PIXAL3D_SOURCE`. Qwen and its
version pins are needed only when reproducing the optional local baseline.

Direct-GPT outputs, exact prompts, source depth images, and mask-audit manifests
are stored together. No API or repository token is stored or read by GenPC++.
