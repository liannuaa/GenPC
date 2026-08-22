# 当前 GPT ImageGen + Pixal3D + v15/v18 完整流程

更新时间：2026-08-22

状态：Redwood 十样本零样本候选流程已经完整运行；GPT 语义图、Pixal3D
完整模型和 v15 全局配准已冻结。v18-gatefix 融合完成十样本并在冻结后达到
平均 `CD-L1/EMD x1e2 = 1.675656/2.726899`，低于 GenPC 论文平均
`1.74/2.88`。融合仍待逐样本视觉确认，尚不能称为最终接受版本。

## 1. 方法目标与整体思路

输入是一个真实扫描得到的 partial point cloud，输出是在原 partial 坐标系下
对齐的完整物体。当前路线不再使用旧的外部 MoGe→Hunyuan3D 组合，而是：

```text
raw partial point cloud
  │
  ├─ GenPC 保存的相机与深度投影
  │      ├─ depth.png
  │      ├─ camera.pth
  │      └─ point_uv.npy
  │
  ├─ GPT ImageGen：partial depth → 完整语义/RGB 图
  │      └─ gpt_image.png
  │
  ├─ Pixal3D：完整语义图 → 完整 textured mesh + 100k point cloud
  │      ├─ pixal3d.glb
  │      └─ pixal3d_sampled_100k.ply
  │
  ├─ v15 visibility-aware proper-Sim3 TTT
  │      └─ 位姿、统一尺度和位置对齐后的完整 Pixal 模型
  │
  ├─ v18-gatefix partial-core local fusion
  │      ├─ 可靠时：局部拉动 generated surface，再加入 exact partial
  │      └─ 不可靠时：完整 Pixal + exact partial 的安全并集回退
  │
  ├─ 冻结 prediction
  │
  └─ 冻结后读取 GT，计算 CD-L1 / EMD
```

它仍然可以看作从 GenPC 演化而来：保留 GenPC 的 partial 投影、保存相机、
2D+3D test-time registration 和原始观测约束；把语义补全替换成 GPT
ImageGen，把完整 3D prior 替换成 Pixal3D，再加入 visibility-aware Sim3
和 completeness-preserving local fusion。

## 2. 零样本与泛化约束

整个推理阶段遵守以下约束：

- 不在 Redwood test geometry 上训练或微调；
- 十个样本使用同一套配准、融合参数和安全门；
- 不根据 sample ID 或类别选择几何参数；
- 类别文本只用于 GPT 图像补全，不参与 3D transform 或融合路由；
- GT、CD 和 EMD 不参与图像生成、Pixal3D、配准、形变、融合或回退；
- GPT 图像和原始 Pixal GLB/PLY 一旦接受便冻结，不为难例重新生成；
- 配准只允许 proper rotation、一个 isotropic scale 和 translation；
- 融合不删除 Pixal 点，不使用局部缩放、剪切或各向异性形变；
- 低置信度时回退，不用 GT 或 case-specific 参数强行通过。

十样本固定顺序为：

```text
01184 05117 05452 06127 06145 06188 06830 07136 07306 09639
```

## 3. Stage 0：partial、深度图与保存相机

### 3.1 输入与文件

- raw partial：`data/<sample>.ply`；
- 深度投影：`gpt_version/<sample>/depth.png`，`512×512`；
- 相机目录：
  `workspace/redwood_onestage_rawdepth_512_stage2_20260714/<sample>`；
- 相机与像素映射：`camera.pth`、`point_uv.npy`。

深度图负责提供原始物体在图像中的姿态、大小、占画面比例和局部结构。
`SavedCameraProjector` 负责把 partial 和 complete point cloud 投影到同一个
保存相机中，并保留 GenPC 已验证的竖直坐标约定。

深度图与 GPT 语义图不要求逐像素完全重合。当前方法把它们视为大致对齐，
真正的几何约束来自保存相机下的 silhouette、same-ray visible depth、
coverage、leakage 和局部 3D surface evidence。

## 4. Stage 1：GPT ImageGen 完整语义图

### 4.1 目的

将 partial 深度图补全为同一视角、同一位置和相近投影尺寸下的完整真实物体图，
为 Pixal3D 提供高质量完整语义先验。

### 4.2 输入输出

- 输入：`gpt_version/<sample>/depth.png`；
- 完整 prompt：`gpt_version/<sample>/prompt.txt`；
- 输出：`gpt_version/<sample>/gpt_image.png`，`1254×1254`；
- `01184`、`05452` 还有用户提供的结构参考图。

### 4.3 Prompt 原则

1. depth 是相机、姿态、投影大小、位置和可见轮廓的权威输入；
2. 只补全缺失/遮挡部分，不旋转到 canonical product view；
3. 不重新居中、不 zoom、不改变物体在画面中的占比；
4. 只生成一个完整物体，纯白背景；
5. 禁止地面、阴影、文字、水印、人物和额外物体；
6. 类别只用于说明物体是什么；
7. 结构参考只约束真实部件关系，例如 `01184` 的两个平行同轴轮子、
   `05452` 的薄曲面椅背，不进入后续样本分支。

GPT ImageGen 的在线 serving checkpoint、seed、steps、CFG、scheduler 和
negative prompt 不公开，因此记录为 `UNKNOWN`。这一阶段依靠冻结输出和
逐样本 verbatim prompt 复现研究状态，不声称可由本地权重精确重跑。

## 5. Stage 2：Pixal3D 完整 3D 生成

### 5.1 实现与模型

实现：`scripts/run_pixal3d_gpt_batch.py`。

- Pixal3D 源码：`models/Pixal3D`；
- Pixal3D 权重：`models/Pixal3D-weights`；
- DINOv3：`models/dinov3-vitl16-pretrain-lvd1689m`；
- Pixal3D 内部 MoGe-2：`models/moge-2-vitl/model.pt`；
- RMBG：`models/RMBG-2.0`；
- attention backend：`xformers`；
- NAF：官方 release checkpoint。

Pixal3D 内部已经使用 MoGe-2 估计相机 FOV 和距离，因此旧流程中的外部
partial→MoGe→complete transform composition 不再需要。

### 5.2 预处理

Pixal3D 官方 preprocessing：

- 去除白色背景；
- 使用官方 `1.1` foreground crop；
- 保存实际输入为 `pixal3d_input.png`；
- 根据该输入使用本地 MoGe-2 估计 camera parameters。

### 5.3 十样本共享生成参数

- seed：`42`；
- pipeline：`1024_cascade`；
- sparse structure：12 steps，guidance 7.5，rescale 0.7，`rescale_t=5.0`；
- shape：12 steps，guidance 7.5，rescale 0.5，`rescale_t=3.0`；
- texture：12 steps，guidance 1.0，rescale 0，`rescale_t=3.0`；
- GLB remesh target：300000 faces；
- texture：2048；
- PLY：seed 42 deterministic uniform surface sampling，100000 points。

### 5.4 输出与冻结边界

- `gpt_version/<sample>/pixal3d_input.png`；
- `gpt_version/<sample>/pixal3d.glb`；
- `gpt_version/<sample>/pixal3d_sampled_100k.ply`；
- `gpt_version/<sample>/pixal3d_metadata.json`。

原始 GLB 和 100k PLY 是完整性主体。配准与融合只能写 derivative，不能覆盖
这些文件，也不能为某个难例重新生成 GLB。

### 5.5 重跑命令

已有接受结果默认会跳过，不加 `--overwrite`：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
/opt/data/private/cr/miniconda3/envs/genpc/bin/python \
scripts/run_pixal3d_gpt_batch.py \
  --ids 01184 05117 05452 06127 06145 06188 06830 07136 07306 09639
```

## 6. Stage 3：v15 visibility-aware proper-Sim3 配准

### 6.1 变换约束

配准只允许：

```text
p_registered = s R p_pixal + t
```

其中 `R∈SO(3)`、`s>0` 是一个统一尺度、`t∈R³`。禁止 axis-wise scale、
shear、affine warp、point deletion 和 non-rigid deformation。

### 6.2 为什么使用 visible partial-to-partial

完整 Pixal 模型包含不可见背面。直接 complete-to-partial ICP 或 symmetric
Chamfer 会错误惩罚这些合法完整区域，容易把沙发长度、桌面或物体整体尺度缩小。

当前方法将 complete Pixal 通过保存相机做 z-buffer，只取当前视角可见且受到
partial 支持的局部表面，再与 raw partial 比较。优化等价于：

```text
visible part of complete Pixal  ↔  observed raw partial
```

不可见完整区域不进入 partial-distance loss，因此保持 Pixal3D 的完整性。

### 6.3 两条候选路线

v15 是一个 GT-free guarded router：

```text
frozen Pixal + raw partial + saved camera
  ├─ v8 GenPC/PCA proper-Sim3 fallback
  └─ v12 batched global SO(3) + visible-depth/local TTT
          ├─ observable confidence gate
          └─ full-resolution do-no-harm guard
                 ├─ pass：采用 fast SO(3) TTT
                 └─ fail：采用 v8 fallback
```

#### Fast SO(3) 路线

- 24 个 proper PCA rotations；
- 叠加固定 Euler offset lattice；
- 共 648 个 rotation hypotheses；
- scales：`0.6, 0.8, 1.0, 1.2, 1.4`；
- GPU batched coarse z-buffer ranking；
- 每个 rotation 只保留最佳 coarse scale；
- 12 个 fine 3D candidates；
- 最多 8 个 local rotation refinements；
- evidence 足够时 early stop。

Fast observable gate：

- low-resolution IoU `>= 0.85`；
- coverage `>= 0.90`；
- leakage `<= 0.08`；
- normalized visible-depth error `<= 0.10`；
- normalized trim70 surface error `<= 0.012`。

#### Full-resolution do-no-harm guard

在 `512×512` 下定义：

```text
render_score = IoU + 0.15 × coverage - 0.45 × leakage
```

fast route 只有同时满足 observable gate，并且其 full-resolution score 不低于
fallback score 超过 `0.005`，才允许替换 fallback。

### 6.4 v15 十样本路由

| sample | v15 route |
| --- | --- |
| 01184 | GenPC/PCA fallback |
| 05117 | GenPC/PCA fallback |
| 05452 | GenPC/PCA fallback |
| 06127 | GenPC/PCA fallback |
| 06145 | fast global SO(3) TTT |
| 06188 | GenPC/PCA fallback |
| 06830 | fast global SO(3) TTT |
| 07136 | GenPC/PCA fallback |
| 07306 | GenPC/PCA fallback |
| 09639 | GenPC/PCA fallback |

共享 gate 自动升级了桌面长宽轴有问题的 `06145` 和姿态较差的 `06830`，同时
保留其他样本，尤其避免再次缩短 `07136` 沙发。

### 6.5 输出

根目录：`gpt_version/_pixal_guarded_unified_registration_v15_20260822`。

每个样本包含：

- `<sample>_unified_registration_v14_registered_100k.ply`；
- `<sample>_unified_registration_v14_registered_mesh.glb`；
- `<sample>_unified_registration_v14.npy`；
- `<sample>_unified_registration_v14_partial_gray_pixal_red.ply`；
- `<sample>_unified_registration_v14_projection.png`；
- `<sample>_unified_registration_v14_info.json`。

文件 stem 保留 `v14` 是历史实现命名，外层方法与输出 root 才是 v15 guarded
router。判断方法版本时应读取 JSON 的 `method` 字段和 root，而不是只看 stem。

### 6.6 当前重跑路由命令

当前保留了 v8 fallback 与 v12 fast roots，因此可直接重建 v15 routed outputs：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
/opt/data/private/cr/miniconda3/envs/genpc/bin/python \
scripts/select_pixal_guarded_unified_registration_v15.py
```

如果从零生成 v8/v12，需要先运行其历史前置流程。当前尚未把全部历史 stage
封装成单一 launcher；论文代码整理时应把 v8 fallback 导出、v12 fast search
和 v15 router 合并成一个只读 frozen-Pixal 入口。

## 7. Stage 4：v18-gatefix partial-core 局部融合

### 7.1 要解决的问题

proper-Sim3 可以对齐整体 pose 和统一尺度，但无法让不同 prior 中的局部部件
完全一致，例如椅子腿、沙发局部边缘、扶手和轮子附近可能仍有双层表面。

融合目标是：

- partial 是局部真实几何核心；
- Pixal3D 是完整性主体；
- 只把 partial 附近、高残差且可信的 generated surface 拉向 partial；
- 背面、远场、已经对齐区域保持不动；
- 不通过删掉 Pixal 模型来消除重影。

### 7.2 Same-camera 对应

1. 用保存相机分别 z-buffer raw partial 与 registered Pixal；
2. 在相同或邻近像素上建立 visible correspondences；
3. 使用 depth、3D distance、normal 和 image-grid coverage 做几何 gate；
4. 对映射到同一个 generated point 的多个 partial hits 取 median target；
5. 只保留高残差尾部作为局部编辑 handles。

残差阈值为：

```text
tau = max(0.006 D, quantile_0.55(residual))
```

其中 `D` 是 raw partial bounding-box diagonal。低残差区域不直接作为拉动把手。

### 7.3 Sparse translation-only deformation graph

在 registered 100k Pixal 点上构建稀疏图：

- node spacing ratio：`0.045`；
- 96–256 nodes；
- graph KNN：6；
- 每个 surface point 受 4 个近邻 nodes 影响；
- 只有距离 handles 小于 `0.055` graph diagonal 的 nodes 激活；
- 每个 node 只有 translation，没有 local rotation、scale 或 affine matrix。

优化目标可概括为：

```text
L = L_robust-handle + 35 L_graph-smooth + 14 L_identity
```

- Adam，120 iterations；
- learning rate：`0.02`；
- data trim quantile：`0.80`；
- node displacement 最大 `0.03` graph diagonal。

### 7.4 点级 C1 compact support

仅靠 graph node 激活仍可能把很小的位移传播到较远区域，因此 v18 再加入
point-level compact window。设 point 到最近 handle 的距离为 `d`：

```text
u = clip((0.070 D - d) / (0.070 D - 0.025 D), 0, 1)
w = u²(3 - 2u)
p_deformed = p + w Δp_graph
```

- `d <= 0.025D`：保留完整 graph displacement；
- `0.025D < d < 0.070D`：C1 smooth decay；
- `d >= 0.070D`：位移严格为 0。

因此远场和无观测完整区域不是“强正则近似不动”，而是数学上直接不动。

### 7.5 融合安全门

局部编辑必须同时满足：

- visible-correspondence gate 通过；
- handles 数量至少 256；
- handle mean distance 改善至少 8%；
- active graph nodes 比例不超过 80%；
- far-field maximum displacement ratio 不超过 0.001；
- silhouette IoU 下降不超过 0.015；
- coverage 下降不超过 0.015；
- leakage 增加不超过 0.02。

任一条件失败，`deformed_complete` 立即回退为未形变的 v15 registered 100k。
回退逻辑不查看 GT 或 sample ID。

注意：最初 v18 diagnostic 没有把 visible-correspondence gate 组合进最终 accept
条件。全十样本运行发现该问题后，当前有效入口增加了强制 gate。必须使用：

```text
scripts/run_pixal_strict_compact_local_edit_v18_gatefix.py
```

不要直接把早期未 gatefix 的 `run_pixal_strict_compact_local_edit_v18.py`
作为最终批处理入口。

### 7.6 最终 fused cloud

无论 local edit 是否通过：

```text
fused = full_100k_pixal_body + exact_raw_partial
```

通过时使用 locally deformed Pixal body；失败时使用原 registered Pixal body。

关键 invariants：

- `all_pixal_points_retained = true`；
- `partial_core_exact = true`；
- `points_deleted = false`；
- `local_scale_or_shear_used = false`；
- canonical registered output 保持不变；
- deformed mesh 和 PLY 都另存为 derivative。

### 7.7 十样本路由结果

| sample | fusion route | 主要 gate 结果 |
| --- | --- | --- |
| 01184 | local edit | accepted |
| 05117 | safe union fallback | active-node ratio too high |
| 05452 | safe union fallback | active-node ratio too high |
| 06127 | safe union fallback | visible coverage gate failed |
| 06145 | safe union fallback | active-node ratio too high |
| 06188 | local edit | accepted |
| 06830 | safe union fallback | visible coverage gate failed |
| 07136 | local edit | accepted |
| 07306 | local edit | accepted |
| 09639 | local edit | accepted |

“safe union fallback” 仍然完成融合，只是不允许不可靠的 non-rigid local edit。

### 7.8 当前批处理命令

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
/opt/data/private/cr/miniconda3/envs/genpc/bin/python \
scripts/run_pixal_strict_compact_local_edit_v18_gatefix.py \
  --samples 01184 05117 05452 06127 06145 06188 06830 07136 07306 09639
```

输出根：

```text
gpt_version/_pixal_strict_compact_local_edit_v18_20260822
```

当前 derivative 文件仍使用 `residual_local_edit_v17` stem，因为 v18 是建立在
v17 optimizer 之上的严格 compact-support/gate wrapper。论文代码清理时应合并为
单个正式实现并统一命名，但不得在合并时改变冻结结果。

## 8. Stage 5：冻结后 CD-L1 / EMD

GT 只允许在所有 prediction 与 route 冻结后读取。

评估协议：

- `utils.loss_util.Completionloss`；
- CD-L1 与 EMD；
- deterministic FPS；
- 每个 prediction/GT 最多 16384 points；
- metric seed：6145；
- 不用 metric 重新选择 route、参数或 output。

评估器要求每个样本存在 `<sample>/<sample>_fused.ply`。当前 v18 root 已建立
到真实 fused PLY 的规范链接，可直接运行：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
/opt/data/private/cr/miniconda3/envs/genpc/bin/python \
scripts/evaluate_redwood_fused_root.py \
  --output-root gpt_version/_pixal_strict_compact_local_edit_v18_20260822 \
  --samples 01184 05117 05452 06127 06145 06188 06830 07136 07306 09639 \
  --seed 6145
```

## 9. 当前全十样本结果

指标为 `CD-L1/EMD x1e2`：

| sample | fusion route | current CD | current EMD | GenPC CD | GenPC EMD |
| --- | --- | ---: | ---: | ---: | ---: |
| 01184 | local edit | 1.188995 | 1.924919 | 2.31 | 3.17 |
| 05117 | fallback | 1.771338 | 3.032858 | 1.36 | 2.20 |
| 05452 | fallback | 0.934114 | 1.467725 | 1.16 | 1.68 |
| 06127 | fallback | 2.315015 | 4.420082 | 2.86 | 4.85 |
| 06145 | fallback | 0.856748 | 1.469414 | 1.28 | 2.07 |
| 06188 | local edit | 1.141484 | 1.944241 | 1.36 | 2.47 |
| 06830 | fallback | 1.964918 | 3.903052 | 1.38 | 2.97 |
| 07136 | local edit | 1.750700 | 2.535489 | 1.58 | 2.78 |
| 07306 | local edit | 2.819940 | 3.409398 | 2.72 | 4.36 |
| 09639 | local edit | 2.013308 | 3.161812 | 1.43 | 2.29 |
| **mean** | **5 edit / 5 fallback** | **1.675656** | **2.726899** | **1.74** | **2.88** |

结论：

- full-ten mean 的 CD 和 EMD 都超过 GenPC 论文均值；
- `01184`、`05452`、`06127`、`06145`、`06188` 同时超过各自 GenPC 行；
- `07136` 的 EMD 超过 GenPC，但 CD 仍略差；
- `06830`、`07306`、`09639` 是当前主要风险；
- 不能只报告平均值，必须保留逐样本结果和视觉审计。

## 10. 输出目录结构

```text
gpt_version/
├─ <sample>/
│  ├─ depth.png
│  ├─ prompt.txt
│  ├─ gpt_image.png
│  ├─ pixal3d_input.png
│  ├─ pixal3d.glb
│  ├─ pixal3d_sampled_100k.ply
│  └─ pixal3d_metadata.json
│
├─ _pixal_guarded_unified_registration_v15_20260822/
│  └─ <sample>/
│     ├─ *_registered_100k.ply
│     ├─ *_registered_mesh.glb
│     ├─ *.npy
│     ├─ *_partial_gray_pixal_red.ply
│     ├─ *_projection.png
│     └─ *_info.json
│
└─ _pixal_strict_compact_local_edit_v18_20260822/
   ├─ metrics_samples.csv
   ├─ fusion_summary.csv
   ├─ fusion_summary.json
   ├─ fusion_contact_sheet.png
   └─ <sample>/
      ├─ *_deformed_complete_100k.ply
      ├─ *_fused.ply
      ├─ <sample>_fused.ply
      ├─ *_partial_gray_deformed_red.ply
      ├─ *_deformed_mesh.glb
      ├─ *_info.json
      └─ <sample>_v18_before_after.png
```

## 11. 视觉检查方法

对比 PLY 的颜色约定：

- partial：gray；
- registered/deformed Pixal：red。

单样本四视角渲染：

```bash
PYTHONPATH=. /opt/data/private/cr/miniconda3/envs/genpc/bin/python \
scripts/render_pixal_local_edit_comparison.py \
  --sample 09639 \
  --before gpt_version/_pixal_guarded_unified_registration_v15_20260822/09639/09639_unified_registration_v14_partial_gray_pixal_red.ply \
  --after gpt_version/_pixal_strict_compact_local_edit_v18_20260822/09639/09639_residual_local_edit_v17_partial_gray_deformed_red.ply \
  --output gpt_version/_pixal_strict_compact_local_edit_v18_20260822/09639/09639_v18_before_after.png
```

全十样本总览：

```text
gpt_version/_pixal_strict_compact_local_edit_v18_20260822/fusion_contact_sheet.png
```

视觉审核优先检查：

1. partial 与 generated 是否仍有双层表面；
2. 椅子腿、轮子、扶手、沙发边缘是否被自然拉近；
3. complete backside 或长结构是否被缩短/扭曲；
4. local edit 与 fixed region 之间是否出现折痕或密度断层；
5. fallback 样本是否仍有明显生硬并集接缝；
6. 原始 Pixal 完整结构是否全部保留。

## 12. 当前限制与下一步

### 12.1 当前限制

- 5/10 样本触发保守 fallback，说明 shared local edit 还不能覆盖所有形状；
- direct union fallback 可能保留局部 double wall；
- fused point count 随 partial 数量变化，虽然 metric 会统一 FPS 到 16384；
- 09639 可见 handles 分布较广，仍需重点检查椅子腿和底座的视觉形变；
- 07306 指标略差于其 GenPC CD 行，需要确认局部编辑是否真的有视觉收益；
- 06830 因 visible coverage 不足禁止 local edit，目前仍是主要失败样本；
- GPT ImageGen 在线阶段不能精确重跑，只能依赖冻结图像；
- 当前 v16→v17→v18→gatefix 是清晰的实验演化，但正式代码入口仍偏分散；
- SDS 尚未进入当前结果。它只适合作为低权重 seam ablation，不能更新全局
  pose/scale 或破坏 Pixal 完整模型。

### 12.2 推荐下一步

1. 先逐样本视觉确认 v18 contact sheet 和 fused PLY；
2. 把 v16/v17/v18/gatefix 合并为一个正式、无 monkeypatch 的 fusion module；
3. 保持所有当前参数和输出不变，先做等价重构测试；
4. 对 fallback 样本研究 confidence-weighted local handles，而不是降低全局安全门；
5. 增加 seam density/normal continuity 的 GT-free visual gate；
6. 做 direct union、translation graph、compact support、visible-gate fallback 的
   独立消融；
7. 在不访问 GT 的情况下跑更多类别/数据集，验证真正泛化性；
8. 最后才尝试低权重 SDS 或 mesh-based ARAP seam refinement。

## 13. 复现记录与相关文档

- 当前核心规范：`docs/core_registration_pipeline.md`；
- v15 配准设计：`docs/fast_unified_registration_v15.md`；
- 早期 2D/3D TTT 设计：`docs/visibility_aware_pixel_sim3_ttt.md`；
- v18 full-ten 精确结果：
  `reproducibility/pixal_v18_full10_fusion_20260822.md`；
- 接受输出与冻结边界：`PROJECT_STATE.md`；
- 当前任务状态：`PLAN.md`。

当前最重要的冻结原则是：不覆盖 GPT semantic images、原始 Pixal3D
GLB/100k PLY 和 v15 registered outputs；任何进一步融合改进都写入新的 derivative
root，并在冻结后再读取 GT 评估。
