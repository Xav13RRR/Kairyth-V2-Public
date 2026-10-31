# Backend resource use / 后端资源用途

This map describes the included workbench and the resources consumed by its supported workflows. See [deployment](DEPLOYMENT_EN.md), [resource checks](RESOURCE_READINESS.md) and [license details](LICENSE_DECISIONS.md).

本文说明当前工作台各功能实际需要的资源，区分生成新结果与查看/编辑已有资产。

## Feature map / 功能映射

| Feature | Included implementation | Required dependencies |
| --- | --- | --- |
| Asset libraries, animation editing, IK, terrain and director | `motionbuilder_app/` stores, math and frontend modules | Valid input assets, base Python dependencies and the bundled skeleton/preview resources; no new inference checkpoint merely to browse a saved motion |
| New automatic rig | `meshpreps/bind_pipeline.py`: UniRig skeleton → SOMA fitting → SkinTokens skinning | Selected skeleton / skin checkpoints and each backend's Python/CUDA extensions |
| Kimodo text/constraint motion | `kimodo_store.py` → worker → `kimodo_runtime.py` | Motion model, normalization statistics, text encoder and applicable skeleton resources |
| Existing poses and motions | `pose_store.py`, `stores.py`, `library_stores.py` | Existing numeric bundles; importing/editing/extracting poses does not invoke the unavailable image/video models |
| FBX import/export | Background Blender helpers | Blender executable and trusted input files |

New HybrIK image inference and GVHMR video inference are experimental and unavailable; no compatibility package is included. Their client contracts are documented in [backend interfaces](OPTIONAL_RESEARCH_BACKENDS.md).

## Kimodo: motion, text and skeleton resources

The default motion family is `nvidia/Kimodo-SOMA-RP-v1.1`. A complete snapshot needs `config.yaml`, `model.safetensors` and all six statistics files:

```text
stats/motion/global_root/mean.npy
stats/motion/global_root/std.npy
stats/motion/local_root/mean.npy
stats/motion/local_root/std.npy
stats/motion/body/mean.npy
stats/motion/body/std.npy
```

The model configuration selects `stats/motion/`; these files are used to normalize and unnormalize motion features, not only for training. Keep them from the same model snapshot.

| Resource | Role | Fixed reference |
| --- | --- | --- |
| Kimodo model | Diffusion motion generation, separate from text encoding | [Model `6c9233af1180b8151e3c4703477104af5dce9dd5`](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1/tree/6c9233af1180b8151e3c4703477104af5dce9dd5), NVIDIA Open Model License; model SHA256 `ef0a0ca45a6089ab4532dde609785771ae3f38755b4ae6cf314b0213e07cd4a3` |
| LLM2Vec MNTP adapter | Text embeddings using a separately licensed base model | [Adapter config `31474e395ada192e8ed1586db6be79fb3b70c9c0`](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/blob/31474e395ada192e8ed1586db6be79fb3b70c9c0/adapter_config.json) |
| LLM2Vec supervised adapter | Additional text-encoder adapter | [Snapshot `baa8ebf04a1c2500e61288e7dad65e8ae42601a7`](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/tree/baa8ebf04a1c2500e61288e7dad65e8ae42601a7), MIT for its material |
| Meta Llama 3 base | Text encoding, not ordinary library browsing, FK, IK or manual editing | [Agreement `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2/LICENSE) |
| SOMA30 / SOMA77 arrays | Skeleton construction, conversion, skinning helpers and relevant previews | [Kimodo source `e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954`](https://github.com/nv-tlabs/kimodo/tree/e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954/kimodo/assets/skeletons), exact included entries in [resource evidence](resource-license-evidence.json) |

The loader may access model services when cache files are missing; configuration/tokenizer requests also need explicit offline settings for no-network diagnostics. Model families and adapters can have independent revisions and terms. `smplx22`, G1 and MHR alternatives are not implicit requirements of the application's SOMA-RP path.

## UniRig and SkinTokens

The default pipeline is **UniRig skeleton + SOMA fitting + SkinTokens skin**, not every checkpoint available from either upstream project.

| Resource | Actual use |
| --- | --- |
| UniRig `experiments/skeleton/articulation-xl_quantization_256/model.ckpt` | Default skeleton proposal selected by the application task configuration |
| UniRig Michelangelo encoder | Selected by the model configuration and instantiated in the skeleton model; retain GPL-3.0-or-later source/notices |
| `facebook/opt-350m` configuration | Model configuration is obtained before construction from config; this does not imply loading separate OPT pretrained base weights |
| UniRig independent skin checkpoint | Not selected by the default application skinning path |
| SkinTokens `grpo_1400.ckpt` | TokenRig skin-token prediction from mesh and supplied skeleton |
| SkinTokens `skin_vae_2_10_32768/last.ckpt` | VAE constructed by TokenRig; decodes tokens into vertex/bone weights and is also required |
| SkinTokens Michelangelo / Qwen configuration | Mesh encoder and model configuration; nested GPL code and checkpoint rights remain distinct. A Qwen config identifier does not itself require Qwen base weights |

Fixed model references: [UniRig `36842e2b5947e9e60f89275b83208c8e74071c63`](https://huggingface.co/VAST-AI/UniRig/tree/36842e2b5947e9e60f89275b83208c8e74071c63), skeleton checkpoint SHA256 `d8cf9b42d56e7bc316d293597ecf8c4c39a8631cdd44a261ecf388242fabd0f4`; [SkinTokens `79736cad0fd84de384d5eede659b4ebd24effe33`](https://huggingface.co/VAST-AI/SkinTokens/tree/79736cad0fd84de384d5eede659b4ebd24effe33), TokenRig SHA256 `f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692`, VAE SHA256 `4843f49e58afff88345806b94ca82e6cc9d8def6e7432e2853c677b154de0ed4`.

## Blender, VRM and frontend assets

Blender performs FBX conversion through the included `bpy` helpers. Its software terms are separate from the rights in imported models and generated artwork. The modified upstream VRM add-on archive is not included or required by ordinary application FBX export; its specific modifications and bundled terms would need separate review before distribution.

The UI includes project artwork and the Geist font identified in the resource inventory. Geist retains its OFL notices. Oswald is not bundled; configured fallback fonts are used.
