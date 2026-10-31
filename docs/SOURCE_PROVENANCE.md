# Source provenance / 源码来源

This document identifies upstream comparison references, nested licenses and the integration changes shipped with Kairyth. Exact file identities are in [SkinTokens source evidence](source-provenance-skintokens.json), [GPL file evidence](gpl-source-evidence.json) and [resource evidence](resource-license-evidence.json).

本文记录上游来源、嵌套许可和随附的集成改动。比较版本用于核对文件来源，不冒称已确定每段代码最初引入的提交；同一文件中的第三方内容保留其原权利。

## SkinTokens comparison and adaptations

Reference: [SkinTokens `273b691d35989d71cd17ff2895fdc735097b92d1`](https://github.com/VAST-AI-Research/SkinTokens/tree/273b691d35989d71cd17ff2895fdc735097b92d1).

The 63 Python files comprise 59 exact matches after CRLF→LF normalization, two modified files and two maintainer-original integration additions. The [file inventory](source-provenance-skintokens.json) preserves paths, normalized Git blob hashes, upstream URLs and license evidence.

| Adaptation | Description |
| --- | --- |
| `src/model/skin_vae_model.py` | Flash-attention import adapted to the compatibility helper |
| `src/model/tokenrig.py` | Compatibility fallback when flash-attn is unavailable |
| `src/model/_flash_attn.py` | Maintainer-original Windows / SDPA integration helper; adapted third-party portions retain their terms |
| `scripts/motionbuilder_skin_infer.py` | Maintainer-original NPZ inference bridge integrated with the workbench |

## Nested sources and licenses

| Files / material | Fixed source | Terms and attribution |
| --- | --- | --- |
| SkinTokens `src/model/michelangelo/**` and `skin_vae/autoencoders/miche_transformer_blocks.py` | [NeuralCarver/Michelangelo `6d83b0bacef92715dd5179d45647ed9a3d39bc95`](https://github.com/NeuralCarver/Michelangelo/tree/6d83b0bacef92715dd5179d45647ed9a3d39bc95) | GPLv3 selected for this distribution; retain original authors and [nearby notice](../third_party/SkinTokens/GPL_SOURCE_NOTICE.md) |
| Checkpoint implementation | [OpenAI guided-diffusion `22e0df8183507e13a7813f8d38d51b072ca1e67c`](https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/nn.py) | MIT notice retained within the derived GPL-compatible material |
| `FSQ.py` | [Phil Wang / vector-quantize-pytorch `9e5d4ecfdfd72abe9a4baa056d0be21532be4b69`](https://github.com/lucidrains/vector-quantize-pytorch/blob/9e5d4ecfdfd72abe9a4baa056d0be21532be4b69/vector_quantize_pytorch/finite_scalar_quantization.py) | MIT |
| `SimVQ.py` | [SimVQ `d8bd94d9ca30273c8bab1b4194d8e1dc0b9d3f70`](https://github.com/youngsheen/SimVQ/blob/d8bd94d9ca30273c8bab1b4194d8e1dc0b9d3f70/taming/modules/vqvae/quantize.py), earlier [taming-transformers `3ba01b241669f5ade541ce990f7650a3b8f65318`](https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py) | Preserve both MIT sources |
| Attention, embeddings, VAE, modeling outputs and partial Tripo2 encoder | [TripoSG `fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c`](https://github.com/VAST-AI-Research/TripoSG/tree/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c) | MIT near-source attribution with earlier diffusers / Hunyuan notices where present; partial encoder derivation is not a complete segment-level mapping |
| Tripo2 transformer / copied diffusers methods | [diffusers `86ef420b17363c28e36ebad6b58555e9aa8feda2`](https://github.com/huggingface/diffusers/tree/86ef420b17363c28e36ebad6b58555e9aa8feda2) | Preserve Apache-2.0 headers crediting HunyuanDiT Authors, Qixun Wang and HuggingFace |
| FSQ-CVAE backbone attribution | [Biao Zhang / 3DShape2VecSet `8df9b7a55c42d4dcad152294755250a2ab1e34e5`](https://github.com/1zb/3DShape2VecSet/tree/8df9b7a55c42d4dcad152294755250a2ab1e34e5) | MIT, module-level attribution |
| UniRig Michelangelo source | [UniRig `6793c6640ff01c8fb389f3993434124bb43d2933`](https://github.com/VAST-AI-Research/UniRig/tree/6793c6640ff01c8fb389f3993434124bb43d2933/src/model/michelangelo) | Ten files preserve explicit GPL-3.0-or-later headers, NeuralCarver/Michelangelo credit and VAST-AI-Research copyright |

GPL provenance includes the directory-external `miche_transformer_blocks.py`, not only the named Michelangelo folder. The 26-file inventory records exact version evidence. Missing headers in the selected SkinTokens snapshot do not turn GPL-derived material into MIT-only code. Existing later-version rights are preserved; an unverified later option is not asserted for SkinTokens.

## Core and compatibility code

- Maintainer-original Kairyth / MKA UI, editing, orchestration and integration contributions are additionally offered under Apache-2.0; copied third-party material retains its own terms.
- `motionbuilder_app/rigging/joint_names.py`, `soma_fit.py` and `soma_retarget_profile.py` retain NVIDIA 2026 Apache-2.0 notices. Exact original upstream paths for those adaptations are not established by a same-name lookup.
- `pytorch3d/transforms/rotation_conversions.py` is a modified compatibility subset with Meta copyright and BSD-3-Clause. Comparison reference: [PyTorch3D v0.7.7 / `89653419d0973396f3eff1a381ba09a07fffc2ed`](https://github.com/facebookresearch/pytorch3d/blob/89653419d0973396f3eff1a381ba09a07fffc2ed/pytorch3d/transforms/rotation_conversions.py). It is not a full upstream API distribution.
- Bundled Three.js source retains its r172 MIT notices.

The SkinTokens runtime path includes `motionbuilder_skin_infer.py` → TokenRig → `parse_encoder` / Michelangelo components. These are functional dependencies, not merely unused license files; top-level MIT does not replace nested GPL obligations.

## Resources and excluded implementations

Included SOMA resources are identified against [Kimodo `e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954`](https://github.com/nv-tlabs/kimodo/tree/e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954), retaining NVIDIA Apache notices. The application joint array matches the official float64 joints after float32 conversion; its original dtype is not identical. The maximum pre-conversion difference is `2.9135885926123706e-08`. Geist matches [official release 1.4.2 / `4d655731f83507f0258ce5541c297b6553d5d081`](https://github.com/vercel/geist-font/tree/4d655731f83507f0258ce5541c297b6553d5d081), with original OFL notices.

Research implementations, body-model resources and weights not supplied by this package retain their independent terms. [Excluded-source evidence](source-held-evidence.json) distinguishes explicit restricted headers from unresolved wrapper/family provenance. In particular, HybrIK's SMPL / SMPL-X families include MPG-noticed programs; GVHMR's body-model utility contains SMPLify-X-attributed code. Their package-level licenses do not replace those terms. References: [SMPL-X license](https://github.com/vchoutas/smplx/blob/main/LICENSE), [SMPLify-X utility source](https://github.com/vchoutas/smplify-x/blob/master/smplifyx/utils.py).

Model weights, training data and artwork have their own terms; source identity does not establish all rights in those resources. See [licensing scope](../LICENSE_SCOPE.md) and [resource usage](RESEARCH_BACKEND_USAGE.md).
