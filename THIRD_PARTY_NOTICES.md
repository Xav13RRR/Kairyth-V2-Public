# Third-party notices / 第三方许可说明

Third-party software and resources retain their existing copyrights and licenses. The compatible workbench combination uses GPLv3, preserving upstream or-later rights; maintainer-original Kairyth / MKA contributions additionally use Apache-2.0. See [licensing scope](LICENSE_SCOPE.md).

第三方代码与资源保留各自版权及条款。顶层许可证不覆盖所有嵌套组件，代码许可也不自动覆盖模型、权重、字体或图像。

| Component / 组件 | Source / 来源 | Terms / 条款 |
| --- | --- | --- |
| MKA predecessor contributions | Maintainer-original work incorporated into Kairyth | [Apache-2.0](licenses/MKA-Apache-2.0.txt); adapted third-party portions retain their own terms |
| NVIDIA Kimodo | [nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo), `kimodo_runtime/src/` | [Apache-2.0](licenses/Kimodo-Apache-2.0.txt), including preserved NVIDIA notices. Models and data have separate terms |
| UniRig | [VAST-AI-Research/UniRig](https://github.com/VAST-AI-Research/UniRig), `third_party/UniRig/` | Top-level [MIT](licenses/UniRig-MIT.txt), subject to nested component licenses |
| Michelangelo within UniRig | `third_party/UniRig/src/model/michelangelo/` | [GPL-3.0-or-later](third_party/UniRig/src/model/michelangelo/LICENSE), with original per-file headers and [notice](third_party/UniRig/src/model/michelangelo/NOTICE.md) |
| Pointcept within UniRig | `third_party/UniRig/src/model/pointcept/` | [MIT](third_party/UniRig/src/model/pointcept/LICENSE) |
| SkinTokens | [VAST-AI-Research/SkinTokens](https://github.com/VAST-AI-Research/SkinTokens), `third_party/SkinTokens/` | Top-level [MIT](licenses/SkinTokens-MIT.txt), with nested GPL / Apache / MIT sources; [nested notices](third_party/SkinTokens/THIRD_PARTY_NOTICES.md) |
| Michelangelo-derived SkinTokens files | `src/model/michelangelo/**` and `src/model/skin_vae/autoencoders/miche_transformer_blocks.py` | [GPLv3](licenses/Michelangelo-GPL-3.0.txt), version 3 selected for distribution without asserting or removing an unverified later-version option; [scope](third_party/SkinTokens/GPL_SOURCE_NOTICE.md) |
| SkinTokens quantizers and helpers | vector-quantize-pytorch, SimVQ, taming-transformers, TripoSG, guided-diffusion, diffusers, 3DShape2VecSet | Fixed references and retained MIT / Apache notices in the [component inventory](third_party/SkinTokens/THIRD_PARTY_NOTICES.md) and [source provenance](docs/SOURCE_PROVENANCE.md) |
| PyTorch3D compatibility shim | Comparison reference `89653419d0973396f3eff1a381ba09a07fffc2ed`, `pytorch3d/` | [BSD-3-Clause](pytorch3d/LICENSE), [attribution and adaptation scope](pytorch3d/NOTICE.md) |
| Qwen configuration and tokenizer | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), `third_party/SkinTokens/models/Qwen3-0.6B/` | Included [Apache-2.0](third_party/SkinTokens/models/Qwen3-0.6B/LICENSE); model weights are not included |
| Three.js r172 | [mrdoob/three.js r172](https://github.com/mrdoob/three.js/tree/r172), `motionbuilder_app/static/vendor/three/` | [MIT](licenses/Threejs-MIT.txt), with existing copyright headers |
| Blender | Downloaded separately from the official distribution | GPL and the notices accompanying that distribution; Blender binaries are not committed here |

## GPL source notices

The [26-file inventory](docs/gpl-source-evidence.json) identifies ten UniRig files with explicit GPL-3.0-or-later headers and fifteen SkinTokens subtree files plus one directory-external derived file distributed under GPLv3. The SkinTokens checkpoint implementation also retains OpenAI guided-diffusion MIT attribution. Source distributions must carry the preferred-form source, applicable configuration/build/install source, complete license texts and notices. An upstream link is not a replacement for the corresponding source of a modified compiled distribution.

## Included resources

The included minimum SOMA resources retain **Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES**, Apache-2.0. The application float32 joint array matches the float32 conversion of the pinned official joint values. Exact paths and hashes are listed in [resource evidence](docs/resource-license-evidence.json).

Geist Sans is provided under SIL OFL 1.1: **Copyright 2024 The Geist Project Authors**; repository notices also identify **Copyright 2023 Vercel Inc. and basement.studio**. Both original release notices are preserved; see [font notices](motionbuilder_app/static/fonts/FONT-NOTICES.md) and [fixed source evidence](docs/font-provenance-evidence.json). Oswald is not bundled.

Project artwork is identified per asset in the resource inventory. Its distribution scope does not grant additional trademark rights or rights to other artwork.

## Separately licensed, excluded components

GVHMR, HybrIK research implementations, DPVO, detector code and model weights are not part of this source package. Relevant upstream sources and terms include:

- [GVHMR](https://github.com/zju3dv/GVHMR) and its [pinned research license](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/LICENSE), with independent nested/body-model terms.
- [HybrIK](https://github.com/jeffffffli/HybrIK): top-level MIT does not replace nested MPG / SMPL / SMPL-X program and model terms.
- [Ultralytics 8.2.42](https://github.com/ultralytics/ultralytics/blob/v8.2.42/LICENSE) uses AGPL terms. Any future integration must satisfy the applicable distribution and network-service obligations.

Model weights, text encoders, body models, regressors, modified VRM add-ons and other assets require their own applicable permissions. Code inclusion does not imply model or asset redistribution permission. See [resource readiness](docs/RESOURCE_READINESS.md) and [license details](docs/LICENSE_DECISIONS.md).
