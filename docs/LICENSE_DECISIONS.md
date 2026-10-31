# License details / 许可说明

The workbench code, model weights and input assets can have different licenses. Use the [licensing scope](../LICENSE_SCOPE.md) and [third-party notices](../THIRD_PARTY_NOTICES.md) for code terms; use the [resource inventory](resource-license-evidence.json) for individual resources.

工作台代码、模型权重和输入资产可能适用不同条款。代码许可不自动提供模型、人体资源、字体或图像的使用/再分发权。

## Included source

- The compatible workbench combination uses GPLv3, preserving existing upstream or-later options. Maintainer-original Kairyth / MKA contributions additionally use Apache-2.0.
- UniRig's ten Michelangelo-derived files retain explicit GPL-3.0-or-later headers. SkinTokens' fifteen subtree files and the directory-external `miche_transformer_blocks.py` are distributed under GPLv3; preserve any existing rights and the guided-diffusion MIT portions. See [exact file evidence](gpl-source-evidence.json).
- Kimodo code retains NVIDIA Apache-2.0 notices. PyTorch3D-derived code retains BSD-3-Clause; other nested code retains its MIT / Apache / GPL terms.
- Distributing compiled versions requires the corresponding source and notices for the actual modified version, including relevant build/install source. A link to unmodified upstream source is not sufficient for modified binaries.

## Models and research resources

| Component | License boundary |
| --- | --- |
| Kimodo SOMA-RP model | The [pinned model license](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1/blob/6c9233af1180b8151e3c4703477104af5dce9dd5/LICENSE) is the NVIDIA Open Model License, not the code's Apache license |
| LLM2Vec adapters and Llama base | McGill's MIT terms apply to its material; the Meta Llama base retains its [own agreement](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2/LICENSE) and access requirements |
| UniRig / SkinTokens weights | Their model-card terms are separate from nested GPL implementation code and from training-data rights |
| MPG / SMPL / SMPL-X | Program, model and derived-data terms are independent. A research-use agreement or top-level MIT package label does not establish permission to redistribute these materials |
| GVHMR / HybrIK | Their source/checkpoint terms and nested body-model terms must be considered separately; implementations and weights are not included here |
| Ultralytics 8.2.42 | [Upstream AGPL terms](https://github.com/ultralytics/ultralytics/blob/v8.2.42/LICENSE) apply to the relevant software; any future integration must satisfy its actual distribution/network-service obligations |
| Modified VRM add-on | The modified archive is not included; its changes, bundled components, license notices and corresponding source need file-specific evidence |
| Fonts and artwork | Geist retains OFL 1.1. Other assets are governed by their individual inventory entries; code licensing does not grant trademark rights |

Process separation and JSON/NPZ exchange do not themselves settle combined-work obligations. Obtain separately licensed software and models through their official sources under the terms applicable to your use.

## Current functionality

HybrIK image inference and GVHMR video inference are experimental and unavailable; compatibility packages are not included. Other editing workflows depend on their own inputs and resources. [Backend interfaces](OPTIONAL_RESEARCH_BACKENDS.md) describe client protocols, while [resource usage](RESEARCH_BACKEND_USAGE.md) identifies the dependencies of supported workflows.
