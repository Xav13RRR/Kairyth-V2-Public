# Licensing scope / 授权范围

## Workbench code / 工作台代码

The compatible main-program combination is offered under **GNU GPL version 3** ([LICENSE](LICENSE)), to the extent the relevant rights permit that combination. Existing GPL “or any later version” grants remain available. Maintainer-original Kairyth and MKA predecessor contributions are additionally offered under **Apache License 2.0** ([text](licenses/Apache-2.0.txt)).

工作台兼容组合采用 GPLv3，上游已有的 or-later 权利保留。维护者原创的 Kairyth / MKA 前身项目贡献另提供 Apache-2.0 授权；这不把同文件中的第三方内容或整个 GPL 组合作品重新授为 Apache-only。

Original workbench UI, animation/IK/timeline/director logic, orchestration, scripts, tests and documentation fall within that first-party contribution scope. Copied or adapted third-party material retains its own notices and terms, including within files containing original integration work.

## Third-party software / 第三方软件

| Material | Applicable terms |
| --- | --- |
| UniRig Michelangelo-derived files | Existing **GPL-3.0-or-later** headers, authors and notices; the top-level UniRig MIT license does not replace them |
| SkinTokens Michelangelo-derived files, including the directory-external `miche_transformer_blocks.py` | GNU GPL **version 3** is used for this distribution; no unverified later-version option is asserted or removed. Retain the embedded guided-diffusion MIT material and notices |
| Other SkinTokens components | MIT / Apache and their nested-source notices, as identified in the [source inventory](docs/source-provenance-skintokens.json) |
| Kimodo and NVIDIA-noticed rigging code | Apache-2.0 notices; model, text-encoder and asset licenses remain separate |
| PyTorch3D compatibility code | BSD-3-Clause, including Meta copyright and modification scope |
| Three.js and other vendor source | Their existing MIT / BSD / Apache or other upstream terms |
| Geist font | SIL Open Font License 1.1, not the application's GPL / Apache code grant |

See [third-party notices](THIRD_PARTY_NOTICES.md), [file-level GPL evidence](docs/gpl-source-evidence.json) and [source provenance](docs/SOURCE_PROVENANCE.md). Distributions must retain the applicable license texts, attribution and preferred modification source, including relevant build/install source for any compiled distribution.

## Models and assets / 模型与资源

Code licenses do not grant rights to model weights, training data, body models, regressors, skeleton arrays, artwork, fonts or plug-ins. The [resource inventory](docs/resource-license-evidence.json) identifies exact included resources, source references and excluded items.

代码许可证不自动覆盖权重、训练数据、人体模型、回归矩阵、骨架数组、图像、字体或插件。已包含的 SOMA 资源保留 NVIDIA Apache-2.0 版权和来源；项目图像按资源清单中的逐项范围提供，不产生额外商标授权或对其他资产的授权。

GVHMR, HybrIK research implementations, detector code, model weights and restricted MPG / SMPL / SMPL-X materials are not included. GVHMR's research/non-profit terms, HybrIK's nested terms and any separately obtained body-model agreements remain applicable to their respective materials. A package's top-level MIT license is not permission to redistribute independently restricted nested code or models. An external process or JSON/NPZ interface does not by itself remove GPL/AGPL or other combination obligations.

New HybrIK image inference and GVHMR video inference are experimental and unavailable; compatibility packages are not supplied in this version. Existing asset editing is independent of those unavailable entries.

Local uploads, generated records, user data, credentials, environments and caches are not distributed or licensed by this repository. Obtain separately licensed dependencies from their official sources under the terms applicable to your use.
