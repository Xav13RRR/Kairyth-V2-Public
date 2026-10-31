# PyTorch3D compatibility shim / 兼容代码来源

This directory provides selected compatibility interfaces, not a complete PyTorch3D distribution.

`transforms/rotation_conversions.py` contains adapted PyTorch3D rotation conversions. Comparison reference: [PyTorch3D v0.7.7, commit `89653419d0973396f3eff1a381ba09a07fffc2ed`](https://github.com/facebookresearch/pytorch3d/blob/89653419d0973396f3eff1a381ba09a07fffc2ed/pytorch3d/transforms/rotation_conversions.py). The reference identifies corresponding implementations, not a verified original acquisition revision.

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

BSD-3-Clause terms are preserved in [LICENSE](LICENSE). Adaptations select/rework rotation helpers and expose only the interfaces needed by the integration; the full upstream API is not provided.

本目录提供有限兼容接口。旋转转换实现改编自 PyTorch3D，保留 Meta 版权与 BSD-3-Clause 条款；应用代码许可证不替代这些权利。上述固定版本为来源比对参考，改编范围见 [源码来源](../docs/SOURCE_PROVENANCE.md)。
