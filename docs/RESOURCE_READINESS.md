# Resource readiness / 资源就绪

The base workbench can start without AI models. Missing resources affect the feature that uses them, not every page. HybrIK image inference and GVHMR video inference are experimental and unavailable; their compatibility packages are not included.

基础工作台可在未安装 AI 模型时启动。缺资源只影响依赖它的功能；HybrIK 图片新推理、GVHMR 视频新推理尚未开放，本版不含兼容包。

## Offline checks / 离线检查

Run from the project directory with the installed service Python:

```powershell
$py = '.\runtime\python\Scripts\python.exe'
& $py scripts/resource_readiness.py --help
& $py scripts/resource_readiness.py --feature all --presence-only
& $py scripts/resource_readiness.py --feature core --json
& $py scripts/resource_readiness.py --feature skintokens
```

The checker reads the [resource inventory](resource-license-evidence.json). It does not download resources, import model packages, unpickle weights, start a service or accept license terms. Exit code `2` is a missing/mismatch advisory, not a reason to disable the whole UI.

`--presence-only` does not read file contents. Full checks compare SHA256 where an expected hash is available; files without an expected hash are `unversioned`. Startup/install advisories use presence-only checks and omit the two unavailable inference groups. Manual `--feature all` remains a file check, not a feature-enablement command.

| State / 状态 | Meaning / 含义 |
| --- | --- |
| `physical=present/missing` | File availability / 文件是否存在 |
| `integrity=match/mismatch/unversioned/not-checked` | Exact identity versus the inventory, not model quality / 与清单比较文件身份，不是模型效果 |
| `rights` / `publication` | Separate license and distribution classification / 独立许可与分发分类；文件存在或匹配不授予权利 |

`physical_ready` / `integrity_ready` cover selected inventory entries, not installed CUDA extensions, every model cache or the quality of a generated result.

## Feature dependencies / 功能资源

| Feature | Required resources | Source and terms |
| --- | --- | --- |
| Skeleton editing, fitting and previews | Included `motionbuilder_app/assets/skeletons/somaskel30/joints.npy` | Float32 conversion of the [pinned official SOMA joint data](https://github.com/nv-tlabs/kimodo/tree/e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954/kimodo/assets/skeletons), retaining NVIDIA Apache-2.0 notices |
| Kimodo generation | Motion model, `config.yaml`, six motion statistics files, text encoder and SOMA resources | [Kimodo model snapshot](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1/tree/6c9233af1180b8151e3c4703477104af5dce9dd5); model terms differ from code Apache-2.0 |
| Kimodo text encoding | LLM2Vec MNTP / supervised adapters and their Llama base | Adapter MIT does not replace the Llama base agreement or official access requirements |
| New skeleton proposals | UniRig skeleton checkpoint and compatible native extensions | [UniRig model snapshot](https://huggingface.co/VAST-AI/UniRig/tree/36842e2b5947e9e60f89275b83208c8e74071c63), with model terms separate from nested GPL code |
| Skin-weight prediction | Both SkinTokens TokenRig and VAE checkpoints; matching Qwen configuration | [SkinTokens model snapshot](https://huggingface.co/VAST-AI/SkinTokens/tree/79736cad0fd84de384d5eede659b4ebd24effe33); both checkpoints are needed |
| FBX conversion | Official Blender executable | Separate Blender distribution and its license notices |
| Web typography | Included Geist Sans and configured fallback fonts | [Geist source and OFL evidence](font-provenance-evidence.json); Oswald is not bundled |

The detailed [resource-use map](RESEARCH_BACKEND_USAGE.md) distinguishes generation from existing-asset editing and identifies optional upstream paths. Restricted body-model resources, model weights and the modified VRM add-on are not bundled. The full inventory also records excluded resource identities; their presence in that inventory is not an installation requirement for this release.

## Manual paths / 手动路径

For separately licensed skeleton resources, `KAIRYTH_SKELETONS_ROOT` points to the directory containing `somaskel30/joints.npy`:

```powershell
$env:KAIRYTH_SKELETONS_ROOT = 'D:\MyAssets\skeletons'
& $py scripts/resource_readiness.py --feature core
```

For check-only mappings, provide a JSON file mapping inventory paths to files:

```json
{
  "motionbuilder_app/assets/skeletons/somaskel30/joints.npy": "D:\\MyAssets\\skeletons\\somaskel30\\joints.npy"
}
```

```powershell
& $py scripts/resource_readiness.py --feature core --manual-map 'D:\MyAssets\paths.json'
```

Manual mappings change only the check target. They do not install resources, configure model loaders, enable unavailable features or change applicable licenses. See [license details](LICENSE_DECISIONS.md).
