# Local deployment (Windows)

[中文](DEPLOYMENT.md) | [English](DEPLOYMENT_EN.md) | [README](../README_EN.md)

## 1. Environment and feature dependencies

- Base workbench: Windows 10/11 x64, PowerShell 5.1/7, Python 3.10 x64 and a WebGL2 browser.
- FBX import/export: Blender 4.5.11 LTS; the installer can download the official portable distribution.
- Automatic skeletons/skinning: separate Python 3.11 environments, UniRig / SkinTokens checkpoints and matching CUDA extensions.
- Kimodo text-to-motion: PyTorch, motion model, text encoder and their applicable usage permissions.

New HybrIK image inference and GVHMR video inference are unavailable. Compatibility packages are planned for a future update and are not included in this version.

The base editor starts without every AI environment. GPU memory, drivers and native-extension requirements depend on the chosen models; base-service tests do not establish model compatibility on every hardware configuration. See [resource readiness](RESOURCE_READINESS.md) and [licensing scope](../LICENSE_SCOPE.md).

## 2. Install the base service

Install Git and Python 3.10 x64, then run:

```powershell
git clone https://github.com/Xav13RRR/Kairyth-V2-Public.git
Set-Location .\Kairyth-V2-Public
Set-ExecutionPolicy -Scope Process Bypass
.\install.ps1
```

Base dependencies use `requirements-service-lock.txt`; it is not an AI / CUDA environment lock. The installer creates `runtime/python` when absent, installs base dependencies and downloads Blender when needed. Existing environments are not automatically upgraded; incomplete environments produce an error.

Select Python or initially skip Blender:

```powershell
.\install.ps1 -PythonExe 'C:\Path\To\Python310\python.exe' -SkipBlender
$py = '.\runtime\python\Scripts\python.exe'
& $py -m pip check
.\start_motionbuilder.ps1
```

Open <http://127.0.0.1:7861>. Empty libraries on first launch are expected. If the port is occupied, use `.\start_motionbuilder.ps1 -Port 7862`. Stop with `.\stop_motionbuilder.ps1`. This is a local single-user application; do not expose its port directly to the Internet.

The default Blender path is `runtime/blender/blender-4.5.11-windows-x64/blender.exe`. Alternatively set `MOTIONBUILDER_BLENDER` to a trusted existing executable.

## 3. Kimodo

The following is a matching PyTorch / torchvision example for Python 3.10. Check [PyTorch's version guidance](https://pytorch.org/get-started/previous-versions/) and the NVIDIA driver before using it. Do not blindly replace dependencies in a working environment.

```powershell
$py = '.\runtime\python\Scripts\python.exe'
& $py -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
.\scripts\install_kimodo_dependencies.ps1
& $py -m pip check
& $py -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
.\scripts\verify_kimodo_runtime.ps1
```

Kimodo source is included; do not run `pip install -e` in its source directory without a `pyproject`. The default motion model is `nvidia/Kimodo-SOMA-RP-v1.1`; text encoding also uses Meta Llama 3 and McGill-NLP LLM2Vec. Follow [Kimodo's official instructions](https://github.com/nv-tlabs/kimodo) and review the separate code, motion-model, adapter and base-language-model terms.

Authenticate with the standard client interactively instead of putting a token into scripts or terminal history:

```powershell
$env:HF_HOME = "$PWD\kimodo_runtime\src\.mka_cache\huggingface"
.\runtime\python\Scripts\hf.exe auth login
.\start_motionbuilder.ps1
```

Start from the same terminal so the service inherits cache settings. Caches are not source-distribution inputs; never paste actual tokens into issues. A loader may contact the model service when files are missing. For offline diagnostics, configure offline mode explicitly and prepare all required model files beforehand.

## 4. UniRig and SkinTokens

Install Miniforge / Conda and create separate project-local environments:

```powershell
conda create -y --prefix "$PWD\third_party\envs\unirig" python=3.11 pip
conda create -y --prefix "$PWD\third_party\envs\skintokens" python=3.11 pip
$rig = '.\third_party\envs\unirig\python.exe'
$skin = '.\third_party\envs\skintokens\python.exe'
& $rig -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
& $skin -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
& $rig -m pip install -r third_party/UniRig/requirements.txt
& $skin -m pip install -r third_party/SkinTokens/requirements.txt
```

`spconv`, `torch_scatter`, `torch_cluster` and flash-attn where used need builds matching Windows, Python, PyTorch and CUDA. The commands above do not automatically resolve every native dependency on every machine. Obtain matching upstream wheels or build with MSVC / CUDA Toolkit. Without compatible extensions the affected AI module is unavailable, while the base editor remains usable.

Obtain weights under the instructions and terms from [UniRig](https://github.com/VAST-AI-Research/UniRig) and [SkinTokens](https://github.com/VAST-AI-Research/SkinTokens). SkinTokens provides:

```powershell
Push-Location .\third_party\SkinTokens
& ..\envs\skintokens\python.exe download.py --model
Pop-Location
```

The default binding pipeline uses:

```text
third_party/UniRig/experiments/skeleton/articulation-xl_quantization_256/model.ckpt
third_party/SkinTokens/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt
third_party/SkinTokens/experiments/skin_vae_2_10_32768/last.ckpt
```

It runs UniRig skeleton prediction, SOMA fitting and SkinTokens skinning; the independent UniRig skin checkpoint is not required for this default path. OPT / Qwen configuration identifiers do not imply that their separate base pretrained weights are required. See the [resource-use map](RESEARCH_BACKEND_USAGE.md).

```powershell
& $rig -c "import torch, torch_scatter, torch_cluster, spconv.pytorch; print(torch.__version__, torch.cuda.is_available())"
& $skin -c "import torch, transformers, diffusers; print(torch.__version__, torch.cuda.is_available())"
& $rig -m pip check
& $skin -m pip check
```

Complete a rigging/skinning check with an asset you are permitted to use, storing output under local `data/`.

## 5. Resource checks, data and troubleshooting

```powershell
$py = '.\runtime\python\Scripts\python.exe'
& $py scripts/resource_readiness.py --help
& $py scripts/resource_readiness.py --feature all --presence-only
& $py scripts/resource_readiness.py --feature core --json
```

Resource checks separate file availability, exact identity and license state. They do not download or execute models. Exit code `2` is a missing/mismatch advisory, not a requirement to disable the whole workbench. See [resource readiness](RESOURCE_READINESS.md).

| Symptom | Check |
| --- | --- |
| Workbench online but rigging not ready | The feature's Python environment, native extensions, checkpoint paths and model configuration |
| Hugging Face 401 / 403 | Complete the official model-access steps using your own account, then authenticate through the normal client |
| FBX conversion unavailable | Project-local Blender or `MOTIONBUILDER_BLENDER` |
| Port occupied | Choose a free port; do not stop an unrelated service |
| Experimental inference POST returns HTTP 503 | Compatibility packages are not supplied in this version |

`data/`, `runtime/`, `third_party/envs/` and caches remain local. Back up data independently before upgrades; do not commit them with source or reuse credentials/environments of unknown origin.
