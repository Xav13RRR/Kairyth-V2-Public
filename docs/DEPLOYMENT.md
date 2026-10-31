# 本地部署（Windows）

[中文](DEPLOYMENT.md) | [English](DEPLOYMENT_EN.md) | [返回首页](../README.md)

## 1. 环境与功能依赖

- 基础工作台：Windows 10/11 x64、PowerShell 5.1/7、Python 3.10 x64、WebGL2 浏览器。
- FBX 导入/导出：Blender 4.5.11 LTS；安装器可下载官方便携版。
- 自动骨架与蒙皮：独立 Python 3.11 环境、UniRig / SkinTokens 权重及匹配的 CUDA 扩展。
- Kimodo 文本动作生成：PyTorch、动作模型、文本编码器及各自使用许可。

HybrIK 图片新推理与 GVHMR 视频新推理尚未开放，兼容包计划后续提供，本版不包含这些兼容包。

基础编辑界面无需所有 AI 环境即可启动。GPU 内存、驱动和原生扩展需求由所选模型决定；基础服务测试不代表每个硬件组合都能运行模型。模型与资源条款见 [资源指南](RESOURCE_READINESS.md) 和 [授权范围](../LICENSE_SCOPE.md)。

## 2. 安装基础服务

安装 Git 和 Python 3.10 x64，然后执行：

```powershell
git clone https://github.com/Xav13RRR/Kairyth-V2-Public.git
Set-Location .\Kairyth-V2-Public
Set-ExecutionPolicy -Scope Process Bypass
.\install.ps1
```

基础服务依赖使用 `requirements-service-lock.txt` 中的约束；它不是 AI / CUDA 环境锁文件。安装器在缺失时创建 `runtime/python`，安装基础依赖，并在需要时下载 Blender。已有环境不会自动升级；不完整的环境会报错。

指定 Python 或先跳过 Blender：

```powershell
.\install.ps1 -PythonExe 'C:\Path\To\Python310\python.exe' -SkipBlender
$py = '.\runtime\python\Scripts\python.exe'
& $py -m pip check
.\start_motionbuilder.ps1
```

默认打开 <http://127.0.0.1:7861>；首次启动资产库为空。端口被占用时可用 `.\start_motionbuilder.ps1 -Port 7862`。关闭运行 `.\stop_motionbuilder.ps1`。服务仅供本机单用户使用，不应直接暴露到公网。

Blender 默认位置是 `runtime/blender/blender-4.5.11-windows-x64/blender.exe`；也可将 `MOTIONBUILDER_BLENDER` 设为可信的现有可执行文件。

## 3. Kimodo

下面是 Python 3.10 的匹配 PyTorch / torchvision 安装示例；使用前核对 [PyTorch 版本说明](https://pytorch.org/get-started/previous-versions/) 与 NVIDIA 驱动。不要在可用环境中盲目覆盖依赖。

```powershell
$py = '.\runtime\python\Scripts\python.exe'
& $py -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
.\scripts\install_kimodo_dependencies.ps1
& $py -m pip check
& $py -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
.\scripts\verify_kimodo_runtime.ps1
```

Kimodo 源码已包含，不要对缺少 `pyproject` 的源码目录执行 `pip install -e`。默认动作模型为 `nvidia/Kimodo-SOMA-RP-v1.1`；文本编码器还依赖 Meta Llama 3 和 McGill-NLP LLM2Vec。按 [Kimodo 官方说明](https://github.com/nv-tlabs/kimodo) 获取所需模型，并分别阅读代码、动作模型、adapter 和基础语言模型条款。

使用标准客户端交互登录，不把令牌写进脚本或终端历史：

```powershell
$env:HF_HOME = "$PWD\kimodo_runtime\src\.mka_cache\huggingface"
.\runtime\python\Scripts\hf.exe auth login
.\start_motionbuilder.ps1
```

从同一终端启动，使服务继承缓存配置。缓存目录不进入源码分发；不要把真实令牌贴入 issue。模型缺失时加载器可能访问模型服务，离线诊断应明确设置离线环境并提前准备完整模型文件。

## 4. UniRig 与 SkinTokens

安装 Miniforge / Conda，并为两个后端创建独立项目环境：

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

`spconv`、`torch_scatter`、`torch_cluster` 及部分路径使用的 flash-attn 需要匹配 Windows、Python、PyTorch 和 CUDA 的构建；上述命令不能保证在所有硬件上自动解决这些原生依赖。按上游说明取得匹配 wheel 或使用 MSVC / CUDA Toolkit 编译。没有兼容扩展时，该 AI 模块不可用，基础编辑仍可使用。

按 [UniRig](https://github.com/VAST-AI-Research/UniRig) 与 [SkinTokens](https://github.com/VAST-AI-Research/SkinTokens) 的模型说明及许可取得权重。SkinTokens 提供：

```powershell
Push-Location .\third_party\SkinTokens
& ..\envs\skintokens\python.exe download.py --model
Pop-Location
```

默认绑定链路使用以下权重：

```text
third_party/UniRig/experiments/skeleton/articulation-xl_quantization_256/model.ckpt
third_party/SkinTokens/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt
third_party/SkinTokens/experiments/skin_vae_2_10_32768/last.ckpt
```

应用默认使用 UniRig 预测骨架、SOMA 拟合、SkinTokens 蒙皮，不要求下载 UniRig 的独立 skin checkpoint。OPT / Qwen 的配置标识也不代表需要另外下载其基础预训练权重。详情见 [后端资源用途](RESEARCH_BACKEND_USAGE.md)。

```powershell
& $rig -c "import torch, torch_scatter, torch_cluster, spconv.pytorch; print(torch.__version__, torch.cuda.is_available())"
& $skin -c "import torch, transformers, diffusers; print(torch.__version__, torch.cuda.is_available())"
& $rig -m pip check
& $skin -m pip check
```

用有权使用的模型完成一次绑定/蒙皮检查，输出保存到本地 `data/`。

## 5. 资源检查、数据与排错

```powershell
$py = '.\runtime\python\Scripts\python.exe'
& $py scripts/resource_readiness.py --help
& $py scripts/resource_readiness.py --feature all --presence-only
& $py scripts/resource_readiness.py --feature core --json
```

资源检查区分文件存在、精确身份与许可状态，不会下载或执行模型。退出码 `2` 表示缺失/不匹配提示，不要求关闭整个工作台。详见 [资源就绪](RESOURCE_READINESS.md)。

| 现象 | 检查方向 |
| --- | --- |
| 工作台在线但绑定未就绪 | 对应 Python 环境、原生扩展、checkpoint 路径与模型配置 |
| Hugging Face 401 / 403 | 使用自己的账户完成官方模型访问步骤，然后正常登录客户端 |
| FBX 转换不可用 | 项目内 Blender 或 `MOTIONBUILDER_BLENDER` 路径 |
| 端口被占用 | 换用空闲端口；不要关闭不属于当前实例的服务 |
| 两项实验性推理返回 HTTP 503 | 本版尚未提供兼容包，属于当前功能状态 |

`data/`、`runtime/`、`third_party/envs/` 和缓存只保存在本地。升级前独立备份数据，不将其与源码一起提交；不要复用不明来源的凭据或环境。
