# Security and local data / 安全与本地数据

Kairyth is a local, single-user workbench bound to 127.0.0.1. It is not a hardened public hosting service. Import only assets you trust; some scientific/animation formats may contain pickled data. Do not expose the API or encoder ports to the Internet.

Kairyth 面向本机单用户使用，监听 127.0.0.1，不是经加固的公网托管服务。只导入可信资产，部分科学计算/动画格式可能包含 pickle 数据，不要将 API 或编码器端口直接暴露到互联网。

## Credentials / 凭据

Use interactive Hugging Face login or process environment variables. Never commit tokens, private keys, credentials files, logs containing secrets, screenshots showing credentials, or user uploads. Do not attach WSL disks or runtime directories to issues.

使用交互式 Hugging Face 登录或进程环境变量。不要提交令牌、私钥、凭据文件、含密钥日志、凭据截图及用户上传内容，也不要把 WSL 磁盘或运行环境附在 issue 中。

If a credential reaches Git history, revoke/rotate it at the provider before handling repository history. Deleting it in a later commit does not revoke the credential. Contact the maintainer through an agreed private channel; do not post actual secrets in an issue, even in a private repository.

若凭据进入 Git 历史，应先在服务提供方撤销/轮换，再处理仓库历史；后续提交删除文件不会撤销凭据。通过双方约定的私密渠道联系维护者，即使仓库私有也不要在 issue 中粘贴实际密钥。

## Publication checks / 发布检查

Run `python scripts/check_publication.py --history` from a proper checkout after staging the intended files. The script checks Git's index, rejects private/generated paths and notebook outputs, and optionally scans reachable history for known credential formats and suspicious assignments. It prints locations only.

在正常 checkout 暂存预期文件后执行上述命令。检查针对 Git 暂存区，拒绝私有/生成路径及 notebook 输出，可选扫描可见历史中的已知凭据格式与可疑赋值，仅报告位置。

The checks do not validate token activity, scan arbitrary binary containers or prove absence of all secrets. Git LFS data and model assets require separate review. Configure GitHub secret scanning/push protection where available; this repository does not assume those account-level features are enabled.

检查不会验证令牌有效性，不覆盖任意二进制容器，也不能证明不存在所有密钥。Git LFS 实体和模型资源需要单独审查。可用时应配置 GitHub 密钥扫描/推送保护，本仓库不假设账户已启用这些功能。

## Data retention / 数据保留

`data/` and runtime/cache directories are local-only and should not be committed. Back up local data before source updates or cleanup. Synthetic test fixtures must not be copied from actual generation or upload histories.

`data/` 及运行时/缓存目录仅保存在本地，不应提交到源码仓库。更新源码或清理前应独立备份本地数据；合成测试夹具不得从真实生成/上传记录复制。
