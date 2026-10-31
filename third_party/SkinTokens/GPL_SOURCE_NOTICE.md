# GPL source-family notice / GPL 来源声明

Kairyth attribution notice, added 2026-10-31. This supplements the package MIT license and preserves existing GPL / MIT rights.

The 15 Python files under `src/model/michelangelo/` and `src/model/skin_vae/autoencoders/miche_transformer_blocks.py` are enumerated with hashes and fixed references in [the file-level evidence](../../docs/gpl-source-evidence.json). They derive from NeuralCarver/Michelangelo snapshot `6d83b0bacef92715dd5179d45647ed9a3d39bc95`, via SkinTokens `273b691d35989d71cd17ff2895fdc735097b92d1`.

**Version used for this distribution: GNU GPL version 3.** The fixed Michelangelo repository supplies GPLv3. These SkinTokens files do not establish a per-file later-version option, so this distribution complies using version 3 without asserting or removing any existing later-version right. UniRig's explicit GPL-3.0-or-later headers are a separate source set. Empty initialization files are included in the family inventory without a separate claim of protected authorship.

Full GPLv3 text: [licenses/Michelangelo-GPL-3.0.txt](../../licenses/Michelangelo-GPL-3.0.txt). Attribution: NeuralCarver/Michelangelo original authors; VAST-AI-Research and contributors for their adaptations. The checkpoint implementation also derives from OpenAI guided-diffusion; retain its [MIT notice](../../licenses/GuidedDiffusion-MIT.txt). A GPL-compatible MIT input retains its notice without making the combined derived file MIT-only.

SkinTokens adaptations include architecture, attention / flash-attention and VAE integration. Kairyth's compatibility-helper changes and NPZ bridge are described in [source provenance](../../docs/SOURCE_PROVENANCE.md) and the [63-file inventory](../../docs/source-provenance-skintokens.json).

Distribute the preferred-form Python source, configurations, relevant bridge/build/install source, full license texts and notices together. A compiled distribution requires the corresponding source of the actual version, including modifications and build/install scripts; an upstream URL is not a substitute. Model weights are not included or licensed by this notice.

中文：本分发按 GPLv3 提供，保留已有权利，不以顶层 MIT 覆盖派生 GPL。目录外的 `miche_transformer_blocks.py` 同属来源范围；guided-diffusion 的 MIT 声明继续保留。重新打包时须携带适用的完整源码与许可，模型权重另有条款。
