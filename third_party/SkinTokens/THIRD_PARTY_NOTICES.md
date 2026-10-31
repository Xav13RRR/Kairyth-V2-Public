# SkinTokens nested-source notices / 嵌套来源说明

This supplemental notice identifies nested sources in SkinTokens snapshot `273b691d35989d71cd17ff2895fdc735097b92d1`. It does not replace [SkinTokens' MIT license](LICENSE) or existing rights. The compatible Kairyth combination uses GPLv3; see the adjacent [GPL source notice](GPL_SOURCE_NOTICE.md) for version selection and corresponding-source requirements.

本说明记录 SkinTokens 的嵌套来源，顶层 MIT 不替代各文件中的 GPL / Apache / MIT 条款。

| Material / 内容 | Sources / 来源 | License texts / 许可原文 |
| --- | --- | --- |
| `src/model/michelangelo/**` and derived portions of `src/model/skin_vae/autoencoders/miche_transformer_blocks.py` | NeuralCarver/Michelangelo, GPL version 3 used for this distribution; exact 15+1 inventory in root evidence | [GPLv3](../../licenses/Michelangelo-GPL-3.0.txt), [source notice](GPL_SOURCE_NOTICE.md) |
| Checkpoint implementation in `miche_transformer_blocks.py` | OpenAI guided-diffusion | [MIT](../../licenses/GuidedDiffusion-MIT.txt) |
| `skin_vae/autoencoders/FSQ.py` | Phil Wang / vector-quantize-pytorch | [MIT](../../licenses/VectorQuantize-MIT.txt) |
| `skin_vae/autoencoders/SimVQ.py` | youngsheen/SimVQ and earlier CompVis/taming-transformers | [SimVQ MIT](../../licenses/SimVQ-MIT.txt), [taming MIT](../../licenses/TamingTransformers-MIT.txt) |
| Tripo2 attention, embeddings, VAE, modeling outputs and partial encoder implementation | VAST-AI-Research/TripoSG, with earlier diffusers/Hunyuan implementations where identified | [TripoSG MIT](../../licenses/TripoSG-MIT.txt); existing Apache notices retained |
| `skin_vae/transformers/tripo2_transformer.py` and copied diffusers methods | HunyuanDiT Authors, Qixun Wang, HuggingFace | [Apache-2.0](../../licenses/Diffusers-Apache-2.0.txt) |
| FSQ-CVAE backbone attribution | Biao Zhang / 3DShape2VecSet | [MIT](../../licenses/3DShape2VecSet-MIT.txt), module-level attribution |

The 63 Python files comprise 59 normalized exact upstream matches, two modifications and two maintainer-original integration additions. See [source provenance](../../docs/SOURCE_PROVENANCE.md) and [file hashes and fixed references](../../docs/source-provenance-skintokens.json). Header omissions in the fixed SkinTokens source do not remove the applicable upstream licenses.

When repackaging this backend, include the relevant license texts, notices, modifications and corresponding build/install source; do not leave notice links pointing outside the distributed package. Model and asset terms remain separate from source-code licensing.
