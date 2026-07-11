各位老师好，我的新开源项目已发布：GleamLM——面向教育和研究的小型语言模型。

纯 PyTorch 从零实现，零 HuggingFace 依赖，覆盖四源混合中文数据，第一阶段开放的GleamLM-Nano（40M参数）预训练基座模型PPL 13.65，单卡12GB显存即可成功训练，Windows/Linux 双平台兼容。

项目技术栈对齐Qwen3等现代主流大模型，覆盖LLM全链路：数据管线（下载→清洗→去重→字符加权配比）→ BBPE 分词器训练（自研BBPE，零外部依赖）→ Decoder-only 模型（SwiGLU / GQA / RoPE / QK-Norm）→ AMP + DDP 训练（断点续训保存 optimizer/scheduler/scaler 全量状态）→ SFT 微调/ DPO 对齐（ChatML + loss mask）→模型量化 → KV Cache 流式推理。

项目地址：https://github.com/philexohf/gleamlm，欢迎各位老师试用，有问题直接提 issue。

项目持续开发中， 点个 Star ⭐ 收藏，更新不错过。