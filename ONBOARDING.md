**当一个语言模型最终答错时，正确答案在模型内部的中间计算里到底经历了什么？是从来就没有形成一个有竞争力的候选，还是曾经形成过、但在后续层中被其他信号压制了？** 这篇论文给出了一套校准过的方法来自动分辨这两种情况。

---

## 问题定义

我们研究两类本质不同的错误：

- **Formation failure（形成失败）**：正确答案 token 在模型的任何中间层都没有成为有竞争力的候选（即没有进入全词汇表的 top-k，且对竞争者的 margin 非正）。换句话说，模型内部从头到尾都没有形成正确答案这个候选。

- **Preservation failure（保持失败）**：正确答案曾在某些中间层进入有竞争力的区域（进入 top-k 且 margin 为正），但到最终层失去了这个优势。换句话说，正确答案曾经形成过，但在后续计算中被压制了。

这两种错误的最终输出都是错的，但内部机制完全不同，因此需要不同的处理策略。这也是我们做这个区分的原因。

---

## 方法

模型是逐层计算的。我们对每个样本做一件事：**在每一层，用解码器（logit lens / tuned lens / cosine）读中间表示，得到正确答案 token 的概率、在全词汇表中的 rank、以及它和竞争者的 margin**。把这些逐层串起来，就得到一条"正确答案的信息轨迹"。

根据这条轨迹是否"曾经进入竞争区域，但最终层失败"，我们把错误分为 formation failure 和 preservation failure。

---

## 核心发现

**1. 过去常用的判断方法（"正确答案是否曾超过错误答案"）是不可靠的。**

我们构造了一个 random-competitor null：把真正的竞争 token 替换成随机 token 后，正确答案"在某层超过对手"的比例高达 99%（高于实际观察到的 92%）。这说明"曾经超过"几乎是一个必然事件，任何 token 都可能在某层被正确答案暂时超过。因此这个判据不能用来判断正确答案是否曾经真正有竞争力。这也是为什么需要更严格的、基于全词汇表 rank 的判据。

**2. 绝大多数错误是 formation failure，preservation failure 是少数。**

用严格判据（rank 进入 top-k 且 margin 为正）测量，发现 preservation failure 大约只占错误的 8-9%，且集中在知识型问答任务（TriviaQA、HotpotQA）；在数学推理（GSM8K）上，当前的单 token 测量方法检测不到（0%），这是一个测量边界问题，不代表数学推理中不存在这种现象。

**3. 后续是否会衰减是可以提前预测的，且预测能力可迁移。**

只用轨迹的前半段特征，就能预测后半段是否发生衰减。线性分类器达到约 0.75 的 AUC（非线性随机森林达 0.92），并且这个预测信号能跨任务、跨模型迁移。

**4. 这个分类有实际用处（干预相关性）。**

我们做了校准的 steering 实验：给正确答案 token 的 logit 加一个小的固定 bonus。结果：preservation failure 的恢复率约 30%，而 formation failure 只有约 6.4%（差异约 4-5 倍）。这说明"知道但丢了"的错误确实更容易被挽救，而"根本没形成"的错误无法通过这种干预修复。

---

## 每个章节在做什么

| 章节 | 内容 |
|---|---|
| **1. Introduction** | 动机：同样答错但内部机制可能不同；旧判据有问题；提出校准的轨迹框架。 |
| **2. Related Work** | 相关工作（解码中间表示、知识定位、hallucination 预测、机制可解释性），并明确与最近工作（Jiang et al. 2024 等）的差异。 |
| **3. Problem Definition** | 形式化定义 formation/preservation failure、rank、两种 margin、信息轨迹、竞争衰减标签。 |
| **4. Experimental Setup** | 6 个模型（Qwen3-4B/8B/14B、Qwen2.5-7B、Llama-3.1-8B、Mistral-7B）× 3 个任务（TriviaQA、HotpotQA、GSM8K），lens 构造与统计方法。 |
| **5. Results** | 核心结果：旧判据无效、两种错误的轨迹不同、preservation 是少数、与任务相关、对阈值稳健。 |
| **6. Trajectory Prediction** | 前半段轨迹预测后半段衰减；与单层快照、最终层置信度、时序外推等基线对比；跨任务/跨模型迁移。 |
| **7. Behavioral Intervention & Component Analysis** | gold-direction ablation、norm-matched 对照、直接 logit 归因（定位到 layer-35 MLP）、校准 steering 验证分类的干预相关性。 |
| **8. InfoDyn-Bench** | 把所有轨迹、派生指标、失败标签、预测分割打包成公开资源（6 模型 × 3 任务）。 |
| **9. Discussion & Limitations** | 明确我们测的是"可解码性"而非"知识本身"；steering 实验目前是 oracle proof-of-concept；单 token 追踪的边界。 |
| **10. Conclusion** | 总结。 |

---

## 目前实验的完整性

✅ 已有且较扎实：
- 旧判据无效性（random-competitor null）
- Correct vs incorrect 轨迹分离（raw/tuned/cosine 三种解码器一致）
- 竞争相变分析 + 阈值稳健性
- 轨迹预测（线性 0.75 / RF 0.92 / endpoint-free 0.66）+ 时序基线
- 跨任务、跨模型迁移
- Gold-direction ablation + norm-matched 对照
- 组件归因（layer-35 MLP 是最大负贡献）
- Calibrated steering（preservation 30% vs formation 6.4%）

---


---

## 当前定位

> 这是一个**方法 + 发现**的工作：校准的测量框架 + formation/preservation taxonomy + 轨迹预测 + 干预相关性 + benchmark。
