# Onboarding 文档（2026-08 更新版）

**当一个语言模型最终答错时，正确答案在模型内部的中间计算里到底经历了什么？是从来就没有形成一个有竞争力的候选，还是曾经形成过、但在后续层中被其他信号压制了？** 这篇论文给出了一套校准过的方法来自动分辨这两种情况，并证明这种区分有因果机制基础、可预测、可指导干预。

师兄你上次 onboard 后，论文经历了六轮审稿迭代（Strong Reject 2/10 → Weak Reject 3/10），补了 10+ 个实验、换了主指标、重建了因果证据链。这份文档是当前状态的完整快照。

---

## 一、问题定义（不变）

- **Formation failure（形成失败）**：正确答案 token 在任何中间层都没进入竞争力区域（top-k 且对最终竞争者的 margin 为正）。
- **Preservation failure（保持失败）**：正确答案曾在中间层进入竞争力区域，但到最终层失去优势。

两种错误的输出一样错，但需要完全不同的补救：formation 要增强能力（训练/检索），preservation 只需要轻量的推理时干预（把已有信号保住）。

**严格判据（Eq. 8）**：`∃ 中间层: rank ≤ 5 且 CIS > 0`，且最终层 `CIS < 0`。其中 CIS = log p(gold) − log p(最终层最强竞争者)。

---

## 二、论文现在的完整故事（五幕）

**第一幕：为什么要区分**（Intro）
两种失败需要相反的补救，分不清就浪费预算。

**第二幕：为什么难 / 领域常用的尺子全是坏的**（测量学贡献，三块互相印证的证据）
1. "中间层曾超过错误输出"：92.2% 的错误都满足，但随机对照 98.8%、强度匹配对照 98.7% 也满足 → 几乎任何 token 都会被"穿越"，这个判据无信息量。
2. 多 token 答案的"任一 token 曾进 top-5"（min 规则）：全量 1494 错误上给出 90.6%，但严格的 ALL 规则（每个 token 同层进 top-5）只有 13.2% → min 规则随答案长度虚高。
3. 最极端的例子：长文本任务 QuALITY 在 min 规则下 27.6%（排第 2 高），ALL 规则下 0.5%（垫底）→ 长答案的信号形成是碎片化的，首 token 能形成、全序列几乎从不。

**第三幕：严格测量下的分布**
- 严格 Eq.8：总体 10.0% [8.4, 11.4]，HotpotQA 14.2% → GSM8K 0.0%
- 序列级 ALL：总体 13.2%，HotpotQA 24.2%、TriviaQA 23.0% 最高，GSM8K 4.3%、QuALITY 0.5% 最低
- 结论：preservation 是少数，集中在短答案实体型任务；程序性计算要么对要么从未形成

**第四幕：因果证据链（五层递进，每层堵上一层的漏洞）**
1. 归因：DLA + 组件分解两法一致，末段 MLPs 是压制源，layer 35 是最大单一直接贡献者
2. Peak restoration（α=0.5 时 preservation 独占 5.0% vs 0.0%）：干预中间态有效，与最终 rank 无关
3. Rank-matched（final rank 配平后 7.0×，配对 CI [1.8%, 19.3%]，McNemar 0.070 不称显著）：排除"终点近"的混淆
4. Gold-direction patching（10.9×，Fisher p=0.005；正交化 null 精确归零 0/109，p=0.003）：衰减的 gold 信号是因果的
5. Component-selective patching（单层 MLP 恢复 31.2%，随机正交方向 p<0.001）：压制因果、方向特异、但**分布于 31–35 层 MLPs**（35 与 34 无显著差异）；zero-ablation 35 层反而最伤（该层还承载必要计算）

**第五幕：实用闭环**
- 前半段轨迹预测衰减：线性 0.75（headline）/ RF 0.92（上界）/ endpoint-free 0.66
- 非oracle 触发器（只用模型自己的 top-5 动态，无 gold）：AUC 0.669（pooled；within-task 平均约 0.60，GSM8K 0.82）
- 实际驱动干预：任务内校准触发器 + RepE steering，top 20% 覆盖 → **+67% 效率增益、64% 精度（基线 33%）、省 80% 预算**；oracle 辅助版 +127%
- 发布 InfoDyn-Bench（6 模型 × 7 任务 × 全层轨迹）

---

## 三、关键数字速查表（投稿口径，全过一致性审计）

| 项目 | 数字 | 备注 |
|---|---|---|
| Crossing 伪判据 | 92.2% vs null 98.8%/98.7% | 三方对齐（图/正文/caption 从同一 JSON 生成）|
| 严格 Eq.8 总体 | 10.0% [8.4, 11.4] | 1494 错误，7 任务 |
| ALL 总体 | 13.2%（min 规则上界 90.6%）| Table 3 主指标 |
| 预测 headline | **0.75 ± 0.02**（线性 5-fold CV）| 全文唯一 headline；0.79=单split（CI内）、0.92=RF上界，均标注身份 |
| Logit bonus | 30.0% vs 6.4%（4.7×）| oracle 概念验证 |
| Peak restoration | 2.5×（α=0.8）；α=0.5 独占 | |
| Rank-matched | 7.0×（12.3% vs 1.8%）| 配对统计，不称显著 |
| Activation patching | 10.9×（7.3% vs 0.7%），null 0/109 | p=0.005/0.003，统计上决定性的因果检验 |
| Component patching | 31.2%（γ=4），randorth=0 | 分布式压制 |
| 非 oracle 闭环 | +67% @20% 覆盖，精度 64% vs 33% | 任务内校准 |
| oracle 闭环 | +127%，省 72% | |

---

## 四、六轮审稿演变（简要）

| 轮次 | 分数 | 主要问题 → 处理 |
|---|---|---|
| 1-3 | 3→5→6 | 意识到方法漏洞（selection bias、readout 校准）→ 加 null、tuned lens |
| 4 | 2/10 Strong Reject | 数字不一致（88.2 vs 92 vs 92.2）+ 指标混写 → 全部对齐 |
| 5 | 3/10 Weak Reject | 6 项：ALL 主指标、Fig.2、泄漏、layer-35 因果、配对统计、跨任务严格率 → 全部闭环 |
| 6（当前）| 待审 | 防御性措辞清理、AUC 口径统一、破折号清除、图与正文数字统一 |

审稿人承诺：核心 4 项做完 → borderline/weak accept。目前 4 项已闭环。

---

## 五、代码结构（lostresearch/）

| 脚本 | 实验 | 运行环境 |
|---|---|---|
| `run_full.py` | 轨迹采集（7 任务全量）| 服务器 GPU |
| `run_matched_null.py` | crossing 伪判据的两种 null | 本地 |
| `run_teacher_forced_multitoken.py` | ALL/MEAN/JOINT/MIN 序列级判据（--n 1600 全量）| 服务器 GPU |
| `run_strict_task_rates.py` | 严格 Eq.8 每任务率 + CI | 本地 |
| `run_taxonomy_robustness.py` | k 敏感性 + 答案长度 + 定义一致性 | 本地 |
| `prediction.py` 等 | 轨迹预测（0.75/0.92）| 本地 |
| `run_inference_available_predictor.py` | 非 oracle 预测器（0.669 pooled）| 本地 |
| `run_nonoracle_predictor_v3.py` | 全深度特征版（within-task ~0.60-0.65）| 本地 |
| `run_peak_restoration.py` | peak 态恢复 | 服务器 GPU |
| `run_matched_rank_intervention.py` | rank 匹配 + peak 恢复（7.0×）| 服务器 GPU |
| `run_activation_patching.py` | gold 方向 patch（10.9×）| 服务器 GPU |
| `run_component_patching.py` | 组件级 patch（31.2%，分布式）| 服务器 GPU |
| `run_non_oracle_intervention.py` | oracle 闭环（+127%）| 服务器 GPU |
| `run_nonoracle_trigger_intervention.py` | 非 oracle 闭环（+67%）| 服务器 GPU |

结果文件都在 `outputs/data/*.json`；图脚本在 `iclr2027/figures/gen_fig*.py`（**全部从结果 JSON 自动读数**，不再手工填数字）。

---

## 六、两个 tex 文件的关系

| 文件 | 角色 |
|---|---|
| `iclr2027_conference.tex` | 完整版（18 页），所有内容的底稿，实验记录齐全 |
| `iclr2027/main.tex` | 投稿版：正文 + 附录分割（纯移动，零内容删减），正文目标 9 页，附录承接稳健性检验/旧证据/表格 |

---

## 七、当前待办（按优先级）

1. **页数**：main.tex 编译确认正文页数（预估 11-12 页，超 9 页限制 2-3 页）→ 需要压缩或再挪内容
2. **可复现性**：匿名代码/数据链接（Anonymous GitHub）、随机种子表（都是 42）、Bench 生成脚本链接
3. **跨模型 Eq.8**：严格每任务率目前只有 Qwen3-8B（审稿人要求"按任务、模型"），其他 5 个模型需补跑轨迹或论文明确划界
4. **非 oracle 预测器数字口径**：论文现用 pooled 0.669；内部审计发现 task-grouped 只有 0.46（任务身份泄漏）、within-task 平均 0.60。**这是一个已知的潜在审稿风险点**，待定处理方式（候选：换成 within-task 逐任务报告）

---

## 八、请师兄重点 review 的三个地方

1. **Intervention 章节**（conference 版 §7）：五层因果链是否逻辑递进、有没有过度声明
2. **Table 3 + §3.3 多 token 段**：ALL/min 双指标叙事是否清楚、QuALITY 55 倍落差的解释是否站得住
3. **main.tex 的分割方案**：正文/附录的切分边界是否合理（哪些还该挪）

---

## 当前定位

> 这是一个**方法 + 发现**的工作：校准的测量框架（揭穿三种伪判据）+ formation/preservation taxonomy（严格判据下的分布）+ 可预测性（0.75）+ 五层因果证据链 + 可部署的干预触发器 + InfoDyn-Bench。
