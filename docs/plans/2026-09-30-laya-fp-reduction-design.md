# Laya v2 增训：难负样本驱动的误报率压降 — 设计文档

**日期**：2026-09-30
**作者**：Sisyphus
**状态**：已批准
**依据**：`research/dataset_research.md`（2026-09-30 数据集洞察报告）

---

## 1. 目标

当前 v1 增训模型误报率偏高（600 holdout 误报 4/231=1.7%，且该 holdout 负样本
本身是 FGRC 提示句式，真实误报率被低估）。依据数据集洞察报告，完成：

1. **下载报告涉及的完整数据集**（CCL2023-FCC、ChiFraud、TeleAntiFraud-28k、Phishing Email）
2. **数据清洗 + 标注**（含 FBS_SMS 去空格修复、FGRC 标签噪声审计、评测字段口径统一）
3. **构造八体裁难负样本**（报告核心论断：误报率唯一有效杠杆是难负样本质量）
4. **14 类体系增训**（新增 rebate_scam 刷单返利 —— 发案量第一、当前缺位的类别）
5. **训练 v2 并四套评测验证**

**验收标准**（硬性，`check_acceptance.py` 扩展为 4 门槛）：

| # | 门槛 | 目标 |
|---|---|---|
| 1 | **难负评测集 FPR**（主目标） | **≤ 2%** |
| 2 | 中文 is_scam acc / recall（600 中文 holdout，绝对门槛） | ≥ 0.90 / ≥ 0.95 |
| 3 | 14 类 accuracy（手写集 38，固定可比） | ≥ 0.70 |
| 4 | 刷单返利类别召回（CCL 真实数据支撑） | ≥ 0.80 |

---

## 2. 用户决策摘要（来自 brainstorming）

| 维度 | 选择 |
|---|---|
| 商用约束 | **仅个人研究，不商用** → CCL2023-FCC（禁商用条款）与 ChiFraud（无 LICENSE）可用 |
| 难负样本构造 | **抓取 + 多模型生成混合**（arkcli 调火山方舟 ≥3 模型家族分散风格） |
| 类别体系 | **13 → 14 类最小增量**（新增 rebate_scam），二元 is_scam 仍为主判定 |
| FPR 目标 | **激进档 ≤2%** → 设计必须内置挖掘-复核-增训迭代闭环 |
| 总体方案 | **方案 A 两阶段闭环**：数据扩充+训练 v2 → 若 FPR>2% 滚动挖掘迭代（预算 2 轮） |
| 训练平台 | M4 Pro MPS + LoRA（沿用现有管线，`train_local_lora.py` 不动） |

---

## 3. 端到端流程

```
Phase A: 数据获取与构造
  scripts/fetch_datasets.py (扩展)  → 下载 CCL/ChiFraud/TeleAntiFraud/PhishingEmail
  scripts/scrape_antifraud.py       → 抓取反诈宣传/政务通知真实文本
  scripts/generate_negatives.py     → arkcli 多模型生成（体裁2-8 + 简单负样本 + 对照对）
  scripts/audit_fgrc_labels.py      → v1 模型标签噪声审计（抽查分歧）
  scripts/build_hardneg_evalset.py → 先切出难负评测集（id 不相交断言）

Phase B: 数据集组装
  scripts/build_dataset.py (扩展)   → 新 loader + 14类映射 + 配比上限 + 去重
                                      → datasets/training/{train,val,test}.jsonl v2

Phase C: 训练与导出（现有脚本不动）
  scripts/train_local_lora.py       → M4 Pro MPS, LoRA r=8, 交叉熵双任务
  scripts/export_local_onnx.py      → models/laya-onnx-multilingual-finetuned-v2/

Phase D: 四套评测 + 验收
  难负评测集(FPR主门槛) / 600中文holdout / 手写集38 / public.jsonl 400
  scripts/check_acceptance.py       → 4 门槛判定

Phase E: 挖掘闭环（FPR>2% 时触发，硬上限 2 轮）
  scripts/mine_hard_negatives.py    → v2 模型给万级候选池打分
                                    → 高分 benign（模型误报样本）人工复核清单
                                    → 剔除假负例后增补 → 重训 → 重测
```

---

## 4. 数据获取与清洗

### 4.1 新增下载（扩展 fetch_datasets.py，沿用 curl/git clone 绕代理）

| 数据集 | 获取方式 | 用途 | 预期规模 |
|---|---|---|---|
| CCL2023-FCC | GitHub clone（GJSeason/CCL2023-FCC） | 正样本主力：刷单返利 35,459 + 12 类受害人笔录 | ~102,762 |
| ChiFraud | GitHub clone（xuemingxxx/ChiFraud） | 诈骗侧仅「地下贷款」→ loan_scam；正常网页文本作 benign 补充（限额） | 411,434 取子集 |
| TeleAntiFraud-28k | HF curl（Apache-2.0） | ASR 转写对话正样本（多轮体裁）+ 正常通话 benign | ~28,511 |
| Phishing Email | kagglehub（凭据缺失优雅跳过） | 英文 benign 补充（39,595 合法邮件） | 可选 |

### 4.2 清洗规则

- **FGRC-SCD**：保留现有映射；实现期用 v1 模型跑一遍训练样本，抽查 scam↔benign
  分歧做人工确认（报告已证实细类标签错配：顺丰赠品被标投资理财；二元用途影响可控）
- **FBS_SMS**：去空格修复（逐字加空格让中文分词失效）；占位符（URL/DIGIT/PLACE/
  CELLPHONE）不可逆向，保持原样并记录
- **CCL2023-FCC**：受害人笔录 ≠ 嫌疑人话术（体裁不同源）。同案多份笔录去重 +
  1024 字符截断 + **非刷单返利类正样本占比上限 ≤20%** 防笔录文体主导
  （**勘误**：原「CCL 笔录 ≤20%」与 §7.1「刷单返利 ≥25% 正样本」矛盾——
  rebate_scam 同样来自 CCL 笔录。修正为 rebate 豁免、其余 11 类受限；
  体裁风险由 Gate 4 刷单返利召回 + 分体裁 FPR 监控兜底）
- **TeleAntiFraud**：ASR 转写无标点，保留原样（真实部署输入也来自 ASR）
- **ChiFraud**：正常网页文本 benign 占比上限 ≤10%（文体偏移控制，2022-06~2023-06
  时间漂移记录在案）
- **全源**：跨源 sha1 去重（扩展现有机制）
- **字段口径统一**：`public.jsonl` 补 category、`eval.jsonl` 的 expected_category
  统一为 `category`（报告 §6 指出的口径不一致）

---

## 5. 类别体系：13 → 14 类

- `schemas/scam_categories.py` 新增 `rebate_scam`（刷单返利），插入位置紧邻 job_scam
- 新增映射表：CCL 12 类 → 14 类。确定的锚点：刷单返利类→rebate_scam、
  贷款代办信用卡类→loan_scam、虚假征信类→loan_scam、网络赌博类→spam_general、
  网黑案件→spam_general；其余类别（冒充客服/公检法/领导熟人/投资理财/购物服务/
  婚恋交友/机票退改签）在实现期随 loader 单元测试逐一确定，**规则：12 类全覆盖、
  无 default 落空**（落入哪个类都必须显式有依据）
- 新增映射表：ChiFraud 11 类（灰产供给侧视角，仅地下贷款可映射 loan_scam，其余弃用）
- 新增映射表：TeleAntiFraud 标签 → 14 类
- **二元 is_scam 仍是主判定指标**；低样本类（crypto 15 条/marketing 15 条）保持
  现状（synthetic 种子只进 train），允许其召回不达标（报告：低样本类不承担召回指标）

---

## 6. 难负样本构造（误报率压降的核心）

### 6.1 八体裁基础集（总量 ~10-12k，全部 is_scam=0，category 一律 benign，
体裁信息记在 source 字段供分体裁 FPR 统计）

| # | 体裁 | 方式 | 目标 | 要点 |
|---|---|---|---|---|
| 1 | 反诈宣传/预警劝阻 | **抓取** | 800-1000 | 公安/政府反诈语录、十个凡是、12381/96110 标准话术；含全套诈骗关键词却恰是官方内容，纯关键词检测器必然误报 |
| 2 | 真实金融通知 | 生成 | 1,000-1,500 | 验证码/额度/账单/积分，银行券商保险支付署名 |
| 3 | 电商物流通知 | 生成 | 1,000-1,500 | 订单异常/退货/发货/地址变更 |
| 4 | 招聘兼职广告 | 混合 | 1,000-1,500 | FGRC 无风险类自然样本 + 生成补充 |
| 5 | 商业促销 | 抓取+生成 | 1,000-1,500 | 满减/补贴/积分兑换（与刷单返利获利话术共现） |
| 6 | 政务公共通知 | 种子+生成 | 1,000-1,500 | FBS Other 101 条种子，社保/医保/公积金/ETC/家长群 |
| 7 | 个人社交往来 | 生成 | 1,000-1,500 | 家人/拼团/红包/代付 —— 含转账意图但无诈骗结构 |
| 8 | 引流非诈骗 | 生成 | 1,000-1,500 | 交友/直播/内容引流（平台引流占 66%，本身合法） |

> 基础集目标量级与报告 §五配比表（每体裁 ≥1,500）对齐：45k 训练集下负样本
> ~25k，难负 60-70% 需 15-17k，基础 10-12k + 一轮挖掘增量可达。若 v2 首测
> 即达 FPR≤2%，挖掘轮可不跑 —— 此时难负配比自然落在 40-50%，属可接受偏差
> （难负质量优先于配比数字）。

### 6.2 生成协议

- **多模型分散**：arkcli 调 ≥3 个不同家族模型轮转生成，`source` 字段记录生成器
  （防止模型学到「像某个模型写的」而非「像正常短信」）
- **同构对照对**（ScamShield 先例）：金融通知 vs 冒充客服、物流通知 vs
  delivery_fraud、招聘 vs job_scam 各 ~100 对，结构语气完全一致、唯一差异
  是否索费/索验证码，把关键信号隔离为唯一可学习特征
- **风格散布**：长度/正式度/实体名变体；每体裁人工抽检 ~20 条
- **简单负样本**（自然流量）：FBS Other 101 + UCI ham + 生成日常短信
  （验证码/快递/会议/家人问候）~1,500 条

### 6.3 生成样本 ≤ 总量 30% 约束

难负+简单负样本中**生成**部分（体裁 2-8 生成量 + 简单负样本，扣除抓取的体裁 1/5）
估算 ~12-13k，对 45-60k 总量天然在 30% 以内；构建时断言（超限则降生成量，
优先保抓取样本）。

### 6.4 挖掘闭环（Phase E，FPR≤2% 的保障）

- 候选池扩到万级（生成 10k 候选）→ v2 模型打分 → 取得分最高 benign
  （即模型误报样本）→ 人工复核剔除假负例（真诈骗被误标 benign 的）→ 增补训练
- 依据：静态构造假负例率 15-25%，滚动挖掘可压到 3-6%（报告 §四）
- **硬上限 2 轮**；每轮后重测难负评测集；2 轮后仍 >2% 如实报告差距

---

## 7. 数据配比与评测

### 7.1 训练集配比（二元口径，总量 45-60k）

| 维度 | 规格 |
|---|---|
| 诈骗正样本 | 40-50%，刷单返利 ≥25%（CCL 真实数据），投资理财类降采样（FGRC 过度采样 19k） |
| 难负样本 | 负样本的 60-70% |
| 提示类负样本（FGRC 无风险） | 负样本的 10-15%（从 v1 主导降为配角） |
| 简单负样本 | 负样本的 20-25% |
| CCL 非刷单返利笔录 | 正样本内 ≤20%（rebate_scam 豁免——勘误见 §4.2） |
| ChiFraud 正常网页 | benign 内 ≤10% |
| 英文 | ~15%（现有比例，+ Phishing Email 合法邮件） |

### 7.2 评测集（四套）

1. **难负评测集（新建，永不进训练）**：8 体裁 × 50-100 条 held-out → FPR ≤2% 主门槛；
   误报率分母 = 难负样本（非提示类 —— 报告：在提示类上测的 FPR 显著低于真实值）
2. **600 条中文 holdout**（数据集重建后重新采样，门槛为绝对值）：zh acc ≥0.90 /
   recall ≥0.95（与 v1 数字非同批样本，门槛沿用绝对值而非对比）
3. **手写集 38**（固定，前后直接可比）：14 类 acc ≥0.70 不回退
4. **public.jsonl 400**（固定）：英文侧不回退

评测输出：**按类别召回**（微平均会掩盖关键类别崩溃）+ **分体裁 FPR** +
**阈值-召回曲线**（0.5 之外附三段式阈值参考点：先定目标精确率反解阈值 A/B，
再在该阈值上度量召回 —— 报告 §五决策协议）。

---

## 8. 训练与导出（沿用现有管线）

- 平台：Apple M4 Pro（MPS），LoRA r=8（target Wqkv+Wo，1.15M 可训参数）
- 目标：交叉熵双任务（noul + choice），fp32，OneCycleLR
- 样本预算 45-60k（v1 15k/73min → 预估 4-5 小时）
- epochs 3-4，`--max-chars 1024` 截断保留，预编码 + 精确 padding
- `scripts/train_local_lora.py` **不动**，只换输入数据文件
- `scripts/export_local_onnx.py` **不动** → `models/laya-onnx-multilingual-finetuned-v2/`
- 验证：ONNX vs PyTorch logit diff < 1e-3（现有验证步骤）

---

## 9. 组件与文件布局

**新增**：

| 组件 | 职责 |
|---|---|
| `scripts/scrape_antifraud.py` | 抓取公安/政府反诈宣传文本（curl 模式，来源入 source 字段） |
| `scripts/generate_negatives.py` | arkcli 多模型生成（体裁 2-8 + 简单负样本 + 对照对），逐条记录生成器 |
| `scripts/audit_fgrc_labels.py` | v1 模型跑 FGRC 样本，输出标签分歧抽查清单 |
| `scripts/mine_hard_negatives.py` | Phase E 滚动挖掘（复用 `src.laya_onnx.OnnxLayaClient`，无需训练栈） |
| `scripts/build_hardneg_evalset.py` | 训练混入前切出难负评测集，id 不相交断言 |

**修改**：`schemas/scam_categories.py`（14 类 + 三张新映射表）、
`scripts/fetch_datasets.py`（4 个新源）、`scripts/build_dataset.py`（新 loader +
配比/上限/去重/评测集排除）、`scripts/check_acceptance.py`（4 门槛）、
`src/eval.py` + `src/report.py`（分类别召回、分体裁 FPR、阈值曲线）。

**产出文件**：
- `datasets/hard_negatives/*.jsonl`（生成/抓取难负样本，**git 跟踪** —— 沿用
  synthetic_extra.jsonl 先例：数据即证据，训练可复现）
- `datasets/hardneg_eval.jsonl`（难负评测集，git 跟踪）
- `models/laya-onnx-multilingual-finetuned-v2/`（gitignored）

**错误处理**：
- HF gated/401 → 跳过并提示（现有模式）；Kaggle 凭据缺失 → Phishing Email 优雅跳过
- arkcli 单模型失败 → 重试后换模型家族；生成溯源记录
- 挖掘闭环硬上限 2 轮，不达标如实报告
- 长文本 1024 截断（现有）；ASR/多轮对话沿用 FGRC 笔录格式

**测试**：
- 新 loader + 三张映射表单元测试（无需模型）：rebate_scam 在 14 类中、
  CCL 12 类全映射、ChiFraud 仅地下贷款映射
- 配比不变量测试：上限遵守（CCL≤20%、ChiFraud≤10%、生成≤30%）、
  训练/评测 id 不相交（污染防护）
- 现有 115 测试保持通过；v2 导出后重跑模型依赖测试

---

## 10. 风险与边界

| 风险 | 缓解 |
|---|---|
| FPR≤2% 单模型一次训练达不到（先例靠四模型集成+每周重校准） | 挖掘闭环预算 2 轮；仍不达标如实报告 —— 这是激进目标，不是硬承诺 |
| CCL 笔录体裁与短信不同源，可能引入分布偏移 | 正样本 ≤20% 上限 + 分体裁 FPR 监控 |
| 生成样本质量参差 / 假负例混入 | 多模型分散 + 人工复核清单 + 挖掘轮人工剔除 |
| ChiFraud 时间漂移（2022→2023 类别占比剧变） | 只取子集作 benign 补充 ≤10%，不入正样本主体 |
| arkcli 生成 API 不稳定 | 重试 + 换家族；生成量可分批补齐，不阻塞其余管线 |
| 刷单返利召回 ≥0.80 可能因 CCL 笔录体裁而偏乐观 | 评测集里刷单返利样本独立成组报告 |

**诚实预期**：报告明确指出文本单模态下「召回 ≥99% 与误报 <1%」双约束不可同时
达成；本轮把误报率压到可用区间、补齐最高发类别，是把模型从「关键词检测器」
升级为「结构判别器」的一步，终极误报率仍需生产误报日志反哺。

---

## 11. 未验证项（实现期确认）

- CCL2023-FCC / ChiFraud 仓库实际文件结构与下载路径（实现时探测文件清单再下载，
  沿用「用 HF API / GitHub 页面列真实文件再下载」的既有教训）
- TeleAntiFraud HF 侧文件是否可直接 curl（Apache-2.0 声明 vs 魔搭 gated 的差异）
- arkcli 可用模型家族清单与生成质量基线（生成前先小批量试产）
- FGRC 标签噪声全量比例（审计脚本给出量化数字，但只做二元用途）
