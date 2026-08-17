# Word2Vec 词向量模型复现实验报告

**题目**：基于 PyTorch 的 Word2Vec（CBOW / Skip-gram）复现与 NNLM、RNNLM 基线对比实验  

**实验类型**：算法复现 / 对比实验  

**训练语料**：英文维基 enwiki（及同规格 enwik9 级清洗语料）  

**评测基准**：Google Word Analogy（`word-test.v1.txt`，lowercase 匹配）  

**日期**：2026 年  

---

## 摘要

本报告对 Mikolov 等人提出的 Word2Vec 词向量方法进行完整复现，实现了连续词袋模型（CBOW）与跳字模型（Skip-gram），并采用负采样（Negative Sampling）与高频词子采样（Subsampling）加速训练。为验证预测式词向量相对传统神经语言模型的优势，本实验进一步实现前馈神经语言模型（NNLM）与循环神经语言模型（RNNLM）作为基线。

**全部主实验均在 enwiki 语料设定下完成**（主方法使用 enwik9 级大规模清洗维基；基线使用 `enwiki_train.txt`，因全 Softmax 代价限制词表为 5 万）。评测采用统一的 Google 词类比测试集与小写匹配协议。主要结果如下：

| 模型 | Overall | Semantic | Syntactic |
|------|--------:|---------:|----------:|
| **Skip-gram**（主方法） | **36.02%** | **33.99%** | **37.70%** |
| **CBOW**（主方法） | **26.37%** | 14.80% | 35.97% |
| **NNLM**（基线） | **9.61%** | 4.55% | 12.80% |
| **RNNLM**（基线） | **2.40%** | 2.69% | 2.22% |

结果表明：在同一 enwiki 语料与统一评测协议下，（1）负采样 Word2Vec 显著优于 NNLM / RNNLM 基线；（2）Skip-gram 在本设定下 Overall 与语义类比均优于 CBOW；（3）基线中 NNLM 强于简单 RNNLM。

**关键词**：Word2Vec；CBOW；Skip-gram；负采样；NNLM；RNNLM；enwiki；词类比

---

## 1 引言

### 1.1 研究背景

词的分布式表示是现代自然语言处理的基础。传统 one-hot 编码维度高、语义无关。Bengio 等人提出的神经概率语言模型（NNLM）引入低维连续词向量，但全词表 Softmax 训练代价高。Mikolov 等人于 2013 年提出 Word2Vec，通过 CBOW / Skip-gram 与负采样、子采样等技术，使大规模语料上的词向量训练高效可行，并在词类比任务上表现突出。

### 1.2 复现目标

1. 在 PyTorch 中实现负采样版 **CBOW** 与 **Skip-gram**；  
2. 实现 **NNLM**、**RNNLM** 基线并导出输入词向量；  
3. 在 **enwiki** 语料上完成训练，并在 Google 词类比集上统一评测；  
4. 定量分析主方法与基线、CBOW 与 Skip-gram 的差异。

### 1.3 主要工作

- 数据管线：词表、`min_count`、负采样 \(f^{0.75}\)、子采样；  
- 模型：CBOW / Skip-gram / NNLM / RNNLM；  
- 训练：SGD、学习率衰减、checkpoint、tqdm；  
- 评测：`word_analogy_eval.py` + lowercase。

---

## 2 相关工作

| 方法 | 代表工作 | 核心思想 | 本文角色 |
|------|----------|----------|----------|
| NNLM | Bengio et al., 2003 | 前馈网络预测下一词，全 Softmax | 基线 |
| RNNLM | Mikolov 等 | 循环网络建模序列 | 基线 |
| Word2Vec | Mikolov et al., 2013 | CBOW/Skip-gram + 负采样等 | 主方法 |

本复现聚焦**负采样**路径。

---

## 3 方法原理

### 3.1 Skip-gram

用中心词 \(w_t\) 预测窗口内上下文词。负采样目标：

\[
\mathcal{L}_{\text{SG}}
=
-\log\sigma\!\left(u_{w_t}^{\top} v_{w_c}\right)
-\sum_{k=1}^{K}\log\sigma\!\left(-u_{w_t}^{\top} v_{w_{n_k}}\right)
\]

导出并评测 **输入向量 \(u\)**（中心词表）。

### 3.2 CBOW

用上下文预测中心词。上下文向量（掩码平均）：

\[
h=\frac{1}{|\mathcal{C}|}\sum_{w\in\mathcal{C}} u_w
\]

负采样损失形式与 Skip-gram 类似，以 \(h\) 替代中心词向量。导出 **上下文侧 \(u\)**。

### 3.3 负采样与子采样

\[
P(w)\propto f(w)^{0.75},\qquad
P_{\text{keep}}(w)=\left(\sqrt{\frac{f(w)}{t}}+1\right)\frac{t}{f(w)}
\]

默认 \(K=5\)，\(t=10^{-5}\)。

### 3.4 基线 NNLM

前 \(n=5\) 词 embedding 拼接 → \(\tanh\) 隐层 → 全 Softmax 预测下一词；词向量 = 输入 embedding。

### 3.5 基线 RNNLM

简单 Elman RNN + BPTT（seq_len=35）→ 全 Softmax；词向量 = 输入 embedding。  
类比仅用静态向量，RNN 信息多在隐状态，故类比常弱于 NNLM。

### 3.6 与原版差异（简要）

固定窗口；学习率分段衰减；CBOW 用 mean+mask；基线因算力限制 `max_vocab=50000`。

---

## 4 实验设置

### 4.1 环境

| 项目 | 配置 |
|------|------|
| 操作系统 | Windows |
| GPU | NVIDIA GeForce RTX 5060 Laptop（约 8GB） |
| 框架 | PyTorch（CUDA） |
| 语言 | Python 3 |

### 4.2 语料（全文仅 enwiki）

本报告**所有主实验结果均基于 enwiki 语料设定**，不再使用 text8 作为主对比语料。

| 用途 | 语料 | 说明 |
|------|------|------|
| CBOW / Skip-gram | enwiki / enwik9 级大规模清洗维基 | 词表约 42.8 万（min_count 过滤后） |
| NNLM / RNNLM | `enwiki_train.txt` | 同系列 enwiki；全 Softmax 故限制词表 5 万 |

预处理：空白分词、轻量去标点、`min_count` 过滤；基线另设 `max_vocab=50000`。

### 4.3 模型与产出文件

| 模型 | 产出文件 | 词表×维度 | 语料 | 关键设定 |
|------|----------|-----------|------|----------|
| **Skip-gram** | `skipgram_enwik9_best.txt` | 427865 × 300 | enwiki 级 | 负采样；window=5；SGD |
| **CBOW** | `word_embedding.txt` | 427865 × 300 | enwiki 级 | 负采样；window=5；SGD |
| **NNLM** | `nnlm_enwiki.txt` | 50000 × 300 | enwiki | context=5；全 Softmax |
| **RNNLM** | `rnnlm_enwiki_d300.txt` | 50000 × 300 | enwiki | seq=35；简单 RNN |

说明：主方法与基线优化目标、词表规模不完全相同；维度统一为 300；评测协议完全一致。学习率按模型族独立设定。

### 4.4 评测协议

- 测试集：`word-test.v1.txt`，**19544** 题；  
- 规则：\(v(b)-v(a)+v(c)\)，余弦最近邻，排除 a,b,c；  
- **`--lowercase`** 小写匹配；  
- 准确率 = Correct / Valid。

```bash
python word_analogy_eval.py --embedding-path <文件> --test-path word-test.v1.txt --lowercase
```

### 4.5 公平性

| 维度 | CBOW / Skip-gram | NNLM / RNNLM |
|------|------------------|--------------|
| 语料系列 | enwiki | enwiki |
| 词表 | ~42.8 万 | 5 万 |
| OOV（类比） | 0 | 4938 |
| 维度 | 300 | 300 |
| 目标 | 负采样上下文 | 全 Softmax LM |

---

## 5 实验结果

### 5.1 主结果总表

**表 1 词类比主结果（enwiki + lowercase）**

| 模型 | 向量文件 | 词表 | 维 | Valid | OOV | **Overall** | Semantic | Syntactic |
|------|----------|-----:|---:|------:|----:|------------:|---------:|----------:|
| **Skip-gram** | skipgram_enwik9_best.txt | 427865 | 300 | 19544 | 0 | **36.02%** | **33.99%** | **37.70%** |
| **CBOW** | word_embedding.txt | 427865 | 300 | 19544 | 0 | **26.37%** | 14.80% | 35.97% |
| **NNLM** | nnlm_enwiki.txt | 50000 | 300 | 14606 | 4938 | **9.61%** | 4.55% | 12.80% |
| **RNNLM** | rnnlm_enwiki_d300.txt | 50000 | 300 | 14606 | 4938 | **2.40%** | 2.69% | 2.22% |

相对 Skip-gram 的 Overall：

- Skip-gram / CBOW ≈ **1.37×**  
- Skip-gram / NNLM ≈ **3.7×**  
- Skip-gram / RNNLM ≈ **15×**

### 5.2 Skip-gram 分关系结果

**表 2 Skip-gram 语义关系**

| 关系 | Total | Valid | OOV | Correct | Acc (%) |
|------|------:|------:|----:|--------:|--------:|
| capital-common-countries | 506 | 506 | 0 | 400 | **79.05** |
| capital-world | 4524 | 4524 | 0 | 1732 | **38.28** |
| currency | 866 | 866 | 0 | 85 | 9.82 |
| city-in-state | 2467 | 2467 | 0 | 597 | 24.20 |
| family | 506 | 506 | 0 | 201 | 39.72 |
| **semantic 合计** | 8869 | 8869 | 0 | 3015 | **33.99** |

**表 3 Skip-gram 句法关系**

| 关系 | Total | Valid | OOV | Correct | Acc (%) |
|------|------:|------:|----:|--------:|--------:|
| gram1-adjective-to-adverb | 992 | 992 | 0 | 122 | 12.30 |
| gram2-opposite | 812 | 812 | 0 | 135 | 16.63 |
| gram3-comparative | 1332 | 1332 | 0 | 741 | **55.63** |
| gram4-superlative | 1122 | 1122 | 0 | 131 | 11.68 |
| gram5-present-participle | 1056 | 1056 | 0 | 341 | 32.29 |
| gram6-nationality-adjective | 1599 | 1599 | 0 | 1224 | **76.55** |
| gram7-past-tense | 1560 | 1560 | 0 | 434 | 27.82 |
| gram8-plural | 1332 | 1332 | 0 | 637 | 47.82 |
| gram9-plural-verbs | 870 | 870 | 0 | 260 | 29.89 |
| **syntactic 合计** | 10675 | 10675 | 0 | 4025 | **37.70** |

Skip-gram 在首都类（common 79.05%、world 38.28%）与国籍形容词（76.55%）上显著强于 CBOW，是 Overall 领先的主要来源。

### 5.3 CBOW 分关系结果

**表 4 CBOW 语义关系**

| 关系 | Total | Valid | OOV | Correct | Acc (%) |
|------|------:|------:|----:|--------:|--------:|
| capital-common-countries | 506 | 506 | 0 | 215 | 42.49 |
| capital-world | 4524 | 4524 | 0 | 503 | 11.12 |
| currency | 866 | 866 | 0 | 24 | 2.77 |
| city-in-state | 2467 | 2467 | 0 | 349 | 14.15 |
| family | 506 | 506 | 0 | 222 | **43.87** |
| **semantic 合计** | 8869 | 8869 | 0 | 1313 | **14.80** |

**表 5 CBOW 句法关系**

| 关系 | Total | Valid | OOV | Correct | Acc (%) |
|------|------:|------:|----:|--------:|--------:|
| gram1-adjective-to-adverb | 992 | 992 | 0 | 183 | 18.45 |
| gram2-opposite | 812 | 812 | 0 | 71 | 8.74 |
| gram3-comparative | 1332 | 1332 | 0 | 649 | **48.72** |
| gram4-superlative | 1122 | 1122 | 0 | 181 | 16.13 |
| gram5-present-participle | 1056 | 1056 | 0 | 366 | 34.66 |
| gram6-nationality-adjective | 1599 | 1599 | 0 | 1039 | **64.98** |
| gram7-past-tense | 1560 | 1560 | 0 | 458 | 29.36 |
| gram8-plural | 1332 | 1332 | 0 | 664 | **49.85** |
| gram9-plural-verbs | 870 | 870 | 0 | 229 | 26.32 |
| **syntactic 合计** | 10675 | 10675 | 0 | 3840 | **35.97** |

### 5.4 NNLM 基线分关系结果

**表 6 NNLM 语义关系（enwiki）**

| 关系 | Total | Valid | OOV | Correct | Acc (%) |
|------|------:|------:|----:|--------:|--------:|
| capital-common-countries | 506 | 506 | 0 | 29 | 5.73 |
| capital-world | 4524 | 2137 | 2387 | 28 | 1.31 |
| currency | 866 | 236 | 630 | 0 | 0.00 |
| city-in-state | 2467 | 2394 | 73 | 59 | 2.46 |
| family | 506 | 380 | 126 | 141 | **37.11** |
| **semantic 合计** | 8869 | 5653 | 3216 | 257 | **4.55** |

**表 7 NNLM 句法关系**

| 关系 | Total | Valid | OOV | Correct | Acc (%) |
|------|------:|------:|----:|--------:|--------:|
| gram1-adjective-to-adverb | 992 | 870 | 122 | 36 | 4.14 |
| gram2-opposite | 812 | 380 | 432 | 58 | 15.26 |
| gram3-comparative | 1332 | 1332 | 0 | 268 | **20.12** |
| gram4-superlative | 1122 | 650 | 472 | 76 | 11.69 |
| gram5-present-participle | 1056 | 1056 | 0 | 132 | 12.50 |
| gram6-nationality-adjective | 1599 | 1371 | 228 | 50 | 3.65 |
| gram7-past-tense | 1560 | 1482 | 78 | 209 | 14.10 |
| gram8-plural | 1332 | 1056 | 276 | 102 | 9.66 |
| gram9-plural-verbs | 870 | 756 | 114 | 215 | **28.44** |
| **syntactic 合计** | 10675 | 8953 | 1722 | 1146 | **12.80** |

### 5.5 RNNLM 基线分关系结果

**表 8 RNNLM 语义关系（enwiki, d=300）**

| 关系 | Total | Valid | OOV | Correct | Acc (%) |
|------|------:|------:|----:|--------:|--------:|
| capital-common-countries | 506 | 506 | 0 | 15 | 2.96 |
| capital-world | 4524 | 2137 | 2387 | 9 | 0.42 |
| currency | 866 | 236 | 630 | 0 | 0.00 |
| city-in-state | 2467 | 2394 | 73 | 55 | 2.30 |
| family | 506 | 380 | 126 | 73 | **19.21** |
| **semantic 合计** | 8869 | 5653 | 3216 | 152 | **2.69** |

**表 9 RNNLM 句法关系**

| 关系 | Total | Valid | OOV | Correct | Acc (%) |
|------|------:|------:|----:|--------:|--------:|
| gram1-adjective-to-adverb | 992 | 870 | 122 | 2 | 0.23 |
| gram2-opposite | 812 | 380 | 432 | 8 | 2.11 |
| gram3-comparative | 1332 | 1332 | 0 | 15 | 1.13 |
| gram4-superlative | 1122 | 650 | 472 | 0 | 0.00 |
| gram5-present-participle | 1056 | 1056 | 0 | 38 | 3.60 |
| gram6-nationality-adjective | 1599 | 1371 | 228 | 16 | 1.17 |
| gram7-past-tense | 1560 | 1482 | 78 | 23 | 1.55 |
| gram8-plural | 1332 | 1056 | 276 | 15 | 1.42 |
| gram9-plural-verbs | 870 | 756 | 114 | 82 | **10.85** |
| **syntactic 合计** | 10675 | 8953 | 1722 | 199 | **2.22** |

### 5.6 亮点子任务横向对比

**表 10 代表性子任务准确率（%）**

| 子任务 | Skip-gram | CBOW | NNLM | RNNLM |
|--------|----------:|-----:|-----:|------:|
| capital-common-countries | **79.05** | 42.49 | 5.73 | 2.96 |
| capital-world | **38.28** | 11.12 | 1.31 | 0.42 |
| family | 39.72 | **43.87** | 37.11 | 19.21 |
| nationality-adjective | **76.55** | 64.98 | 3.65 | 1.17 |
| comparative | **55.63** | 48.72 | 20.12 | 1.13 |
| plural | 47.82 | **49.85** | 9.66 | 1.42 |
| plural-verbs | 29.89 | 26.32 | **28.44** | 10.85 |
| **Overall** | **36.02** | 26.37 | 9.61 | 2.40 |

### 5.7 定性分析

**Skip-gram 正确示例**（评测日志）：

- `Athens:Greece :: Baghdad:Iraq` → iraq ✓  
- `Athens:Greece :: Bangkok:Thailand` → thailand ✓  
- `Athens:Greece :: Beijing:China` → china ✓  
- `Athens:Greece :: Berlin:Germany` → germany ✓  

**CBOW**：多数首都题可做对，但弱于 Skip-gram；family 略强于 Skip-gram（43.87% vs 39.72%）。  

**NNLM / RNNLM**：首都类常误为高频国名；RNNLM 更弱。

### 5.8 未纳入主表的结果

| 对象 | 说明 |
|------|------|
| `skip_best.txt` | Overall≈0%，向量异常，**不采用** |
| 无 `--lowercase` 评测 | 专名假 OOV，口径失真，**不采用** |
| text8 上的小规模实验 | 本报告主文**仅保留 enwiki**，不再作为主对比语料 |

---

## 6 分析与讨论

### 6.1 Word2Vec 显著优于基线

Skip-gram（36.02%）与 CBOW（26.37%）均远高于 NNLM（9.61%）与 RNNLM（2.40%）。原因：负采样直接塑造共现几何；全 Softmax LM 以预测下一词为主，词向量为副产品；且基线词表受 Softmax 限制。

### 6.2 Skip-gram 优于 CBOW（本设定）

本实验中 Skip-gram Overall 与 **语义**（33.99% vs 14.80%）明显更高，尤其 capital-world（38.28% vs 11.12%）。符合文献经验：Skip-gram 对稀有词与语义关系往往更强；CBOW 在部分句法/family 上可更稳或略好。句法两者接近（37.70% vs 35.97%）。

### 6.3 句法与语义

主方法句法与语义均较强（Skip-gram 二者接近）；基线仍呈句法 ≥ 语义（NNLM）或整体偏弱（RNNLM）。形态规律比世界知识型语义更易从共现中归纳。

### 6.4 NNLM 强于 RNNLM

二者 Valid/OOV 相同，差距来自表示：NNLM 直接优化上下文 embedding 组合；简单 RNN 信息在 \(h_t\)，静态表类比弱，且更难训练。

### 6.5 语料统一为 enwiki 的意义

将主实验统一到 enwiki，避免 text8 与大语料混比带来的解读困难。基线与主方法仍存在词表规模差异，已在正文声明；在 family 等高覆盖子任务上，主方法仍全面领先，结论稳健。

---

## 7 系统实现说明

### 7.1 代码结构

| 模块 | 文件 |
|------|------|
| CBOW / Skip-gram 模型 | `model.py` |
| 数据与采样 | `input_data.py` |
| Word2Vec 训练 | `word2vec.py` |
| 基线模型 | `baseline_models.py` |
| 基线数据 / 训练 | `lm_data.py`, `train_baseline.py` |
| 类比评测 | `word_analogy_eval.py` |

### 7.2 关键命令

```bash
# 评测（enwiki 结果）
python word_analogy_eval.py --embedding-path skipgram_enwik9_best.txt --test-path word-test.v1.txt --lowercase
python word_analogy_eval.py --embedding-path word_embedding.txt --test-path word-test.v1.txt --lowercase
python word_analogy_eval.py --embedding-path nnlm_enwiki.txt --test-path word-test.v1.txt --lowercase
python word_analogy_eval.py --embedding-path rnnlm_enwiki_d300.txt --test-path word-test.v1.txt --lowercase
```

基线训练示例：

```bash
python train_baseline.py --model nnlm --input enwiki_train.txt --output nnlm_enwiki.txt --emb-dim 300 --hidden 300 --max-vocab 50000 --stream ...
python train_baseline.py --model rnnlm --input enwiki_train.txt --output rnnlm_enwiki_d300.txt --emb-dim 300 --hidden 300 --max-vocab 50000 ...
```

---

## 8 结论

1. 在 **enwiki** 语料与统一 Google 类比评测（lowercase）下，完成了 CBOW、Skip-gram、NNLM、RNNLM 的复现与对比。  
2. **Skip-gram Overall 36.02%**，**CBOW 26.37%**，均显著优于 **NNLM 9.61%** 与 **RNNLM 2.40%**。  
3. 本设定下 **Skip-gram 优于 CBOW**，优势主要体现在语义与专名类关系。  
4. 基线中 **NNLM 强于简单 RNNLM**，与静态词向量类比机制一致。  
5. 局限：基线词表为 5 万；与官方 C 实现仍有固定窗口等细节差异。展望：同词表严格对齐、词相似度评测、更强 RNN/LSTM 基线。

---

## 参考文献

[1] Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean. *Efficient Estimation of Word Representations in Vector Space*. ICLR Workshop, 2013.  

[2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean. *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS, 2013.  

[3] Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin. *A Neural Probabilistic Language Model*. JMLR, 2003.  

[4] Tomas Mikolov, Martin Karafiát, Lukáš Burget, Jan Černocký, Sanjeev Khudanpur. *Recurrent Neural Network Based Language Model*. INTERSPEECH, 2010.  

---

## 附录 A 词向量文件清单（enwiki）

| 文件 | 规格 | 角色 |
|------|------|------|
| skipgram_enwik9_best.txt | 427865 × 300 | Skip-gram 正式结果 |
| word_embedding.txt | 427865 × 300 | CBOW 正式结果 |
| nnlm_enwiki.txt | 50000 × 300 | NNLM 正式基线 |
| rnnlm_enwiki_d300.txt | 50000 × 300 | RNNLM 正式基线 |

## 附录 B 主结果一览（答辩用）

```
语料：enwiki | 评测：Google Analogy + lowercase

Skip-gram  Overall 36.02% | Semantic 33.99% | Syntactic 37.70% | OOV 0
CBOW       Overall 26.37% | Semantic 14.80% | Syntactic 35.97% | OOV 0
NNLM       Overall  9.61% | Semantic  4.55% | Syntactic 12.80% | OOV 4938
RNNLM      Overall  2.40% | Semantic  2.69% | Syntactic  2.22% | OOV 4938

结论：Skip-gram > CBOW ≫ NNLM > RNNLM
```

---

*（完）*
