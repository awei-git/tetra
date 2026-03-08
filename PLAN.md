# Tetra — 市场分析与交易系统

## 定位

合并 minutes (报告) + tetra (数据/模型) 为一个系统。最终取代 Mira analyst agent。

**核心哲学: 不做任何人都能做的分析。** 传统技术分析 (RSI, SMA, 均线交叉) 和简单统计 (momentum ranking, sector rotation) 是 commodity — 你直接问 LLM 就能得到。我们做的是: **基于独特数据组合的新方法，大量使用 token 做别人做不到的分析。**

---

## 现状审计 (2026-03-07)

### 基础设施

| 组件 | 状态 |
|------|------|
| PostgreSQL (Docker) | tetra-db 容器存在，可启动 |
| LaunchAgent | 无 (tetra/minutes 都没注册，只有 Mira) |
| Minutes | 手动运行，15份PDF报告 (Feb-Mar) |
| Tetra | 手动运行，数据新鲜到 Mar 6 |

### 数据库 (20 tables, 数据质量好)

| 表 | 行数 | 最新 | 覆盖 |
|----|------|------|------|
| market.ohlcv | 470K | Mar 6 | 205 symbols, 10年历史 |
| news.articles | 30K | Mar 6 | 301 sources, VADER sentiment |
| economic.values | 51K | Mar 6 | 59 FRED series |
| event.events | 5.6K | Mar 6 | earnings/dividends/splits/IPO |
| polymarket.markets | 10K | Mar 6 | 全部活跃市场 |
| polymarket.snapshots | 480K | Mar 6 | 时间序列快照 |
| fundamentals.financials | 7.9K | - | 财报 (income/balance/CF) |
| fundamentals.shares | 9.5K | Mar 6 | 市值/流通股 |
| factors.daily_factors | 395K | Mar 3 | 81个因子 (大多是commodity) |
| gpt.recommendations | 290 | Mar 4 | 3个LLM provider |

### API Keys

| API | 状态 | 有用的数据 |
|-----|------|-----------|
| **Polygon** | Basic plan, 工作 | OHLCV (daily+分钟), financials, news, options contracts (无snapshot) |
| **Finnhub** | Free tier, 工作 | insider trades, analyst consensus, earnings surprise, IPO calendar |
| **FRED** | 工作 | 59个宏观序列 (yields, CPI, employment, credit spreads, etc) |
| **NewsAPI** | 工作 | 全文新闻搜索 (11K+ results) |
| **AlphaVantage** | 工作 | daily prices, earnings (备用) |
| **Polymarket** | 工作 (Gamma API) | prediction market prices, orderbook, 时间序列 |
| **OpenAI** | 工作 | GPT-5.2 |
| **DeepSeek** | 工作 | DeepSeek V3.2 |
| **Gemini** | 工作 | Gemini 3 Flash |
| **Anthropic** | **未配置** | 需要加 key |

### 现有81个因子 (大部分要扔掉)

```
event.*     (18个) — event counts, momentum, breadth — 有点用但太简单
macro.*     (24个) — VIX/yield/credit/oil z-scores, changes — commodity
mkt.*       (19个) — RSI, SMA, momentum, vol — 纯技术分析，扔掉
news.*      (20个) — sentiment, volume, momentum — VADER太粗糙，要重做
```

### Portfolio (从 secrets.yml)

```
META: 2657 shares (~$2.05M)
GOOG: 3306 shares (~$6.6M)
Others: $545K
401K: $830K
Cash: $640K
Total: ~$10.7M
```

---

## 清理计划 (Phase 0)

### 要做的
1. **停掉 minutes** — 代码迁入 tetra (LLM 辩论引擎 + PDF/email delivery)
2. **清理 tetra factors** — 删掉全部 mkt.* 技术分析因子 (RSI, SMA, momentum 等)
3. **注册 LaunchAgent** — tetra 统一调度
4. **配置 Anthropic key** — 加入 Claude 作为分析 LLM
5. **合并 secrets.yml** — 一份配置

### 要保留的
- 全部数据管道 (market, events, economic, news, fundamentals, polymarket)
- DB schema & session 管理
- Market universe definitions
- LLM clients (从 minutes 迁入)
- PDF/email delivery (从 minutes 迁入)
- Stress testing framework (simulations/paths.py)

### 要扔掉的
- minutes 整个目录 (迁移完成后)
- 全部 mkt.* 因子 (RSI, SMA, drawdown, etc.)
- 简单 macro z-score 因子 (保留原始数据, 重新设计使用方式)
- 现有 GPT recommendation prompt (太简单)
- Factor scoring/weighting system (重新设计)

---

## 新方法论 (Phase 1-3)

### 核心思路: LLM-as-Analyst, 不是 LLM-as-Narrator

现有系统的问题: LLM 只看到价格数据, 然后"讨论"。这和你自己问 ChatGPT 没区别。

新思路: **LLM 是一个有独特信息优势的分析师。** 它能同时处理:
- 30K篇新闻的情绪和叙事变化
- 10K个 prediction market 的价格信号
- 59个宏观指标的交叉关系
- Insider trading patterns
- Analyst consensus shifts
- Earnings surprise patterns

**没有人能同时消化这么多维度的信息。这就是 edge。**

---

### Method 1: Narrative Regime Detection (叙事政权检测)

**用 LLM 分析新闻叙事的结构性变化，不是简单的 sentiment score。**

```
传统方法: "AAPL sentiment = 0.65 (positive)"  ← 无用
我们的方法: "过去72小时，AAPL的叙事从'AI growth story'转向'antitrust risk' —
            这种叙事转换在历史上平均导致3.2%的回调，在当前 risk-on regime 下
            幅度可能更小(1.5-2%)，窗口7-14天"
```

**实现:**

```python
# src/analysis/narrative.py

# Step 1: 每天用 LLM 对当日新闻做 structured narrative extraction
# 不是 sentiment, 而是:
# - dominant_narrative: "AI capex cycle" / "antitrust crackdown" / "earnings momentum"
# - narrative_shift: 和前几天的叙事相比，方向变了吗？
# - narrative_confidence: 多少篇文章在讲同一个故事？
# - counter_narrative: 有没有相反的声音在增长？
# - narrative_novelty: 这个叙事是新的还是老的？新叙事 = 更大的 market impact

# Step 2: 建立 narrative → market impact 的历史数据库
# 当某种叙事出现过后，市场后续怎么走？
# 这需要用 LLM 回溯性地标注历史新闻 (one-time cost)

# Step 3: 当前叙事 vs 历史 pattern matching
# "当前叙事最像2024年X月，当时market做了Y"

# 输入: news.articles (30K rows), daily batches
# 输出: narrative.daily_state 表
# - date, symbol/sector/market
# - dominant_narrative (text)
# - narrative_shift_signal (-1 to +1, 方向变化)
# - shift_magnitude (0 to 1, 变化幅度)
# - historical_parallels (JSON, 历史相似情况)
# - expected_impact (JSON, 预期市场影响)

# Token cost: ~$5-15/day (batch process 50-200 articles through Claude)
```

**为什么这有 edge:**
- VADER sentiment 只能给 +/- 分。LLM 能理解叙事结构变化。
- 叙事转换是 leading indicator (在价格反应之前)。
- 没有现成的 API 或工具能做这个。

---

### Method 2: Cross-Asset Information Flow (跨资产信息流)

**观察信息在不同市场间的传播速度和模式。**

```
传统方法: "VIX涨了 → 市场恐慌" ← 同时发生，无预测价值
我们的方法: "Polymarket '美联储3月降息' 合约从0.45跌到0.28 (48小时内)，
            但债券市场尚未完全定价 — DGS2只动了3bp (历史上类似情况平均
            会动8-12bp) — 这里有一个信息传播的 gap"
```

**实现:**

```python
# src/analysis/information_flow.py

# 核心概念: 不同市场对同一个信息的反应速度不同
# Polymarket: 最快 (crowd wisdom, 24/7, 无交易成本)
# Rates: 中速 (机构主导, 流动性好)
# Equities: 最慢 (散户噪音大, 叙事驱动)

# Step 1: Polymarket Signal Extraction
# - 监控所有宏观相关合约 (Fed decisions, geopolitics, policy)
# - 用 LLM 解析 Polymarket 合约含义 → 映射到传统市场影响
#   "如果 'US recession by Dec 2026' 从 20% → 35%，意味着什么?"
# - 计算: Polymarket implied vs 传统市场 actual 的差距 (gap)
#
# 输入: polymarket.snapshots (480K), polymarket.markets (10K)
# 输出:
# - polymarket.signals (date, market_id, implied_view, current_gap, gap_z_score)
# - polymarket.macro_mapping (market_id → affected_assets, expected_direction)

# Step 2: Information Propagation Speed
# - 当一个信息 (如 FOMC decision, CPI surprise) 发生：
#   - Polymarket 反应多快？(minutes)
#   - FRED/rates 反应多快？(hours)
#   - Equity sectors 反应多快？(hours to days)
# - 如果某个市场"还没反应完"，可能有 trading opportunity

# Step 3: Cross-Asset Divergence Detection
# - 通常 correlated 的资产突然 diverge → 一个是对的，一个是错的
# - 用 LLM 判断: 哪个更可能是对的？(基于信息来源分析)

# Token cost: ~$3-8/day (Polymarket 合约解析 + divergence 分析)
```

**为什么这有 edge:**
- Polymarket 数据大多数人不看 (或只看表面价格)。
- 跨资产信息流的 timing gap 是一个 well-known alpha source，但很少有人用 LLM 来系统化。
- 10K 个 Polymarket 合约是巨大的信息量，只有 LLM 能同时消化。

---

### Method 3: Earnings Intelligence Network (财报情报网络)

**不是看"AAPL beat by 4%"，而是理解整个 earnings season 的信息网络。**

```
传统方法: "AAPL beat estimates → bullish" ← 所有人都知道了
我们的方法: "台积电上周guidance提到'强劲AI需求'，联发科确认了类似趋势，
            NVDA下周报告 — 历史上当供应链公司先确认趋势时，NVDA beat
            概率是82% (vs base rate 65%)。但Polymarket已经price了75%
            beat — implied edge只有7pp"
```

**实现:**

```python
# src/analysis/earnings_network.py

# Step 1: 构建公司关系图
# - 供应链关系 (supplier → customer)
# - 行业同业 (sector peers)
# - 这个用 LLM 一次性构建，存为静态数据
#
# 输出: network.company_graph (JSON)
# - {symbol: {suppliers: [...], customers: [...], peers: [...], themes: [...]}}

# Step 2: Earnings Season Information Cascade
# - 每个 earnings report 不只是一个公司的信息
# - 它也是上下游公司的 leading indicator
# - 用 LLM 从每个 earnings report 中提取:
#   a. 对自身的 signals (obvious, everyone does this)
#   b. 对上下游的 implied signals (非obvious, 我们的 edge)
#   c. 对宏观环境的 signals (recession/growth indicators from micro data)
#
# 输入: event.events (earnings), news.articles (earnings coverage)
# 输出: earnings.cascading_signals 表
# - source_symbol, target_symbol, signal_type, magnitude, confidence
# - "TSMC guidance → NVDA expected beat probability +12%"

# Step 3: Aggregated Micro-to-Macro
# - 把所有 earnings 的 micro signals 聚合成 macro view
# - "本季度62%的科技公司提到capex增长 vs 上季度45% → sector momentum加速"
# - 这种 bottom-up 的 macro view 比看 top-down 指标更早更准

# Token cost: ~$10-20 per earnings season cycle
# (集中在 earnings weeks, 非 earnings weeks 几乎零成本)
```

---

### Method 4: Insider + Analyst Consensus Decomposition

**不是看"insider bought"，而是理解 informed trading pattern。**

```
传统方法: "3 insiders bought AAPL" ← 可能是 10b5-1 自动交易
我们的方法: "CFO在earnings前30天的non-routine窗口买入，同时2个board
            members也买入(不在他们的历史pattern中) — cluster buying
            with pattern break → strong signal (历史 hit rate 71%)"
```

**实现:**

```python
# src/analysis/informed_trading.py

# Finnhub 给我们 insider transactions (118 records for AAPL alone)
# Finnhub 给我们 analyst consensus (buy/hold/sell counts over time)

# Step 1: Insider Pattern Analysis
# - 建立每个insider的历史交易pattern (frequency, timing, size)
# - 检测 pattern breaks:
#   - 非常规时间交易 (不在通常的 vesting/10b5-1 窗口)
#   - 异常 size (远超历史平均)
#   - Cluster buying (多个 insider 同时买)
# - 用 LLM 交叉验证: insider buying + 最近新闻 = 什么信号?

# Step 2: Analyst Consensus Change Analysis
# - 不看绝对数字 (22 buy, 16 hold) — 这是公开信息
# - 看变化的速度和方向:
#   - 从哪家 firm 的 analyst 先动? (先行者通常更informed)
#   - Upgrade/downgrade 的理由和最近earnings的关系
#   - Consensus 变化 vs 股价变化的 lead/lag

# 输入: Finnhub insider trades + analyst recommendations
# 输出: informed.signals 表
# - symbol, date, signal_type (insider_cluster, analyst_momentum, pattern_break)
# - strength, historical_hit_rate, context

# Token cost: ~$2-5/day (LLM only for pattern break interpretation)
```

---

### Method 5: LLM Adversarial Debate with Data (数据驱动的对抗辩论)

**从 Minutes 保留 multi-LLM debate 框架，但彻底重新设计。**

```
Minutes 的方式: "分析大盘股" → 3个LLM各说一通 → 投票 ← 垃圾
新方式: 给每个 LLM 一个角色和独特的数据切片，让他们真正争论
```

**实现:**

```python
# src/analysis/debate.py

# 每个 LLM 扮演不同的 information set:

# Analyst A (Claude): Macro-first
# - 看到: 全部宏观数据, yield curve, credit spreads, FRED series
# - 不看到: 个股新闻, earnings
# - 视角: "从宏观环境出发，哪些资产应该表现好/差?"

# Analyst B (GPT): Micro-first
# - 看到: 个股 earnings, insider trades, analyst consensus, fundamentals
# - 不看到: 宏观数据
# - 视角: "从个股基本面出发，哪些公司有 edge?"

# Analyst C (DeepSeek): Crowd-first
# - 看到: Polymarket 合约, news narrative shifts, social sentiment
# - 不看到: fundamentals, macro data
# - 视角: "市场在想什么? 共识在哪里? 哪里有 mispricing?"

# Debate Structure:
# Round 1: 各自给出 top 5 trades (带数据支撑)
# Round 2: 互相挑战 — "你的数据里没看到X，而我的数据显示..."
# Round 3: 综合 — 哪些 trades 在所有 information sets 下都成立?
#          哪些只在某个 information set 下成立? (更高风险但可能更有 edge)
# Round 4: Risk assessment — "如果你错了，最可能因为什么?"

# 输出:
# - Consensus trades (高信心)
# - Contrarian trades (只有某个信息集支持，但理由充分)
# - Risk warnings (blind spots each analyst identified)

# Token cost: ~$15-25/day (3 LLMs × 4 rounds × detailed prompts)
```

**为什么这比 Minutes 好得多:**
- 每个 LLM 有 genuine information asymmetry (不是同一个 prompt 的微调)
- 辩论产生的是 actionable disagreement，不是平淡的"综合观点"
- Contrarian trades (一个 LLM 看到了别人没看到的) 是真正有 edge 的

---

### Method 6: Scenario Analysis 2.0

**不是历史 stress test (那是 backward-looking)，而是 forward-looking scenario construction。**

```
传统: "如果 COVID 重演，portfolio 跌 X%" ← 不会完全一样
新方式: "给定当前宏观状态(VIX 15, 10Y at 4.2%, HY spread 340bp),
        最可能的 tail risk 是什么? 概率多少? 对我的持仓影响如何?"
```

**实现:**

```python
# src/analysis/scenarios.py

# Step 1: LLM-generated forward scenarios
# 用全部可用数据 (macro, news, Polymarket, insider) 让 LLM 生成:
# - 3 个最可能的 near-term scenarios (1-3月)
# - 每个的概率 (calibrated against Polymarket where possible)
# - 每个对各资产类别的影响 (方向 + 幅度)
# - 1 个 black swan scenario (低概率但高影响)

# Step 2: Portfolio impact
# - 拿到 scenario → 用历史最近似事件的 return pattern
# - 加上当前 portfolio positions
# - 计算: 每个 scenario 下 portfolio PnL, max drawdown
# - 计算: 需要多少 hedge (以及用什么)

# Step 3: 动态更新
# - 每天用新数据 re-calibrate scenario 概率
# - 如果某个 scenario 概率大幅上升 → 预警

# 保留现有 stress test (16 historical scenarios) 作为 validation
# 但主力分析是 forward-looking LLM scenarios

# Token cost: ~$5-10/day
```

---

## 报告设计 (Phase 2)

报告不是数据dump。它是一个 **decision support document**。

### 每日报告结构

```
1. What Happened (1 page)
   - 一段话: 今天市场的本质 (不是"SPY涨了0.5%"，而是"市场在消化...")
   - 叙事变化: 有没有新的叙事出现/旧的消退?
   - 异常: 什么不正常? (信息流 gap, narrative shift, cluster insider buying)

2. What It Means (1-2 pages)
   - Regime: 当前regime + 变化概率
   - Cross-asset 状态: 哪里有 divergence? 什么信号还没被 price in?
   - Earnings intelligence: 供应链在说什么? (如果在 earnings season)

3. What To Do (2-3 pages)
   - New trades (有 novel signal 支撑的)
     - 每个 trade: thesis, data support, risk, confidence
     - 区分: consensus trades vs contrarian trades
   - Existing position review
     - 当前持仓 vs 新信号: 维持/加仓/减仓/止损?
     - 如果你的position和信号冲突: why, 谁可能是对的
   - Portfolio-level
     - Concentration risk
     - Scenario exposure (最可能的3个scenario下的PnL)
     - Hedge 建议 (如果需要)

4. Track Record (0.5 page)
   - 上期建议表现
   - 累计: 胜率, 平均收益
   - 各 method 的表现 (narrative method vs information flow vs earnings network)

5. Forward Calendar (0.5 page)
   - 未来5天重要事件
   - 哪些持仓会受影响
   - 需要提前做什么
```

### 和 Minutes 的根本区别

| Minutes (旧) | Tetra (新) |
|-------------|-----------|
| LLM 看价格数据，讨论 | LLM 有独特数据，分析 |
| 6个 segment 并行辩论 | 3个 LLM 有不同信息集 |
| 建议是"AAPL target $278" | 建议是"AAPL: narrative shifting + cluster insider buying + supply chain confirmation → 72% probability of..." |
| 不 track | 每个建议持续追踪 |
| 不知道你的持仓 | 基于你的持仓给建议 |
| 告诉你发生了什么 | 告诉你该做什么 |

---

## Position Management (Phase 3)

```python
# src/portfolio/
# ├── positions.py     — Position CRUD + daily mark-to-market
# ├── analytics.py     — PnL, risk metrics, scenario exposure
# ├── tracker.py       — Recommendation tracking + performance
# └── advisor.py       — Portfolio-level advice generation

# Position 输入:
# 初始: 从 secrets.yml 导入 (META 2657@$771, GOOG 3306@$1996, etc)
# 更新: 手动 (API endpoint) 或通过 Mira ("我买了100股NVDA at $850")

# 每日自动:
# - 更新 market value
# - 计算 daily PnL (position-level + portfolio-level)
# - 检查 stop loss / target on tracked recommendations
# - 计算 scenario exposure (当前portfolio在各scenario下的表现)
# - 检测 concentration risk 变化
```

---

## Mira 集成 (Phase 4)

**轻量原则: Mira 不变重，Tetra 提供能力。**

### Tetra → Mira (每日推送)

```python
# 每日报告生成后:
# 1. 写入 Mira/artifacts/briefings/{date}_market.md (完整摘要)
# 2. 写入 Mira-bridge/inbox/ (200字精华 + PDF path)
# 3. iPhone 上能看到: "今天市场..., 建议..., portfolio PnL..."
```

### Mira → Tetra (查询)

```python
# 用户通过 Mira 问:
# "NVDA怎么样?" → Tetra 返回 narrative state + signals + position context
# "如果加息呢?" → Tetra 运行 forward scenario analysis
# "portfolio怎么样?" → Tetra 返回 PnL + risk + action items

# 实现: Mira analyst handler import Tetra modules (同机器，直接调用)
# 不需要 HTTP — 都在同一台Mac上
```

### 取代 Mira Analyst

```python
# 当前: Claude + 4个 skill markdown → 纯臆想
# 改为: Claude + Tetra real data → 有数据支撑的回答
#
# Mira/agents/analyst/handler.py 改为调用:
# from tetra.src.mira.handler import get_market_context
# context 包含: narrative state, signals, positions, recent report summary
```

---

## 调度设计

```
每日 (Eastern Time):

16:20  Data ingestion (market, events, news, fundamentals, polymarket)
       — 已有, 保留

17:00  Novel analysis pipeline (新)
       a. Narrative extraction (LLM batch process today's news)
       b. Information flow analysis (Polymarket vs rates vs equities)
       c. Insider pattern detection
       d. Analyst consensus shift analysis

17:30  Portfolio update
       - Position mark-to-market
       - Recommendation tracker update

18:00  LLM Adversarial Debate (新)
       - 3 LLMs with different information sets
       - 4 rounds

18:30  Report generation
       - Assemble analysis + debate results + portfolio status
       - Generate PDF
       - Email delivery
       - Push to Mira

每周六:
       - Narrative → market impact 统计回顾
       - Recommendation track record review
       - Information flow pattern analysis
       - Earnings network update (if in season)
```

---

## DB Schema 变更

```sql
-- 新增
CREATE SCHEMA IF NOT EXISTS narrative;
CREATE SCHEMA IF NOT EXISTS signals;
CREATE SCHEMA IF NOT EXISTS portfolio;
CREATE SCHEMA IF NOT EXISTS tracker;
CREATE SCHEMA IF NOT EXISTS report;
CREATE SCHEMA IF NOT EXISTS network;

-- narrative.daily_state (叙事检测)
-- date, scope (symbol/sector/market), dominant_narrative, shift_signal,
-- shift_magnitude, counter_narrative, novelty, historical_parallels, expected_impact

-- signals.information_flow (跨资产信息流)
-- date, source_market, target_market, gap_signal, gap_z_score, context

-- signals.polymarket_implied (Polymarket隐含信号)
-- date, market_id, implied_view, affected_assets, current_gap

-- network.company_graph (公司关系网络)
-- symbol, related_symbol, relation_type, strength

-- network.earnings_cascade (Earnings级联信号)
-- source_symbol, target_symbol, signal_type, magnitude, confidence, date

-- signals.informed_trading (知情交易信号)
-- symbol, date, signal_type, strength, historical_hit_rate, context

-- portfolio.positions (持仓)
-- symbol, shares, avg_cost, entry_date, current_price, market_value, weight

-- portfolio.snapshots (每日快照)
-- date, total_value, daily_return, positions_json

-- tracker.recommendations (建议追踪)
-- id, created_date, symbol, direction, entry, target, stop, status
-- method (narrative/info_flow/earnings_network/debate_consensus/debate_contrarian)
-- thesis, supporting_data, current_pnl, max_favorable, max_adverse

-- report.daily (报告存储)
-- date, pdf_path, summary, regime, portfolio_pnl, sections_json
```

### 要删的因子

```sql
-- 删除全部技术分析因子
DELETE FROM factors.daily_factors WHERE factor LIKE 'mkt.rsi%';
DELETE FROM factors.daily_factors WHERE factor LIKE 'mkt.sma%';
DELETE FROM factors.daily_factors WHERE factor LIKE 'mkt.ma_cross%';
DELETE FROM factors.daily_factors WHERE factor LIKE 'mkt.momentum%';
DELETE FROM factors.daily_factors WHERE factor LIKE 'mkt.breakout%';
DELETE FROM factors.daily_factors WHERE factor LIKE 'mkt.drawdown%';
-- 保留: mkt.return_*, mkt.vol_*, mkt.volume_z_* (原始数据，不是分析)
```

---

## 实施路线

### Phase 0: 清理合并 (1周)
- [ ] Minutes LLM clients + PDF/email → tetra/src/report/
- [ ] 合并 config/secrets
- [ ] 注册 LaunchAgent
- [ ] 删除 commodity factors
- [ ] 加 Anthropic key
- [ ] 新 DB schema

### Phase 1: Narrative + Information Flow (2周)
- [ ] Narrative extraction pipeline (LLM batch news analysis)
- [ ] Polymarket signal mapping (合约 → 资产影响)
- [ ] Cross-asset divergence detection
- [ ] 基础报告生成 (Methods 1+2 输出 → PDF)

### Phase 2: Earnings Network + Informed Trading (2周)
- [ ] Company relationship graph (LLM one-time build)
- [ ] Earnings cascade signal extraction
- [ ] Insider pattern analysis
- [ ] Analyst consensus shift detection
- [ ] 报告增强 (加入 Methods 3+4)

### Phase 3: Adversarial Debate + Portfolio (2周)
- [ ] 重新设计 debate (information asymmetry)
- [ ] Position tracking + daily mark-to-market
- [ ] Recommendation tracker
- [ ] Portfolio-specific report section
- [ ] Forward scenario analysis

### Phase 4: Mira Integration + Polish (1周)
- [ ] Daily push to Mira-bridge
- [ ] Analyst handler replacement
- [ ] Track record analytics
- [ ] 报告模板打磨

---

## Token 预算

| Method | 每日 Token Cost |
|--------|---------------|
| Narrative extraction | $5-15 |
| Information flow analysis | $3-8 |
| Adversarial debate | $15-25 |
| Earnings cascade (季报期) | $10-20 per week |
| Insider/analyst analysis | $2-5 |
| Scenario analysis | $5-10 |
| Mira queries (ad hoc) | $1-5 |
| **Total daily** | **~$30-65** |
| **Total monthly** | **~$900-2000** |

对于 $10M portfolio，每月 $1-2K 的分析成本 < 0.02%。如果能产生哪怕 0.1% 的 alpha，就是 $10K/month。

---

## 设计原则

1. **Novel > Commodity**: 如果别人能免费得到同样的分析，我们不做
2. **Data advantage**: 我们的 edge 是同时消化 30K 新闻 + 10K Polymarket + 59 macro series + insider data
3. **LLM as analyst, not narrator**: LLM 做分析和判断，不是写报告
4. **Track everything**: 不追踪的建议等于废话
5. **Portfolio-aware**: 所有分析最终落到 "你该做什么"
6. **Spend tokens wisely**: 大量 token 花在 novel analysis 上，不花在 formatting 上
