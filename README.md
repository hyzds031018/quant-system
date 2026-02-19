# 智能量化投研平台 (Intelligent Quant Research Platform)

## 1. 系统概述
本系统是一个集成了**实时行情、量化策略回测、AI价格预测、市场情绪分析**于一体的综合性智能投研平台。旨在为投资者提供基于数据驱动的决策支持，通过深度学习模型和经典量化理论，全方位分析市场动态。

---

## 2. 核心模块与原理 (Core Principles)

本系统不仅展示数据，更注重对数据的深度挖掘与解释。以下是三大核心分析模块的实现原理：

### 2.1 量化回测系统 (Quantitative Backtesting)
平台内置了多种经典量化交易策略，支持用户自定义参数进行历史回测，以验证策略的有效性。

*   **均线交叉策略 (Moving Average Crossover)**
    *   **原理**：利用长短周期的移动平均线（MA）来捕捉趋势。
    *   **交易信号**：
        *   **金叉 (Golden Cross)**：当短期均线（如MA20）由下向上穿过长期均线（如MA50）时，视为买入信号。意味着短期趋势走强。
        *   **死叉 (Death Cross)**：当短期均线由上向下穿过长期均线时，视为卖入信号。意味着短期趋势走弱。
    *   **应用**：适用于趋势明显的市场，震荡市中可能出现频繁的虚假信号。

*   **投资组合优化 (Portfolio Optimization)**
    *   **原理**：基于**现代投资组合理论 (Modern Portfolio Theory, MPT)**，即马科维茨均值-方差模型 (Markowitz Mean-Variance Model)。
    *   **核心目标**：在给定的风险水平下最大化预期收益，或在给定的收益目标下最小化风险。
    *   **算法实现**：
        1.  计算多只股票的历史收益率协方差矩阵（衡量资产间的相关性）。
        2.  通过数值优化方法（如SLSQP），求解使得**夏普比率 (Sharpe Ratio)** 最大的权重组合。
    *   **意义**：通过分散投资相关性低的资产，降低非系统性风险。

*   **波动率目标策略 (Volatility Targeting)**
    *   **原理**：根据市场波动率动态调整仓位，以保持组合的风险水平恒定。
    *   **实现**：
        1.  使用**指数加权移动平均 (EWMA)** 预测下一日的波动率。
        2.  计算目标仓位 = 目标波动率 / 预测波动率。
        3.  当波动率上升时，降低仓位（去杠杆）；波动率下降时，增加仓位（加杠杆）。

### 2.2 市场情绪分析 (Sentiment Analysis)
利用自然语言处理 (NLP) 技术，从海量财经新闻中提取市场情绪，作为量化数据的补充。

*   **数据源**：实时接入 **Finnhub** 全球财经新闻流。
*   **分析方法**：**词典匹配法 (Lexicon-based Approach)**。
*   **计算逻辑**：
    1.  **构建情感词典**：包含金融领域的正向词汇（如 surge, gain, profit, bull）和负向词汇（如 drop, loss, miss, bear）。
    2.  **文本扫描**：对每条新闻的标题和摘要进行分词和匹配。
    3.  **情绪打分**：正向词+1分，负向词-1分，根据得分归一化到 [-1, 1] 区间。
    4.  **聚合分析**：计算特定时间窗口内的平均情绪得分和新闻热度（Volume），生成“看多 (Bullish)”、“看空 (Bearish)”或“中性 (Neutral)”的评级。

### 2.3 风险评估体系 (Risk Assessment)
多维度量化指标，帮助投资者识别潜在风险。

*   **年化波动率 (Annualized Volatility)**：收益率标准差的年化值，衡量资产价格变动的剧烈程度。值越高，风险越大。
*   **夏普比率 (Sharpe Ratio)**：`(组合收益率 - 无风险利率) / 波动率`。衡量每承担一单位风险所获得的超额回报。大于1通常被认为是优秀的。
*   **最大回撤 (Max Drawdown)**：在选定周期内，资产价格从最高点跌至最低点的最大跌幅。衡量极端的下行风险。
*   **Beta 系数**：衡量个股相对于大盘（如S&P 500）的敏感度。
    *   Beta > 1: 波动比大盘大（高风险高收益）。
    *   Beta < 1: 波动比大盘小（防御性）。

---

## 3. 功能特性 (Features)

### 📊 市场与数据
- **实时看板**：展示大盘指数、涨跌幅榜单、市场热点。
- **交互式K线**：集成 ECharts，支持缩放、平移，叠加 AI 预测曲线和技术指标。
- **多维度数据**：包含 RSI, MACD, ATR, Stochastic 等主流技术指标。

### 🤖 AI 增强
- **价格预测**：基于 LSTM/Transformer 深度学习模型，预测未来 7 天股价趋势。
- **置信区间**：提供预测价格的 95% 置信区间，量化预测的不确定性。

### 📰 资讯流
- **实时新闻**：聚合个股新闻与宏观市场新闻。
- **智能标签**：自动为每条新闻打上情感标签（Bullish/Bearish）。

---

## 4. 技术架构
- **后端**：Python Flask (提供 RESTful API)
- **数据**：Tiingo API (行情), Finnhub API (新闻), Pandas (数据清洗)
- **前端**：HTML5, CSS3 (Grid Layout), JavaScript (ES6+)
- **可视化**：Apache ECharts
- **模型**：PyTorch / TensorFlow (离线训练), Scipy (组合优化)

## 5. 快速开始
1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```
2. **配置 API Key** (在环境变量中设置 `TIINGO_TOKEN` 和 `FINNHUB_TOKEN`)。
3. **启动服务**：
   ```bash
   python app.py
   ```
4. **访问**：打开浏览器访问 `http://127.0.0.1:5000`。

---

## 6. Deployment (Render free tier)

Recommended free deployment target: Render Web Service.

1. Push this repo to GitHub.
2. Create a new Web Service on Render and connect the repo.
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn app:app`
5. Set environment variables below.

Note: Render free instances sleep after inactivity and may cold-start on the first request.

## 7. Environment variables

Required for login:

- `APP_SECRET`: session secret (random long string)
- `APP_USER`: login username (default `admin`)
- `APP_PASSWORD` or `APP_PASSWORD_HASH`: login password (hash recommended)
- `ENABLE_LOGIN`: set `1` (default) to enable login, `0` to disable
- `SESSION_COOKIE_SECURE`: set `1` on HTTPS deployments

Data / API keys:

- `FINNHUB_TOKEN`: news API key
- `TIINGO_TOKEN`: market data API key (if you use Tiingo)

Optional:

- `AUTH_FILE`: file path for storing password hash (default `.auth.json`)

## 8. Admin page

After login, open `/admin` to change password online.

Behavior:

- If `APP_PASSWORD_HASH` is set, online change is disabled (update the env var instead).
- If `APP_PASSWORD` is set, you can change it online; the new hash is saved into `AUTH_FILE`.
- The file is local to the server. On free-tier deployments, it may be reset after redeploy.

---

## 9. Deployment (Cloud Run)

Cloud Run can build directly from source and deploy with one command.

### 9.1 Prereqs

1. Install Google Cloud CLI and sign in.
2. Set your project and enable APIs.

Example:

```
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

### 9.2 Deploy from source

From the repo root:

```
gcloud run deploy quant-system \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars APP_SECRET=YOUR_SECRET,APP_USER=admin,APP_PASSWORD=YOUR_PASS,FINNHUB_TOKEN=YOUR_FINNHUB,SESSION_COOKIE_SECURE=1
```

Notes:

- `--allow-unauthenticated` makes the service public; the app still requires login.
- `--set-env-vars` replaces any previous env vars, so include all values every deploy.

### 9.3 Free tier

Cloud Run includes a free tier of requests, CPU, and memory per month. See Google Cloud Free Program limits for Cloud Run.

