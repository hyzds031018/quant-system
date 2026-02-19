
            // Minimal jQuery-like fallback if CDN fails
            if (typeof window.$ === 'undefined') {
                (function() {
                    function MiniQuery(elements) {
                        this.elements = elements || [];
                    }
                    MiniQuery.prototype.html = function(content) {
                        if (content === undefined) {
                            return this.elements[0] ? this.elements[0].innerHTML : '';
                        }
                        this.elements.forEach(el => { el.innerHTML = content; });
                        return this;
                    };
                    MiniQuery.prototype.text = function(content) {
                        if (content === undefined) {
                            return this.elements[0] ? this.elements[0].textContent : '';
                        }
                        this.elements.forEach(el => { el.textContent = content; });
                        return this;
                    };
                    MiniQuery.prototype.empty = function() {
                        this.elements.forEach(el => { el.innerHTML = ''; });
                        return this;
                    };
                    MiniQuery.prototype.append = function(html) {
                        this.elements.forEach(el => { el.insertAdjacentHTML('beforeend', html); });
                        return this;
                    };
                    MiniQuery.prototype.on = function(evt, handler) {
                        this.elements.forEach(el => { el.addEventListener(evt, handler); });
                        return this;
                    };
                    MiniQuery.prototype.find = function(selector) {
                        let found = [];
                        this.elements.forEach(el => {
                            found = found.concat(Array.from(el.querySelectorAll(selector)));
                        });
                        return new MiniQuery(found);
                    };
                    function $(selector) {
                        if (selector === window || selector === document) {
                            return new MiniQuery([selector]);
                        }
                        if (typeof selector === 'string') {
                            return new MiniQuery(Array.from(document.querySelectorAll(selector)));
                        }
                        if (selector instanceof Element) {
                            return new MiniQuery([selector]);
                        }
                        if (selector && selector.length) {
                            return new MiniQuery(Array.from(selector));
                        }
                        return new MiniQuery([]);
                    }
                    $.ajax = function(options) {
                        const url = options.url;
                        const method = (options.method || 'GET').toUpperCase();
                        fetch(url, { method })
                            .then(res => {
                                if (!res.ok) throw new Error(res.status);
                                return res.json();
                            })
                            .then(data => options.success && options.success(data))
                            .catch(err => options.error && options.error(err));
                    };
                    window.$ = $;
                })();
            }
        </script>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #adb5bd 0%, #adb5bd 100%);
                color: #333;
                min-height: 100vh;
            }

            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }

            .header {
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }

            .header h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }

            .header .subtitle {
                font-size: 1.1rem;
                opacity: 0.9;
            }

            .controls {
                background: rgba(255,255,255,0.95);
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
            }

            .stock-selector {
                display: flex;
                align-items: center;
                gap: 15px;
                flex-wrap: wrap;
            }

            .stock-selector label {
                font-weight: 600;
                color: #333;
            }

            .stock-selector select {
                padding: 12px 20px;
                border: 2px solid #e1e5e9;
                border-radius: 8px;
                font-size: 16px;
                background: white;
                color: #333;
                cursor: pointer;
                transition: all 0.3s ease;
                min-width: 200px;
            }

            .stock-selector select:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }

            .refresh-btn {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            .refresh-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }

            .dashboard {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                grid-template-areas: 
                    "overview overview overview"
                    "mainchart mainchart mainchart"
                    "tech tech predtable"
                    "metrics metrics metrics"
                    "sentiment risk stockinfo";
                gap: 20px;
                margin-bottom: 20px;
            }

            .card {
                background: rgba(255,255,255,0.95);
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            }

            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 40px rgba(0,0,0,0.15);
            }

            .card-header {
                display: flex;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 2px solid #f0f0f0;
            }

            .card-header i {
                font-size: 1.5rem;
                margin-right: 10px;
                color: #667eea;
            }

            .card-header h3 {
                font-size: 1.3rem;
                color: #333;
                font-weight: 600;
            }

            .chart-wrapper {
                width: 100%;
                height: calc(100% - 60px);
                min-height: 600px;
            }

            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }

            .stat-item {
                background: linear-gradient(45deg, #f8f9fa, #e9ecef);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                transition: all 0.3s ease;
            }

            .stat-item:hover {
                transform: scale(1.05);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }

            .stat-value {
                font-size: 1.8rem;
                font-weight: bold;
                color: #333;
                margin-bottom: 5px;
            }

            .stat-label {
                font-size: 0.9rem;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            .sentiment-indicator {
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            .sentiment-bullish {
                background: linear-gradient(45deg, #28a745, #20c997);
                color: white;
            }

            .sentiment-bearish {
                background: linear-gradient(45deg, #dc3545, #fd7e14);
                color: white;
            }

            .sentiment-neutral {
                background: linear-gradient(45deg, #6c757d, #adb5bd);
                color: white;
            }

            .risk-level {
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
                display: inline-block;
            }

            .risk-low {
                background: #d4edda;
                color: #155724;
            }

            .risk-medium {
                background: #fff3cd;
                color: #856404;
            }

            .risk-high {
                background: #f8d7da;
                color: #721c24;
            }

            .loading {
                display: flex;
                align-items: center;
                justify-content: center;
                height: 200px;
                font-size: 1.2rem;
                color: #666;
            }

            .loading i {
                margin-right: 10px;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .full-width {
                grid-column: 1 / -1;
            }

            .prediction-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }

            .prediction-table th,
            .prediction-table td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #e9ecef;
            }

            .prediction-table th {
                background: #f8f9fa;
                font-weight: 600;
                color: #333;
            }

            .prediction-table tr:hover {
                background: #f8f9fa;
            }

            @media (max-width: 768px) {
                .container {
                    padding: 10px;
                }
                
                .dashboard {
                    grid-template-columns: 1fr;
                }
                
                .header h1 {
                    font-size: 2rem;
                }
                
                .stats-grid {
                    grid-template-columns: repeat(2, 1fr);
                }
            }

            .overview-panel {
                background: rgba(255,255,255,0.95);
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
                margin-bottom: 20px;
            }

            .chart-panel {
                background: rgba(255,255,255,0.95);
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
                margin-bottom: 20px;
            }

            .main-chart-area {
                width: 70%;
                margin: 20px auto;
                height: 500px;
            }

            .indicator-item {
                margin-bottom: 15px;
                padding: 10px;
                border: 1px solid #333;
                border-radius: 5px;
            }

            .indicator-item h4 {
                margin-top: 0;
                font-size: 1em;
            }

            #marketOverviewSection { grid-area: overview; }
            #kLineChartContainer { grid-area: mainchart; height: 700px; background: rgba(255,255,255,0.95); border-radius: 15px; padding: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); backdrop-filter: blur(10px); }
            #technicalChartPanel { grid-area: tech; }
            #predictionPanel { grid-area: predtable; }
            #sentimentPanel { grid-area: sentiment; }
            #riskPanel { grid-area: risk; }
            #atrChartWrapper { grid-area: newindicators1; }
            #stochChartWrapper { grid-area: newindicators2; }
            #willrObvWrapper { grid-area: newindicators3; }
            #modelEvaluationWrapper { grid-area: modelval; }
            #stockInfoPanel { grid-area: stockinfo; }
            #modelMetricsPanel { grid-area: metrics; }

            .loading-small {
                display: flex;
                align-items: center;
                justify-content: center;
                height: 120px;
                font-size: 0.9rem;
                color: #666;
            }

            .loading-small i {
                margin-right: 8px;
                animation: spin 1s linear infinite;
            }

            .technical-indicators-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                padding: 15px;
            }

            .chart-container {
                height: 250px;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 10px;
                padding: 10px;
            }

            .stock-info-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 10px;
                margin-top: 15px;
            }

            .info-item {
                padding: 8px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
            }

            .info-label {
                font-size: 0.8rem;
                color: #666;
                margin-bottom: 4px;
            }

            .info-value {
                font-size: 1rem;
                font-weight: 600;
                color: #333;
            }

            .company-description {
                grid-column: span 2;
                margin-top: 10px;
                padding: 10px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
                font-size: 0.9rem;
                line-height: 1.4;
            }

            .prediction-content {
                padding: 15px;
            }

            .prediction-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
                background: white;
            }

            .prediction-table th,
            .prediction-table td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #e9ecef;
            }

            .prediction-table th {
                background: #f8f9fa;
                font-weight: 600;
                color: #333;
            }

            .prediction-table tr:hover {
                background: #f8f9fa;
            }
        </style>
    </head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-chart-line"></i> Intelligent Stock Analysis System</h1>
            <p class="subtitle">Deep Learning Based Stock Technical Analysis and Risk Assessment Platform</p>
        </div>

        <div class="controls">
            <div class="stock-selector">
                <label for="stockSelect"><i class="fas fa-search"></i> Select Stock:</label>
                <select id="stockSelect">
                    <option value="">Please select stock code...</option>
                </select>
            </div>
        </div>

        <div class="dashboard">
            <div id="marketOverviewSection" class="overview-panel">
                <!-- 甯傚満姒傝 -->
            </div>
            <div id="kLineChartContainer" class="card">
                <div class="card-header">
                    <i class="fas fa-chart-line"></i>
                    <h3>K-Line Chart & Prediction</h3>
                </div>
                <div class="chart-wrapper">
                    <div class="loading">
                        <i class="fas fa-spinner"></i>
                        Please select a stock...
                    </div>
                </div>
            </div>

            <!-- 鎶€鏈寚鏍?-->
            <div id="technicalChartPanel" class="card">
                <div class="card-header">
                    <i class="fas fa-chart-area"></i>
                    <h3>Technical Indicators</h3>
                </div>
                <div class="technical-indicators-grid">
                    <div class="chart-container" id="priceChart">
                        <div class="loading">
                            <i class="fas fa-spinner"></i>
                            Loading...
                        </div>
                    </div>
                    <div class="chart-container" id="rsiChart">
                        <div class="loading">
                            <i class="fas fa-spinner"></i>
                            Loading...
                        </div>
                    </div>
                    <div class="chart-container" id="atrChart">
                        <div class="loading">
                            <i class="fas fa-spinner"></i>
                            Loading...
                        </div>
                    </div>
                    <div class="chart-container" id="stochChart">
                        <div class="loading">
                            <i class="fas fa-spinner"></i>
                            Loading...
                        </div>
                    </div>
                </div>
            </div>

            <!-- 妯″瀷棰勬祴 -->
            <div id="predictionPanel" class="card">
                <div class="card-header">
                    <i class="fas fa-crystal-ball"></i>
                    <h3>Price Prediction</h3>
                </div>
                <div class="prediction-content">
                    <div id="predictionChart" style="height: 300px; margin-bottom: 20px;"></div>
                    <div id="predictionTable"></div>
                </div>
            </div>

            <!-- 妯″瀷鎸囨爣 -->
            <div id="modelMetricsPanel" class="card">
                <div class="card-header">
                    <i class="fas fa-chart-bar"></i>
                    <h3>Model Metrics</h3>
                </div>
                <div id="modelMetricsContent">
                    <div class="loading">
                        <i class="fas fa-spinner"></i>
                        Loading metrics...
                    </div>
                </div>
                <div id="modelInfoContent" style="margin-top: 15px;">
                    <div class="loading">
                        <i class="fas fa-spinner"></i>
                        Loading model info...
                    </div>
                </div>
            </div>

            <!-- 鎯呮劅鍒嗘瀽 -->
            <div id="sentimentPanel" class="card">
                <div class="card-header">
                    <i class="fas fa-heart"></i>
                    <h3>Market Sentiment</h3>
                </div>
                <div id="sentimentContent">
                    <div class="loading">
                        <i class="fas fa-spinner"></i>
                        Analyzing...
                    </div>
                </div>
            </div>

            <!-- 椋庨櫓璇勪及 -->
            <div id="riskPanel" class="card">
                <div class="card-header">
                    <i class="fas fa-shield-alt"></i>
                    <h3>Risk Assessment</h3>
                </div>
                <div id="riskContent">
                    <div class="loading">
                        <i class="fas fa-spinner"></i>
                        Calculating...
                    </div>
                </div>
            </div>

            <!-- 鑲＄エ淇℃伅 -->
            <div id="stockInfoPanel" class="card">
                <div class="card-header">
                    <i class="fas fa-info-circle"></i>
                    <h3>Stock Information</h3>
                </div>
                <div id="stockInfoContent">
                    <div class="loading">
                        <i class="fas fa-spinner"></i>
                        Loading stock info...
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentStock = '';
        let charts = {};

        // 初始化
        $(document).ready(function() {
            loadStockList();
            loadMarketOverview();
            
            // Auto refresh market overview every 5 minutes
            setInterval(loadMarketOverview, 5 * 60 * 1000);
        });

        // 鍔犺浇鑲＄エ鍒楄〃
        function loadStockList() {
            $.ajax({
                url: '/api/stocks',
                method: 'GET',
                success: function(data) {
                    const select = $('#stockSelect');
                    select.empty().append('<option value="">Please select stock code...</option>');
                    data.forEach(stock => {
                        select.append(`<option value="${stock}">${stock}</option>`);
                    });
                    
                    select.on('change', function() {
                        const selectedStock = $(this).val();
                        if (selectedStock) {
                            currentStock = selectedStock;
                            // Clear old charts and content areas
                            clearAllChartsAndContent();
                            loadAllDataForStock(selectedStock);
                        } else {
                            currentStock = '';
                            clearAllChartsAndContent();
                        }
                    });
                },
                error: function() {
                    console.error('Failed to load stock list');
                    $('#stockSelect').empty().append('<option value="">Failed to load list</option>');
                }
            });
        }

        function clearAllChartsAndContent() {
            console.log('Clearing all charts and content');
            // Destroy ECharts instances
            Object.keys(charts).forEach(key => {
                if (charts[key]) {
                    console.log(`Destroying chart: ${key}`);
                    charts[key].dispose();
                    charts[key] = null;
                }
            });
            charts = {}; // Reset charts object

            // Clear chart containers and content areas
            $('#kLineChartContainer .chart-wrapper').html('<div class="loading"><i class="fas fa-spinner"></i> Please select a stock...</div>');
            $('#technicalChart').html('<div class="loading"><i class="fas fa-spinner"></i> Waiting for selection...</div>');
            $('#predictionChart').html('<div class="loading"><i class="fas fa-spinner"></i> Loading prediction chart...</div>');
            $('#predictionTable').html('<div class="loading-small"><i class="fas fa-spinner"></i> Loading prediction data...</div>');
            $('#modelMetricsContent').html('<div class="loading"><i class="fas fa-spinner"></i> Loading metrics...</div>');
            $('#modelInfoContent').html('<div class="loading"><i class="fas fa-spinner"></i> Loading model info...</div>');
            $('#sentimentContent').html('<div class="loading"><i class="fas fa-spinner"></i> Analyzing...</div>');
            $('#riskContent').html('<div class="loading"><i class="fas fa-spinner"></i> Calculating...</div>');
            $('#stockInfoContent').html('<div class="loading"><i class="fas fa-spinner"></i> Loading stock info...</div>');
            
            console.log('Cleanup completed');
        }

        // 主数据加载函数
        function loadAllDataForStock(stockCode) {
            console.log(`Starting to load all data: ${stockCode}`);
            loadAndRenderCombinedKlinePrediction(stockCode);
            loadAndRenderTechnicalIndicators(stockCode);
            loadSentimentAnalysis(stockCode);
            loadRiskAssessment(stockCode);
            loadAndRenderModelEvaluation(stockCode);
            loadPredictionTable(stockCode);
            loadStockBasicInfo(stockCode);
            loadModelMetrics(stockCode);
            loadModelInfo(stockCode);
        }
        
        // 鍔犺浇骞舵覆鏌撳悎骞剁殑K绾垮浘鍜岄娴嬪浘
        async function loadAndRenderCombinedKlinePrediction(stockCode) {
            console.log(`Starting to load K-line chart: ${stockCode}`);
            const chartContainer = document.querySelector('#kLineChartContainer .chart-wrapper');
            
            if (!chartContainer) {
                console.error('Chart container not found');
                return;
            }
            
            console.log('Chart container dimensions:', chartContainer.offsetWidth, chartContainer.offsetHeight);
            chartContainer.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i> Loading K-line and prediction data...</div>';

            try {
                console.log('Starting data request');
                const [kLineResponse, predictionResponse, futureResponse] = await Promise.all([
                    fetch(`/api/stock/${stockCode}`),
                    fetch(`/api/prediction/${stockCode}`),
                    fetch(`/api/future/${stockCode}?days=7&ci=0.95`)
                ]);

                if (!kLineResponse.ok) throw new Error(`K-Line API Error: ${kLineResponse.status}`);
                if (!predictionResponse.ok) throw new Error(`Prediction API Error: ${predictionResponse.status}`);
                if (!futureResponse.ok) throw new Error(`Future API Error: ${futureResponse.status}`);

                const kLineData = await kLineResponse.json();
                const predictionData = await predictionResponse.json();
                const futureData = await futureResponse.json();

                console.log('K-line data sample:', kLineData.kline?.slice(0, 2));
                console.log('Prediction data sample:', predictionData?.slice(0, 2));

                if (!kLineData.success || !kLineData.kline || kLineData.kline.length === 0) {
                    throw new Error(kLineData.error || 'No K-line data');
                }

                // Prepare K-line data
                const data = kLineData.kline;
                console.log(`Processing ${data.length} K-line data points`);
                
                // Create new chart container
                chartContainer.innerHTML = '<div id="kLineChart" style="width: 100%; height: 100%;"></div>';
                const chartDiv = document.getElementById('kLineChart');
                
                if (!chartDiv) {
                    console.error('kLineChart element not found');
                    return;
                }
                
                console.log('Chart container ready');
                charts.combinedKlinePrediction = echarts.init(chartDiv);
                
                // ??K???
                const dates = data.map(item => item.time);
                const volumes = data.map(item => parseFloat(item.volume));
                const candlestickData = data.map(item => [
                    parseFloat(item.open),
                    parseFloat(item.close),
                    parseFloat(item.low),
                    parseFloat(item.high)
                ]);
                const ma5Data = data.map(item => item.ma5);
                const ma10Data = data.map(item => item.ma10);
                const ma20Data = data.map(item => item.ma20);

                // 鍑嗗棰勬祴鏁版嵁
                const ensemblePredictions = data.map(item => item.predicted?.ensemble);
                const lstmPredictions = data.map(item => item.predicted?.attention_lstm);
                const gruPredictions = data.map(item => item.predicted?.gru);
                const transformerPredictions = data.map(item => item.predicted?.transformer);

                let allDates = dates;
                let allVolumes = volumes;
                let allCandles = candlestickData;
                let allMa5 = ma5Data;
                let allMa10 = ma10Data;
                let allMa20 = ma20Data;
                let allEnsemble = ensemblePredictions;
                let allLower = new Array(dates.length).fill(null);
                let allUpper = new Array(dates.length).fill(null);
                let allLstm = lstmPredictions;
                let allGru = gruPredictions;
                let allTransformer = transformerPredictions;
                if (futureData && futureData.success && futureData.dates && futureData.dates.length > 0) {
                    const futureCount = futureData.dates.length;
                    allDates = dates.concat(futureData.dates);
                    allVolumes = volumes.concat(new Array(futureCount).fill(null));
                    allCandles = candlestickData.concat(new Array(futureCount).fill([null, null, null, null]));
                    allMa5 = ma5Data.concat(new Array(futureCount).fill(null));
                    allMa10 = ma10Data.concat(new Array(futureCount).fill(null));
                    allMa20 = ma20Data.concat(new Array(futureCount).fill(null));
                    allEnsemble = ensemblePredictions.concat(futureData.ensemble || new Array(futureCount).fill(null));
                    allLower = allLower.concat(futureData.lower || new Array(futureCount).fill(null));
                    allUpper = allUpper.concat(futureData.upper || new Array(futureCount).fill(null));
                    allLstm = lstmPredictions.concat((futureData.models && futureData.models.attention_lstm) || new Array(futureCount).fill(null));
                    allGru = gruPredictions.concat((futureData.models && futureData.models.gru) || new Array(futureCount).fill(null));
                    allTransformer = transformerPredictions.concat((futureData.models && futureData.models.transformer) || new Array(futureCount).fill(null));
                }

                // 璁剧疆鍥捐〃閫夐」
                const option = {
                    animation: false,
                    tooltip: {
                        trigger: 'axis',
                        axisPointer: {
                            type: 'cross'
                        },
                        formatter: function(params) {
                            const date = params[0].axisValue;
                            let result = `${date}<br/>`;
                            params.forEach(param => {
                                if (param.seriesName === 'K-Line') {
                                    const data = param.data;
                                    result += `Open: ${data[0]?.toFixed(2)}<br/>`;
                                    result += `Close: ${data[1]?.toFixed(2)}<br/>`;
                                    result += `Low: ${data[2]?.toFixed(2)}<br/>`;
                                    result += `High: ${data[3]?.toFixed(2)}<br/>`;
                                } else if (param.seriesName === 'Volume') {
                                    result += `${param.seriesName}: ${param.data?.toLocaleString()}<br/>`;
                                } else if (param.data !== null) {
                                    result += `${param.seriesName}: ${param.data?.toFixed(2)}<br/>`;
                                }
                            });
                            return result;
                        }
                    },
                    legend: {
                        data: ['K-Line', 'MA5', 'MA10', 'MA20', 'Ensemble', 'CI Upper', 'CI Lower', 'LSTM', 'GRU', 'Transformer', 'Volume'],
                        top: '3%'
                    },
                    grid: [{
                        left: '10%',
                        right: '8%',
                        top: '8%',
                        height: '60%'
                    }, {
                        left: '10%',
                        right: '8%',
                        top: '75%',
                        height: '15%'
                    }],
                    xAxis: [{
                        type: 'category',
                        data: allDates,
                        scale: true,
                        boundaryGap: false,
                        axisLine: { onZero: false },
                        splitLine: { show: false },
                        splitNumber: 20,
                        min: 'dataMin',
                        max: 'dataMax',
                        axisPointer: {
                            z: 100
                        }
                    }, {
                        type: 'category',
                        gridIndex: 1,
                        data: allDates,
                        scale: true,
                        boundaryGap: false,
                        axisLine: { onZero: false },
                        axisTick: { show: false },
                        splitLine: { show: false },
                        axisLabel: { show: false },
                        splitNumber: 20,
                        min: 'dataMin',
                        max: 'dataMax'
                    }],
                    yAxis: [{
                        scale: true,
                        splitArea: {
                            show: true
                        }
                    }, {
                        scale: true,
                        gridIndex: 1,
                        splitNumber: 2,
                        axisLabel: { show: false },
                        axisLine: { show: false },
                        axisTick: { show: false },
                        splitLine: { show: false }
                    }],
                    dataZoom: [{
                        type: 'inside',
                        xAxisIndex: [0, 1],
                        start: 50,
                        end: 100
                    }, {
                        show: true,
                        xAxisIndex: [0, 1],
                        type: 'slider',
                        bottom: '0%',
                        start: 50,
                        end: 100
                    }],
                    series: [{
                        name: 'K-Line',
                        type: 'candlestick',
                        data: allCandles,
                        itemStyle: {
                            color: '#ef232a',
                            color0: '#14b143',
                            borderColor: '#ef232a',
                            borderColor0: '#14b143'
                        }
                    }, {
                        name: 'MA5',
                        type: 'line',
                        data: allMa5,
                        smooth: true,
                        lineStyle: {
                            opacity: 0.5,
                            color: '#FF8C00'
                        }
                    }, {
                        name: 'MA10',
                        type: 'line',
                        data: allMa10,
                        smooth: true,
                        lineStyle: {
                            opacity: 0.5,
                            color: '#00BFFF'
                        }
                    }, {
                        name: 'MA20',
                        type: 'line',
                        data: allMa20,
                        smooth: true,
                        lineStyle: {
                            opacity: 0.5,
                            color: '#9400D3'
                        }
                    }, {
                        name: 'Ensemble',
                        type: 'line',
                        data: allEnsemble,
                        smooth: true,
                        lineStyle: {
                            opacity: 0.8,
                            width: 2,
                            color: '#1E90FF'
                        },
                        itemStyle: {
                            color: '#1E90FF'
                        }
                    }, {
                        name: 'CI Upper',
                        type: 'line',
                        data: allUpper,
                        smooth: true,
                        lineStyle: { opacity: 0 },
                        symbol: 'none',
                        stack: 'ci-band'
                    }, {
                        name: 'CI Lower',
                        type: 'line',
                        data: allLower,
                        smooth: true,
                        lineStyle: { opacity: 0 },
                        symbol: 'none',
                        stack: 'ci-band',
                        areaStyle: { color: 'rgba(30, 144, 255, 0.15)' }
                    }, {
                        name: 'LSTM',
                        type: 'line',
                        data: allLstm,
                        smooth: true,
                        lineStyle: {
                            opacity: 0.6,
                            width: 1,
                            type: 'dashed',
                            color: '#FF69B4'
                        }
                    }, {
                        name: 'GRU',
                        type: 'line',
                        data: allGru,
                        smooth: true,
                        lineStyle: {
                            opacity: 0.6,
                            width: 1,
                            type: 'dashed',
                            color: '#32CD32'
                        }
                    }, {
                        name: 'Transformer',
                        type: 'line',
                        data: allTransformer,
                        smooth: true,
                        lineStyle: {
                            opacity: 0.6,
                            width: 1,
                            type: 'dashed',
                            color: '#FFD700'
                        }
                    }, {
                        name: 'Volume',
                        type: 'bar',
                        xAxisIndex: 1,
                        yAxisIndex: 1,
                        data: allVolumes,
                        itemStyle: {
                            color: function(params) {
                                const i = params.dataIndex;
                                if (!data[i]) return '#999999'; return data[i].close > data[i].open ? '#ef232a' : '#14b143';
                            }
                        }
                    }]
                };

                console.log('Setting chart options');
                charts.combinedKlinePrediction.setOption(option);
                console.log('Chart rendering completed');

                // 娣诲姞鑷€傚簲
                window.addEventListener('resize', () => {
                    if (charts.combinedKlinePrediction) {
                        charts.combinedKlinePrediction.resize();
                    }
                });

            } catch (error) {
                console.error('Failed to load K-line and prediction data:', error);
                chartContainer.innerHTML = `<div class="loading">Loading failed: ${error.message}</div>`;
            }
        }
        
        // 加载并渲染所有技术指标
        async function loadAndRenderTechnicalIndicators(stockCode) {
            const priceChartContainer = document.getElementById('priceChart');
            const rsiChartContainer = document.getElementById('rsiChart');
            const atrChartContainer = document.getElementById('atrChart');
            const stochChartContainer = document.getElementById('stochChart');

            try {
                const response = await fetch(`/api/technical_indicators/${stockCode}`);
                if (!response.ok) throw new Error(`Technical Indicators API Error: ${response.status}`);
                const data = await response.json();

                if (data.error) {
                    throw new Error(data.error);
                }

                const dates = data.dates;
                const lastIndex = dates.length - 1;

                // 娓叉煋浠锋牸鍜岀Щ鍔ㄥ钩鍧囩嚎
                if (charts.price) charts.price.dispose();
                charts.price = echarts.init(priceChartContainer);
                charts.price.setOption({
                    title: { text: 'Price and Moving Averages', left: 'center', textStyle: { fontSize: 14 } },
                    tooltip: { trigger: 'axis' },
                    legend: { data: ['Close', 'MA5', 'MA10', 'MA20'], top: 25 },
                    grid: { left: '10%', right: '5%', top: '20%', bottom: '15%' },
                    xAxis: { type: 'category', data: dates.slice(-30), axisLabel: { rotate: 30 } },
                    yAxis: { type: 'value', scale: true },
                    series: [
                        { name: 'Close', type: 'line', data: data.price.close.slice(-30), lineStyle: { color: '#0D1B2A' } },
                        { name: 'MA5', type: 'line', data: data.price.ma5.slice(-30), lineStyle: { color: '#1B263B' } },
                        { name: 'MA10', type: 'line', data: data.price.ma10.slice(-30), lineStyle: { color: '#415A77' } },
                        { name: 'MA20', type: 'line', data: data.price.ma20.slice(-30), lineStyle: { color: '#778DA9' } }
                    ]
                });

                // 娓叉煋RSI
                if (charts.rsi) charts.rsi.dispose();
                charts.rsi = echarts.init(rsiChartContainer);
                charts.rsi.setOption({
                    title: { text: 'RSI Index', left: 'center', textStyle: { fontSize: 14 } },
                    tooltip: { trigger: 'axis' },
                    grid: { left: '10%', right: '5%', top: '20%', bottom: '15%' },
                    xAxis: { type: 'category', data: dates.slice(-30), axisLabel: { rotate: 30 } },
                    yAxis: { type: 'value', scale: true, min: 0, max: 100 },
                    series: [{
                        name: 'RSI',
                        type: 'line',
                        data: data.indicators.rsi.slice(-30),
                        lineStyle: { color: '#E63946' },
                        markLine: {
                            silent: true,
                            data: [
                                { yAxis: 70, lineStyle: { color: '#E63946' }, label: { formatter: 'Overbought' } },
                                { yAxis: 30, lineStyle: { color: '#2A9D8F' }, label: { formatter: 'Oversold' } }
                            ]
                        }
                    }]
                });

                // 娓叉煋ATR
                if (charts.atr) charts.atr.dispose();
                charts.atr = echarts.init(atrChartContainer);
                charts.atr.setOption({
                    title: { text: 'ATR (Volatility)', left: 'center', textStyle: { fontSize: 14 } },
                    tooltip: { trigger: 'axis' },
                    grid: { left: '10%', right: '5%', top: '20%', bottom: '15%' },
                    xAxis: { type: 'category', data: dates.slice(-30), axisLabel: { rotate: 30 } },
                    yAxis: { type: 'value', scale: true },
                    series: [{
                        name: 'ATR',
                        type: 'line',
                        data: data.indicators.atr.slice(-30),
                        lineStyle: { color: '#457B9D' },
                        areaStyle: { color: '#457B9D', opacity: 0.2 }
                    }]
                });

                // 娓叉煋STOCH
                if (charts.stoch) charts.stoch.dispose();
                charts.stoch = echarts.init(stochChartContainer);
                charts.stoch.setOption({
                    title: { text: 'Stochastic Oscillator', left: 'center', textStyle: { fontSize: 14 } },
                    tooltip: { trigger: 'axis' },
                    legend: { data: ['%K', '%D'], top: 25 },
                    grid: { left: '10%', right: '5%', top: '20%', bottom: '15%' },
                    xAxis: { type: 'category', data: dates.slice(-30), axisLabel: { rotate: 30 } },
                    yAxis: { type: 'value', scale: true, min: 0, max: 100 },
                    series: [
                        { 
                            name: '%K', 
                            type: 'line', 
                            data: data.indicators.stoch.k.slice(-30),
                            lineStyle: { color: '#E9C46A' }
                        },
                        { 
                            name: '%D', 
                            type: 'line', 
                            data: data.indicators.stoch.d.slice(-30),
                            lineStyle: { color: '#F4A261' }
                        }
                    ]
                });

            } catch (error) {
                console.error('Failed to load technical indicators:', error);
                const errorMsg = `<div class="loading">Loading failed: ${error.message}</div>`;
                priceChartContainer.innerHTML = errorMsg;
                rsiChartContainer.innerHTML = errorMsg;
                atrChartContainer.innerHTML = errorMsg;
                stochChartContainer.innerHTML = errorMsg;
            }
        }

        // 加载并渲染模型评估指标
        async function loadAndRenderModelEvaluation(stockCode) {
            const evalArea = $('#modelEvaluationWrapper');
            // 绠€鍗曠殑鍔犺浇鏂囨湰
            evalArea.find('span').text('鍔犺浇涓?..');

            try {
                const response = await fetch(`/api/model_evaluation/${stockCode}`);
                if (!response.ok) throw new Error(`Model Evaluation API Error: ${response.status}`);
                const data = await response.json();

                $('#arimaRmse').text(data.arima?.rmse || 'N/A');
                $('#arimaMae').text(data.arima?.mae || 'N/A');
                // 可选添加 MAPE 的展示
                // $('#arimaMape').text(data.arima?.mape || 'N/A'); 

                $('#lstmRmse').text(data.lstm_attention?.rmse || 'N/A');
                $('#lstmMae').text(data.lstm_attention?.mae || 'N/A');

                $('#transformerRmse').text(data.transformer?.rmse || 'N/A');
                $('#transformerMae').text(data.transformer?.mae || 'N/A');

            } catch (error) {
                console.error('Failed to load model evaluation:', error);
                evalArea.find('span').text('閿欒');
            }
        }
        
        // 鍔犺浇棰勬祴琛ㄦ牸 (鏃х殑棰勬祴鍥捐〃閫昏緫宸插悎骞跺埌K绾垮浘)
        function loadPredictionTable(stockCode) {
            console.log('Starting to load prediction data:', stockCode);
            $('#predictionTable').html('<div class="loading-small"><i class="fas fa-spinner"></i> Loading prediction list...</div>');
            $('#predictionChart').html('<div class="loading-small"><i class="fas fa-spinner"></i> Loading prediction chart...</div>');
            
            $.ajax({
                url: `/api/future/${stockCode}?days=7&ci=0.95`,
                method: 'GET',
                success: function(data) {
                    console.log('Prediction data loaded successfully:', data);
                    renderPredictionTable(data);
                    renderPredictionChart(data);
                },
                error: function(error) {
                    console.error('Failed to load prediction data:', error);
                    $('#predictionTable').html('<div class="loading-small">Failed to load prediction list</div>');
                    $('#predictionChart').html('<div class="loading-small">Failed to load prediction chart</div>');
                }
            });
        }

        function renderPredictionChart(data) {
            console.log('Starting to render prediction chart');
            if (!data || !data.success || !data.dates || data.dates.length === 0) {
                $('#predictionChart').html('<p style="text-align:center;">No prediction data</p>');
                return;
            }

            const chartDom = document.getElementById('predictionChart');
            // ?????????????
            if (charts.prediction) {
                charts.prediction.dispose();
            }
            
            // 纭繚瀹瑰櫒瀛樺湪涓旀湁灏哄
            if (!chartDom) {
                console.error('Prediction chart container not found');
                return;
            }

            console.log('Initializing prediction chart');
            charts.prediction = echarts.init(chartDom);

            const dates = data.dates;
            const prices = data.ensemble;
            const lower = data.lower;
            const upper = data.upper;

            console.log('Prediction data:', { dates, prices });

            const option = {
                title: {
                    text: '7-Day Price Prediction Trend',
                    left: 'center',
                    top: 10,
                    textStyle: {
                        fontSize: 14
                    }
                },
                tooltip: {
                    trigger: 'axis',
                    formatter: function(params) {
                        const date = params[0].axisValue;
                        const price = params[0].data;
                        return `${date}<br/>Predicted Price: $${price.toFixed(2)}`;
                    }
                },
                grid: {
                    left: '10%',
                    right: '5%',
                    top: '15%',
                    bottom: '15%'
                },
                xAxis: {
                    type: 'category',
                    data: allDates,
                    axisLabel: {
                        rotate: 30,
                        interval: 0
                    }
                },
                yAxis: {
                    type: 'value',
                    scale: true,
                    splitLine: {
                        show: true,
                        lineStyle: {
                            type: 'dashed'
                        }
                    }
                },
                series: [
                    {
                        name: 'Upper',
                        type: 'line',
                        data: upper,
                        lineStyle: { opacity: 0 },
                        stack: 'confidence-band',
                        symbol: 'none'
                    },
                    {
                        name: 'Lower',
                        type: 'line',
                        data: lower,
                        lineStyle: { opacity: 0 },
                        stack: 'confidence-band',
                        symbol: 'none',
                        areaStyle: {
                            color: 'rgba(30, 144, 255, 0.15)'
                        }
                    },
                    {
                        name: 'Predicted Price',
                        type: 'line',
                        data: prices,
                        smooth: true,
                        showSymbol: true,
                        symbolSize: 8,
                        lineStyle: {
                            width: 3,
                            color: '#1E90FF'
                        },
                        itemStyle: {
                            color: '#1E90FF',
                            borderWidth: 2
                        }
                    }
                ]
            };

            try {
                console.log('Setting prediction chart options');
                charts.prediction.setOption(option);
                console.log('Prediction chart rendering completed');
            } catch (error) {
                console.error('Failed to render prediction chart:', error);
                $('#predictionChart').html('<p style="text-align:center;">Chart rendering failed</p>');
            }
        }

        function renderPredictionTable(data) {
            if (!data || !data.success || !data.dates || data.dates.length === 0) {
                $('#predictionTable').html('<p style="text-align:center;">No prediction data available</p>');
                return;
            }
            let tableHtml = `
                <div style="margin: 15px 0;">
                    <table class="prediction-table">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Predicted Price</th>
                                <th>Lower</th>
                                <th>Upper</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
            data.dates.forEach((d, i) => {
                tableHtml += `
                    <tr>
                        <td>${d}</td>
                        <td>$${parseFloat(data.ensemble[i]).toFixed(2)}</td>
                        <td>$${parseFloat(data.lower[i]).toFixed(2)}</td>
                        <td>$${parseFloat(data.upper[i]).toFixed(2)}</td>
                    </tr>
                `;
            });
            tableHtml += '</tbody></table></div>';
            $('#predictionTable').html(tableHtml);
        }

        // ---- 鎮ㄥ凡鏈夌殑鍏朵粬鍔犺浇鍜屾覆鏌撳嚱鏁?----
        // loadSentimentAnalysis, renderSentimentAnalysis
        // loadRiskAssessment, renderRiskAssessment
        // loadMarketOverview, renderMarketOverview
        // refreshData
        // 瀹冧滑鍙互淇濇寔涓嶅彉锛岄櫎闈炴偍鎯充慨鏀瑰畠浠殑琛屼负鎴栧姞杞界姸鎬佹樉绀?
        function loadSentimentAnalysis(stockCode) {
            $('#sentimentContent').html('<div class="loading"><i class="fas fa-spinner"></i> Analyzing...</div>');
            $.ajax({
                url: `/api/sentiment/${stockCode}`,
                method: 'GET',
                success: function(data) {
                    renderSentimentAnalysis(data);
                },
                error: function() {
                    $('#sentimentContent').html('<div class="loading">鍔犺浇鎯呮劅鍒嗘瀽澶辫触</div>');
                }
            });
        }

        function renderSentimentAnalysis(data) {
            const sentimentClass = data.overall_sentiment.toLowerCase() === 'bullish' ? 'sentiment-bullish' :
                                 data.overall_sentiment.toLowerCase() === 'bearish' ? 'sentiment-bearish' : 'sentiment-neutral';
            const html = `
                <div class="sentiment-indicator ${sentimentClass}">
                    <i class="fas fa-chart-line"></i>
                    ${data.overall_sentiment} Sentiment
                </div>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value">${(data.confidence * 100).toFixed(1)}%</div>
                        <div class="stat-label">Confidence</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.news_volume && data.news_volume.length > 0 ? data.news_volume[data.news_volume.length-1] : 'N/A'}</div>
                        <div class="stat-label">News Volume</div>
                    </div>
                </div>
            `;
            $('#sentimentContent').html(html);
        }

        function loadRiskAssessment(stockCode) {
            $('#riskContent').html('<div class="loading"><i class="fas fa-spinner"></i> Calculating...</div>');
            $.ajax({
                url: `/api/risk_assessment/${stockCode}`,
                method: 'GET',
                success: function(data) {
                    renderRiskAssessment(data);
                },
                error: function() {
                    $('#riskContent').html('<div class="loading">鍔犺浇椋庨櫓璇勪及澶辫触</div>');
                }
            });
        }

        function renderRiskAssessment(data) {
            const riskClass = data.risk_level.toLowerCase() === 'high' ? 'risk-high' :
                             data.risk_level.toLowerCase() === 'medium' ? 'risk-medium' : 'risk-low';
            const html = `
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value">${data.volatility}%</div>
                        <div class="stat-label">Annual Volatility</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.sharpe_ratio}</div>
                        <div class="stat-label">Sharpe Ratio</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.beta}</div>
                        <div class="stat-label">Beta</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.max_drawdown}%</div>
                        <div class="stat-label">Max Drawdown</div>
                    </div>
                </div>
                <div style="text-align: center; margin-top: 15px;">
                    <span class="risk-level ${riskClass}">
                        Risk Level: ${data.risk_level}
                    </span>
                </div>
            `;
            $('#riskContent').html(html);
        }

        function loadMarketOverview() {
            $('#marketOverviewSection').html('<div class="loading"><i class="fas fa-spinner"></i> Loading market overview...</div>');
            $.ajax({
                url: '/api/market_overview',
                method: 'GET',
                success: function(data) {
                    if (data.error) {
                        $('#marketOverviewSection').html(`<div class="loading">Failed to load market overview: ${data.error}</div>`);
                        return;
                    }
                    renderMarketOverview(data);
                },
                error: function(xhr, status, error) {
                    console.error('Failed to load market overview:', error);
                    $('#marketOverviewSection').html('<div class="loading">Failed to load market overview</div>');
                }
            });
        }

        function renderMarketOverview(data) {
            const html = `
                <div class="card-header" style="margin-bottom: 10px;">
                    <i class="fas fa-globe-americas"></i>
                    <h3>Market Overview</h3>
                </div>
                <div class="stats-grid"> 
                    <div class="stat-item">
                        <div class="stat-value">$${parseFloat(data.highest_close_price).toFixed(2)}</div>
                        <div class="stat-label">${data.highest_close_stock} Highest Close</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">$${parseFloat(data.avg_close_price).toFixed(2)}</div>
                        <div class="stat-label">Market Average Close</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${parseFloat(data.highest_pct_change).toFixed(2)}%</div>
                        <div class="stat-label">${data.highest_pct_change_stock} Highest Change</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.latest_date}</div>
                        <div class="stat-label">Latest Trading Day</div>
                    </div>
                </div>
            `;
            $('#marketOverviewSection').html(html);
        }

        // 鍒锋柊鏁版嵁
        function refreshData() {
            if (currentStock) {
                // 显示全局加载提示或禁用按钮
                console.log("Refreshing data for " + currentStock);
                loadAllDataForStock(currentStock);
            }
            loadMarketOverview(); // 鎬绘槸鍒锋柊甯傚満姒傝
        }

        // 响应式图表
        $(window).on('resize', function() {
            Object.values(charts).forEach(chart => {
                if (chart && typeof chart.resize === 'function') {
                    try {
                        chart.resize();
                    } catch (e) {
                        console.warn("Error resizing chart:", e);
                    }
                }
            });
        });

        function loadStockBasicInfo(stockCode) {
            const container = document.getElementById('stockInfoContent');
            container.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i> Loading...</div>';

            fetch(`/api/stock_info/${stockCode}`)
                .then(response => response.json())
                .then(info => {
                    if (info.error) {
                        container.innerHTML = `<div class="loading">${info.error}</div>`;
                        return;
                    }

                    const html = `
                        <div class="stock-info-grid">
                            <div class="info-item">
                                <div class="info-label">Company Name</div>
                                <div class="info-value">${info.name || 'N/A'}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Industry</div>
                                <div class="info-value">${info.industry || 'N/A'}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Market Cap</div>
                                <div class="info-value">${info.market_cap || 'N/A'}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">P/E Ratio</div>
                                <div class="info-value">${info.pe_ratio || 'N/A'}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Dividend Yield</div>
                                <div class="info-value">${info.dividend_yield || 'N/A'}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">EPS</div>
                                <div class="info-value">${info.eps || 'N/A'}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Revenue</div>
                                <div class="info-value">${info.revenue || 'N/A'}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Employees</div>
                                <div class="info-value">${info.employees || 'N/A'}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Founded</div>
                                <div class="info-value">${info.founded || 'N/A'}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Headquarters</div>
                                <div class="info-value">${info.headquarters || 'N/A'}</div>
                            </div>
                            <div class="company-description">
                                <div class="info-label">Company Description</div>
                                <div class="info-value">${info.description || 'No description available.'}</div>
                            </div>
                        </div>
                    `;
                    container.innerHTML = html;
                })
                .catch(error => {
                    console.error('Failed to load stock info:', error);
                    container.innerHTML = '<div class="loading">Failed to load stock information</div>';
                });
        }

        function loadModelMetrics(stockCode) {
            const container = document.getElementById('modelMetricsContent');
            container.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i> Loading metrics...</div>';

            fetch(`/api/metrics/${stockCode}`)
                .then(response => response.json())
                .then(data => {
                    if (!data.success) {
                        container.innerHTML = `<div class="loading">Failed to load metrics: ${data.error || 'unknown error'}</div>`;
                        return;
                    }
                    const m = data.metrics || {};
                    const html = `
                        <div class="stats-grid">
                            <div class="stat-item">
                                <div class="stat-value">${(m.price_rmse_simple ?? 0).toFixed(4)}</div>
                                <div class="stat-label">RMSE Simple</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">${(m.price_rmse_weighted ?? 0).toFixed(4)}</div>
                                <div class="stat-label">RMSE Weighted</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">${(m.price_mae_simple ?? 0).toFixed(4)}</div>
                                <div class="stat-label">MAE Simple</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">${(m.price_mae_weighted ?? 0).toFixed(4)}</div>
                                <div class="stat-label">MAE Weighted</div>
                            </div>
                        </div>
                    `;
                    container.innerHTML = html;
                })
                .catch(error => {
                    container.innerHTML = `<div class="loading">Failed to load metrics: ${error}</div>`;
                });
        }

        function loadModelInfo(stockCode) {
            const container = document.getElementById('modelInfoContent');
            container.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i> Loading model info...</div>';

            fetch(`/api/model_info/${stockCode}`)
                .then(response => response.json())
                .then(data => {
                    if (!data.success) {
                        container.innerHTML = `<div class="loading">Failed to load model info: ${data.error || 'unknown error'}</div>`;
                        return;
                    }
                    const info = data.info || {};
                    const models = info.models || {};
                    const modelList = Object.keys(models)
                        .map(k => `<div class="info-item"><div class="info-label">${k}</div><div class="info-value">${models[k]}</div></div>`)
                        .join('');

                    const html = `
                        <div class="stock-info-grid">
                            <div class="info-item">
                                <div class="info-label">Sequence Length</div>
                                <div class="info-value">${info.sequence_length ?? 'N/A'}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Feature Count</div>
                                <div class="info-value">${info.feature_count ?? 'N/A'}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Target</div>
                                <div class="info-value">${info.target_col ?? 'N/A'}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Symbol</div>
                                <div class="info-value">${info.symbol ?? 'N/A'}</div>
                            </div>
                            ${modelList}
                        </div>
                    `;
                    container.innerHTML = html;
                })
                .catch(error => {
                    container.innerHTML = `<div class="loading">Failed to load model info: ${error}</div>`;
                });
        }
    