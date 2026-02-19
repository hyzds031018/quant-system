#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
import optuna
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('TkAgg')
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
rcParams['figure.figsize'] = 12, 8

# 时间序列分析相关
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 评估指标
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from datetime import datetime, timedelta

class EnhancedStockPredictor:
    """
    增强版股票预测器
    集成技术指标、多模型预测、风险评估等功能
    """
    
    def __init__(self, symbol="AAPL", period="2y"):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.models = {}
        self.predictions = {}
        self.results = {}
        
    def fetch_stock_data(self):
        """
        从本地 CSV 获取股票数据，并计算技术指标
        """
        print(f"Getting {self.symbol} Stock Data...")

        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_data")
        file_path = os.path.join(data_dir, f"{self.symbol}.csv")
        self.data = pd.read_csv(file_path)
        self.data["Date"] = pd.to_datetime(self.data["Date"])
        self.data.set_index("Date", inplace=True)

        if self.data.empty:
            raise ValueError(f"Unable to obtain {self.symbol} data")

        print(f"Successfully obtained data， {len(self.data)} record")
        print(f"data range: {self.data.index[0].date()} 到 {self.data.index[-1].date()}")

        # 计算技术指标
        self._calculate_technical_indicators()

        return self.data
    
    def _calculate_technical_indicators(self):
        """
        计算技术指标
        """
        print("Calculating technical indicators...")
        
        # 价格相关
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Log_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        
        # 移动平均线
        self.data['MA_5'] = self.data['Close'].rolling(window=5).mean()
        self.data['MA_10'] = self.data['Close'].rolling(window=10).mean()
        self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['MA_50'] = self.data['Close'].rolling(window=50).mean()
        
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = self.data['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = ema_12 - ema_26
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        self.data['MACD_Hist'] = self.data['MACD'] - self.data['MACD_Signal']
        
        # 布林带
        sma_20 = self.data['Close'].rolling(window=20).mean()
        std_20 = self.data['Close'].rolling(window=20).std()
        self.data['BB_Upper'] = sma_20 + (std_20 * 2)
        self.data['BB_Middle'] = sma_20
        self.data['BB_Lower'] = sma_20 - (std_20 * 2)
        
        # 波动率
        self.data['Volatility'] = self.data['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # 成交量指标
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']
        
        print("Technical indicator calculation completed!")
    
    def plot_comprehensive_analysis(self):
        """
        绘制综合分析图表
        """
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        
        # 1. 价格和移动平均线
        axes[0, 0].plot(self.data.index, self.data['Close'], label='Close Price', linewidth=2)
        axes[0, 0].plot(self.data.index, self.data['MA_20'], label='MA20', alpha=0.7)
        axes[0, 0].plot(self.data.index, self.data['MA_50'], label='MA50', alpha=0.7)
        axes[0, 0].set_title(f'{self.symbol} Stock Price with Moving Averages')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 布林带
        axes[0, 1].plot(self.data.index, self.data['Close'], label='Close Price', linewidth=2)
        axes[0, 1].plot(self.data.index, self.data['BB_Upper'], label='Bollinger Upper', alpha=0.7)
        axes[0, 1].plot(self.data.index, self.data['BB_Lower'], label='Bollinger Lower', alpha=0.7)
        axes[0, 1].fill_between(self.data.index, self.data['BB_Upper'], self.data['BB_Lower'], alpha=0.1)
        axes[0, 1].set_title('Bollinger Bands')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. RSI
        axes[1, 0].plot(self.data.index, self.data['RSI'], color='purple', linewidth=2)
        axes[1, 0].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        axes[1, 0].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        axes[1, 0].set_title('RSI (Relative Strength Index)')
        axes[1, 0].set_ylabel('RSI')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. MACD
        axes[1, 1].plot(self.data.index, self.data['MACD'], label='MACD', linewidth=2)
        axes[1, 1].plot(self.data.index, self.data['MACD_Signal'], label='Signal', linewidth=2)
        axes[1, 1].bar(self.data.index, self.data['MACD_Hist'], label='Histogram', alpha=0.3)
        axes[1, 1].set_title('MACD')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. 成交量
        axes[2, 0].bar(self.data.index, self.data['Volume'], alpha=0.6, color='orange')
        axes[2, 0].plot(self.data.index, self.data['Volume_MA'], color='red', linewidth=2, label='Volume MA')
        axes[2, 0].set_title('Trading Volume')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. 波动率
        axes[2, 1].plot(self.data.index, self.data['Volatility'], color='red', linewidth=2)
        axes[2, 1].set_title('Volatility (Annualized)')
        axes[2, 1].set_ylabel('Volatility')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 7. 收益率分布
        axes[3, 0].hist(self.data['Returns'].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[3, 0].set_title('Returns Distribution')
        axes[3, 0].set_xlabel('Daily Returns')
        axes[3, 0].grid(True, alpha=0.3)
        
        # 8. 价格散点图
        axes[3, 1].scatter(range(len(self.data)), self.data['Close'], alpha=0.5, c=self.data['Volume'], cmap='viridis')
        axes[3, 1].set_title('Price Scatter (colored by Volume)')
        axes[3, 1].set_xlabel('Time Index')
        axes[3, 1].set_ylabel('Close Price')
        
        plt.tight_layout()
        plt.show()
    
    def test_stationarity_enhanced(self, series, title="Time Series"):
        """
        增强版平稳性检验
        """
        print(f"\n=== {title} Stationarity test ===")
        
        # 滚动统计
        rolling_mean = series.rolling(window=12).mean()
        rolling_std = series.rolling(window=12).std()
        
        # 绘图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 原始序列与滚动统计
        axes[0, 0].plot(series, color='blue', label='Original', linewidth=2)
        axes[0, 0].plot(rolling_mean, color='red', label='Rolling Mean', linewidth=2)
        axes[0, 0].plot(rolling_std, color='black', label='Rolling Std', linewidth=2)
        axes[0, 0].legend(loc='best')
        axes[0, 0].set_title(f'{title} - Rolling Statistics')
        axes[0, 0].grid(True, alpha=0.3)
        
        # ACF 图
        plot_acf(series.dropna(), ax=axes[0, 1], lags=40, alpha=0.05)
        axes[0, 1].set_title('Autocorrelation Function')
        
        # PACF 图
        plot_pacf(series.dropna(), ax=axes[1, 0], lags=40, alpha=0.05)
        axes[1, 0].set_title('Partial Autocorrelation Function')
        
        # Q-Q 图
        from scipy import stats
        stats.probplot(series.dropna(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')
        
        plt.tight_layout()
        plt.show()
        
        # ADF 检验
        print("Augmented Dickey-Fuller Test Results:")
        adf_test = adfuller(series.dropna(), autolag='AIC')
        adf_output = pd.Series(adf_test[0:4], 
                              index=['Test Statistic', 'p-value', 'Lags Used', 'Observations Used'])
        
        for key, value in adf_test[4].items():
            adf_output[f'Critical Value ({key})'] = value
            
        print(adf_output)
        
        # 判断结果
        if adf_test[1] <= 0.05:
            print("The series is stationary (reject the null hypothesis)")
        else:
            print("The series is not stationary (accept the null hypothesis)")
            
        return adf_test[1] <= 0.05
    
    def optimize_arima_parameters(self, series, max_p=5, max_d=2, max_q=5):
        """
        使用 Optuna 优化 ARIMA 参数
        """
        print("Optimizing ARIMA parameters using Optuna...")
        
        def objective(trial):
            p = trial.suggest_int('p', 0, max_p)
            d = trial.suggest_int('d', 0, max_d)
            q = trial.suggest_int('q', 0, max_q)
            
            split_point = int(len(series) * 0.8)
            train_data = series[:split_point]
            test_data = series[split_point:]
            
            model = ARIMA(train_data, order=(p, d, q))
            fitted = model.fit()
            forecast = fitted.forecast(steps=len(test_data))
            mse = mean_squared_error(test_data, forecast)
            return mse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        best_params = study.best_params
        print(f"Optimal parameters: p={best_params['p']}, d={best_params['d']}, q={best_params['q']}")
        print(f"Optimal MSE: {study.best_value:.6f}")
        
        return (best_params['p'], best_params['d'], best_params['q'])
    
    def build_arima_model(self, target_col='Close', optimize=True):
        """
        构建增强版 ARIMA 模型
        """
        print(f"\n=== Build an ARIMA model (target: {target_col}) ===")
        
        # 数据预处理
        series = self.data[target_col].dropna()
        
        # 对数变换
        log_series = np.log(series)
        
        # 平稳性检验
        is_stationary = self.test_stationarity_enhanced(log_series, f"Log {target_col}")
        
        # 如果不平稳，进行差分
        if not is_stationary:
            diff_series = log_series.diff().dropna()
            self.test_stationarity_enhanced(diff_series, f"Differenced Log {target_col}")
            model_series = diff_series
        else:
            model_series = log_series
        
        # 数据分割
        split_point = int(len(model_series) * 0.8)
        train_data = model_series[:split_point]
        test_data = model_series[split_point:]
        
        print(f"Training set size: {len(train_data)}, Test set size: {len(test_data)}")
        
        # 参数优化
        if optimize:
            best_order = self.optimize_arima_parameters(train_data)
        else:
            best_order = (2, 1, 2)  # 默认参数
        
        # 模型训练
        print(f"Using Parameters {best_order}Training the ARIMA model...")
        model = ARIMA(train_data, order=best_order)
        fitted_model = model.fit()
        
        print(fitted_model.summary())
        
        # 预测
        forecast_result = fitted_model.get_forecast(steps=len(test_data), alpha=0.05)
        forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        
        # 保存结果
        self.models['arima'] = fitted_model
        self.predictions['arima'] = {
            'train_data': train_data,
            'test_data': test_data,
            'forecast': forecast,
            'conf_int': conf_int,
            'original_series': series,
            'log_series': log_series
        }
        
        # 计算评估指标
        mse = mean_squared_error(test_data, forecast)
        mae = mean_absolute_error(test_data, forecast)
        rmse = np.sqrt(mse)
        
        self.results['arima'] = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'order': best_order
        }
        
        print(f"ARIMA Model Evaluation:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        
        return fitted_model
    
    def plot_arima_results(self):
        """
        绘制 ARIMA 模型结果
        """
        if 'arima' not in self.predictions:
            print("Please run the ARIMA model first")
            return
            
        pred_data = self.predictions['arima']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 模型预测结果
        axes[0, 0].plot(pred_data['train_data'].index, pred_data['train_data'], 
                       label='Training Data', color='blue', linewidth=2)
        axes[0, 0].plot(pred_data['test_data'].index, pred_data['test_data'], 
                       label='Actual', color='green', linewidth=2)
        axes[0, 0].plot(pred_data['test_data'].index, pred_data['forecast'], 
                       label='Forecast', color='red', linewidth=2)
        
        # 置信区间
        axes[0, 0].fill_between(pred_data['test_data'].index,
                               pred_data['conf_int'].iloc[:, 0],
                               pred_data['conf_int'].iloc[:, 1],
                               color='pink', alpha=0.3, label='Confidence Interval')
        
        axes[0, 0].set_title('ARIMA Model Forecast')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 残差分析
        residuals = pred_data['test_data'] - pred_data['forecast']
        axes[0, 1].plot(residuals, color='red', linewidth=2)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 残差分布
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 预测 vs 实际
        axes[1, 1].scatter(pred_data['test_data'], pred_data['forecast'], alpha=0.7)
        axes[1, 1].plot([pred_data['test_data'].min(), pred_data['test_data'].max()],
                       [pred_data['test_data'].min(), pred_data['test_data'].max()],
                       'r--', linewidth=2)
        axes[1, 1].set_xlabel('Actual')
        axes[1, 1].set_ylabel('Predicted')
        axes[1, 1].set_title('Actual vs Predicted')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def calculate_var_and_drawdown(self, returns, confidence_level=0.05):
        """
        计算 VaR 和最大回撤
        """
        # VaR 计算
        var = np.percentile(returns.dropna(), confidence_level * 100)
        
        # 最大回撤计算
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return var, max_drawdown, drawdown
    
    def backtest_simple_strategy(self):
        """
        简单交易策略回测
        """
        print("\n=== Trading strategy backtesting ===")
        
        # 策略：基于移动平均线交叉
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['Close']
        signals['short_ma'] = self.data['MA_10']
        signals['long_ma'] = self.data['MA_20']
        
        # 生成交易信号
        signals['signal'] = 0
        signals['signal'][10:] = np.where(
            signals['short_ma'][10:] > signals['long_ma'][10:], 1, 0
        )
        signals['positions'] = signals['signal'].diff()
        
        # 计算策略收益
        signals['returns'] = signals['price'].pct_change()
        signals['strategy_returns'] = signals['signal'].shift(1) * signals['returns']
        
        # 计算累积收益
        signals['cumulative_returns'] = (1 + signals['returns']).cumprod()
        signals['cumulative_strategy_returns'] = (1 + signals['strategy_returns']).cumprod()
        
        # 风险指标
        var, max_dd, drawdown = self.calculate_var_and_drawdown(signals['strategy_returns'])
        
        # 绘制回测结果
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. 价格和信号
        axes[0].plot(signals.index, signals['price'], label='Price', linewidth=2)
        axes[0].plot(signals.index, signals['short_ma'], label='Short MA', alpha=0.7)
        axes[0].plot(signals.index, signals['long_ma'], label='Long MA', alpha=0.7)
        
        # 买入信号
        buy_signals = signals[signals['positions'] == 1]
        axes[0].scatter(buy_signals.index, buy_signals['price'], 
                       marker='^', color='green', s=100, label='Buy Signal')
        
        # 卖出信号
        sell_signals = signals[signals['positions'] == -1]
        axes[0].scatter(sell_signals.index, sell_signals['price'], 
                       marker='v', color='red', s=100, label='Sell Signal')
        
        axes[0].set_title('Trading Strategy Signals')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 累积收益对比
        axes[1].plot(signals.index, signals['cumulative_returns'], 
                    label='Buy & Hold', linewidth=2)
        axes[1].plot(signals.index, signals['cumulative_strategy_returns'], 
                    label='Strategy', linewidth=2)
        axes[1].set_title('Cumulative Returns Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 回撤
        axes[2].fill_between(signals.index, drawdown, 0, alpha=0.3, color='red')
        axes[2].plot(signals.index, drawdown, color='red', linewidth=2)
        axes[2].set_title(f'Strategy Drawdown (Max: {max_dd:.2%})')
        axes[2].set_ylabel('Drawdown')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 策略统计
        total_return = signals['cumulative_strategy_returns'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(signals)) - 1
        annual_volatility = signals['strategy_returns'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        print(f"Strategy backtesting results:")
        print(f"Total yield: {total_return:.2%}")
        print(f"Annualized Rate of Return: {annual_return:.2%}")
        print(f"Annualized volatility: {annual_volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Maximum Drawdown: {max_dd:.2%}")
        print(f"VaR (95%): {var:.2%}")
        
        return signals
    

def main():
    """
    主函数 - 展示完整的创新功能
    """
    # 初始化预测器
    predictor = EnhancedStockPredictor(symbol="AAPL", period="2y")
    
    # 1. 获取数据
    predictor.fetch_stock_data()
    
    # 2. 综合技术分析
    predictor.plot_comprehensive_analysis()
    
    # 3. ARIMA 建模
    predictor.build_arima_model(optimize=True)
    predictor.plot_arima_results()
    
    # 4. 交易策略回测
    predictor.backtest_simple_strategy()


if __name__ == '__main__':
    main()
