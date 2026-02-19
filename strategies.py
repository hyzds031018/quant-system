import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass
import warnings

# Use EWMA for volatility forecasting if arch is not available
class VolatilityModel:
    @staticmethod
    def predict_volatility(returns, span=60):
        # Exponentially Weighted Moving Average (RiskMetrics)
        return returns.ewm(span=span).std() * np.sqrt(252)

@dataclass
class BacktestResult:
    signals: pd.Series
    daily_returns: pd.Series
    cumulative_returns: pd.Series
    metrics: dict
    trades: list = None

def calculate_trades(signals, commission=0.0003):
    """
    Extract trade log from signals.
    Assumes 'signal' column is 1 (Long) or 0 (Cash).
    """
    trades = []
    in_position = False
    entry_date = None
    entry_price = 0.0
    
    # Iterate over signals
    # Use itertuples for speed
    for row in signals.itertuples():
        # Check signal change
        # signal is 1 or 0
        sig = row.signal
        price = row.price
        date = row.Index
        
        if sig == 1 and not in_position:
            # Buy
            in_position = True
            entry_date = date
            entry_price = price
        elif sig == 0 and in_position:
            # Sell
            in_position = False
            exit_date = date
            exit_price = price
            # Calculate PnL
            gross_return = (exit_price - entry_price) / entry_price
            cost = commission * 2 # Entry + Exit (approx)
            net_return = gross_return - cost
            pnl = (exit_price - entry_price) - (entry_price * cost) # Simplified PnL based on 1 unit? No, percentage.
            
            trades.append({
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'entry_price': float(entry_price),
                'exit_price': float(exit_price),
                'return': float(net_return)
            })
            
    # Handle open position at end
    if in_position:
         trades.append({
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'exit_date': 'Open',
            'entry_price': float(entry_price),
            'exit_price': float(signals.iloc[-1]['price']),
            'return': float((signals.iloc[-1]['price'] - entry_price)/entry_price - commission)
        })
        
    return trades

class MovingAverageStrategy:
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, prices):
        signals = pd.DataFrame(index=prices.index)
        signals['price'] = prices
        signals['short_mavg'] = prices.rolling(window=self.short_window, min_periods=1).mean()
        signals['long_mavg'] = prices.rolling(window=self.long_window, min_periods=1).mean()
        
        # 1.0 = Buy, 0.0 = Hold/Neutral, -1.0 = Sell (but here we just use 1/0 for long-only)
        # Simplified: Long when Short > Long
        signals['signal'] = 0.0
        signals.loc[signals.index[self.short_window:], 'signal'] = np.where(
            signals['short_mavg'][self.short_window:] > signals['long_mavg'][self.short_window:],
            1.0, 0.0
        )
        signals['positions'] = signals['signal'].diff()
        return signals

    def backtest(self, prices, initial_capital=100000.0, commission=0.0003):
        signals = self.generate_signals(prices)
        positions = pd.DataFrame(index=signals.index).fillna(0.0)
        
        market_returns = prices.pct_change().fillna(0)
        strategy_returns = market_returns * signals['signal'].shift(1).fillna(0)
        
        # Apply costs
        trades = signals['positions'].abs().fillna(0)
        costs = trades * commission
        strategy_returns = strategy_returns - costs

        cumulative = (1 + strategy_returns).cumprod()
        total_return = cumulative.iloc[-1] - 1 if not cumulative.empty else 0
        
        # Metrics
        sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() != 0 else 0
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()
        
        trades_log = calculate_trades(signals, commission)

        return BacktestResult(
            signals=signals,
            daily_returns=strategy_returns,
            cumulative_returns=cumulative,
            metrics={
                'Total Return': f"{total_return:.2%}",
                'Sharpe Ratio': f"{sharpe:.2f}",
                'Max Drawdown': f"{max_drawdown:.2%}",
                'Trades': int(len(trades_log))
            },
            trades=trades_log
        )

class PortfolioOptimizer:
    def __init__(self, returns):
        self.returns = returns
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.num_assets = len(self.mean_returns)

    def maximize_sharpe_ratio(self, risk_free_rate=0.0):
        def negative_sharpe(weights):
            p_ret = np.sum(self.mean_returns * weights) * 252
            p_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(252)
            return -(p_ret - risk_free_rate) / p_vol if p_vol > 0 else 0

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_weights = self.num_assets * [1. / self.num_assets,]
        
        result = minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        opt_weights = result.x
        if result.success:
            ret = np.sum(self.mean_returns * opt_weights) * 252
            vol = np.sqrt(np.dot(opt_weights.T, np.dot(self.cov_matrix, opt_weights))) * np.sqrt(252)
            sharpe = (ret - risk_free_rate) / vol
            return {
                'weights': dict(zip(self.returns.columns, np.round(opt_weights, 4))),
                'return': ret,
                'volatility': vol,
                'sharpe': sharpe
            }
        return None

class VolatilityTargetStrategy:
    """
    Adjust position size to maintain constant volatility data.
    """
    def __init__(self, target_vol=0.15):
        self.target_vol = target_vol

    def backtest(self, prices):
        returns = prices.pct_change()
        realized_vol = VolatilityModel.predict_volatility(returns, span=21) # 1-month rolling vol
        
        # Position size = Target Vol / Realized Vol
        # Cap leverage at 1.5x for safety
        leverage = (self.target_vol / realized_vol).clip(0, 1.5).shift(1) # Rebalance next day
        
        strategy_returns = returns * leverage
        cumulative = (1 + strategy_returns).cumprod()
        
        return {
            'leverage': leverage,
            'cumulative_returns': cumulative,
            'metrics': {
                'Target Vol': f"{self.target_vol:.0%}",
                'Realized Vol': f"{strategy_returns.std() * np.sqrt(252):.2%}",
                'Total Return': f"{cumulative.iloc[-1] - 1:.2%}"
            }
        }

class MACDStrategy:
    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, prices):
        signals = pd.DataFrame(index=prices.index)
        signals['price'] = prices
        
        # Calculate MACD
        exp1 = prices.ewm(span=self.fast, adjust=False).mean()
        exp2 = prices.ewm(span=self.slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=self.signal, adjust=False).mean()
        
        signals['macd'] = macd
        signals['signal_line'] = signal_line
        signals['histogram'] = macd - signal_line
        
        # Generate signals (Crossover)
        signals['signal'] = 0.0
        signals.loc[signals.index[self.slow:], 'signal'] = np.where(
            signals['macd'][self.slow:] > signals['signal_line'][self.slow:], 1.0, 0.0
        )
        signals['positions'] = signals['signal'].diff()
        
        return signals

    def backtest(self, prices, initial_capital=100000.0, commission=0.0003):
        signals = self.generate_signals(prices)
        
        # Calculate returns
        market_returns = prices.pct_change().fillna(0)
        strategy_returns = market_returns * signals['signal'].shift(1).fillna(0)
        
        # Apply transaction costs (simplified)
        trades = signals['positions'].abs().fillna(0)
        # Cost is roughly proportional to position change value. 
        # Here we approximate cost as percentage deduction from return.
        # This is a simplification but works for vectorized backtest.
        costs = trades * commission
        strategy_returns = strategy_returns - costs
        
        cumulative = (1 + strategy_returns).cumprod()
        total_return = cumulative.iloc[-1] - 1 if not cumulative.empty else 0
        
        sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() != 0 else 0
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()
        
        trades_log = calculate_trades(signals, commission)
        
        return BacktestResult(
            signals=signals,
            daily_returns=strategy_returns,
            cumulative_returns=cumulative,
            metrics={
                'Total Return': f"{total_return:.2%}",
                'Sharpe Ratio': f"{sharpe:.2f}",
                'Max Drawdown': f"{max_drawdown:.2%}",
                'Trades': int(len(trades_log))
            },
            trades=trades_log
        )

class RSIStrategy:
    def __init__(self, window=14, overbought=70, oversold=30):
        self.window = window
        self.overbought = overbought
        self.oversold = oversold

    def generate_signals(self, prices):
        signals = pd.DataFrame(index=prices.index)
        signals['price'] = prices
        
        # Calculate RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        signals['rsi'] = rsi.fillna(50)
        
        # Generate signals
        # Buy when RSI < Oversold, Sell when RSI > Overbought
        # But for trend following? No, RSI is mean reversion usually.
        # Classic: Cross above 30 -> Buy. Cross below 70 -> Sell.
        
        signals['signal'] = 0.0
        # We need stateful logic for RSI usually (hold until sell signal).
        # Vectorized is tricky. Let's do simple thresholds:
        # If mean reversion: Signal = 1 if RSI < 30, 0 if RSI > 70? 
        # Better: Long if RSI < 30, exit if RSI > 70.
        
        curr_sig = 0
        sig_list = []
        for r in signals['rsi']:
            if r < self.oversold:
                curr_sig = 1
            elif r > self.overbought:
                curr_sig = 0
            sig_list.append(curr_sig)
            
        signals['signal'] = sig_list
        signals['positions'] = signals['signal'].diff()
        return signals

    def backtest(self, prices, initial_capital=100000.0, commission=0.0003):
        signals = self.generate_signals(prices)
        
        market_returns = prices.pct_change().fillna(0)
        strategy_returns = market_returns * signals['signal'].shift(1).fillna(0)
        
        trades = signals['positions'].abs().fillna(0)
        costs = trades * commission
        strategy_returns = strategy_returns - costs
        
        cumulative = (1 + strategy_returns).cumprod()
        total_return = cumulative.iloc[-1] - 1 if not cumulative.empty else 0
        
        sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() != 0 else 0
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()
        
        trades_log = calculate_trades(signals, commission)
        
        return BacktestResult(
            signals=signals,
            daily_returns=strategy_returns,
            cumulative_returns=cumulative,
            metrics={
                'Total Return': f"{total_return:.2%}",
                'Sharpe Ratio': f"{sharpe:.2f}",
                'Max Drawdown': f"{max_drawdown:.2%}",
                'Trades': int(len(trades_log))
            },
            trades=trades_log
        )


def find_volatility_opportunities(
    stock_list=None,
    top_n=5,
    bottom_n=5,
    lookbacks=(5, 10, 20),
    momentum_window=5,
    momentum_min=0.0,
    momentum_abs=True,
    vol_series_window=20,
    vol_series_length=20,
):
    from data_manager import data_manager

    if stock_list is None:
        stock_list = data_manager.get_stock_list()

    max_lb = max(lookbacks)
    results = []

    for symbol in stock_list:
        df = data_manager.load_stock_data(symbol)
        if df is None or df.empty or "Close" not in df.columns:
            continue

        if "Date" in df.columns:
            df = df.sort_values("Date")

        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(close) < (max_lb + 1) or len(close) < (momentum_window + 1):
            continue

        returns = close.pct_change().dropna()
        if len(returns) < max_lb:
            continue

        vols = {}
        for lb in lookbacks:
            if len(returns) >= lb:
                vols[lb] = float(np.std(returns.iloc[-lb:]) * np.sqrt(252))
            else:
                vols[lb] = None

        if any(v is None for v in vols.values()):
            continue

        if len(lookbacks) == 3:
            weights = [0.5, 0.3, 0.2]
        else:
            weights = [1.0 / len(lookbacks)] * len(lookbacks)

        weighted_vol = float(
            sum(weights[i] * vols[lb] for i, lb in enumerate(lookbacks))
        )

        momentum = float(close.iloc[-1] / close.iloc[-1 - momentum_window] - 1.0)
        if momentum_min and momentum_min > 0:
            if momentum_abs:
                if abs(momentum) < momentum_min:
                    continue
            else:
                if momentum < momentum_min:
                    continue

        vol_series_daily = returns.rolling(vol_series_window).std().dropna()
        vol_series_ann = vol_series_daily * np.sqrt(252)
        if vol_series_length and vol_series_length > 0:
            vol_series_daily = vol_series_daily.iloc[-vol_series_length:]
            vol_series_ann = vol_series_ann.iloc[-vol_series_length:]
        vol_series_daily_list = (
            [float(v) for v in vol_series_daily.tolist()] if len(vol_series_daily) else []
        )
        vol_series_ann_list = (
            [float(v) for v in vol_series_ann.tolist()] if len(vol_series_ann) else []
        )

        results.append(
            {
                "symbol": symbol,
                "vol_weighted": weighted_vol,
                "vol_1w": vols.get(lookbacks[0]),
                "vol_2w": vols.get(lookbacks[1]) if len(lookbacks) > 1 else None,
                "vol_1m": vols.get(lookbacks[2]) if len(lookbacks) > 2 else None,
                "momentum": momentum,
                "last_close": float(close.iloc[-1]),
                "vol_series": vol_series_ann_list,
                "vol_series_daily": vol_series_daily_list,
                "vol_series_window": vol_series_window,
                "vol_series_length": len(vol_series_ann_list),
            }
        )

    sorted_desc = sorted(results, key=lambda x: x["vol_weighted"], reverse=True)
    sorted_asc = list(reversed(sorted_desc))
    rank_desc = {item["symbol"]: idx + 1 for idx, item in enumerate(sorted_desc)}
    rank_asc = {item["symbol"]: idx + 1 for idx, item in enumerate(sorted_asc)}
    for item in results:
        item["rank_high"] = rank_desc.get(item["symbol"])
        item["rank_low"] = rank_asc.get(item["symbol"])

    high = sorted_desc[:top_n]
    low = sorted_asc[:bottom_n]

    return {
        "long": high,
        "short": low,
        "universe": len(stock_list),
        "scanned": len(results),
        "lookbacks": list(lookbacks),
        "momentum_window": momentum_window,
        "momentum_min": momentum_min,
        "momentum_abs": momentum_abs,
        "vol_series_window": vol_series_window,
        "vol_series_length": vol_series_length,
    }
