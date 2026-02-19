import random
import json
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from data_manager import data_manager

# 加载颜色配置文件（如果有）
try:
    with open('myColors.json', 'r', encoding='utf-8') as f:
        myColors = json.load(f)
except FileNotFoundError:
    myColors = {}

def get_statistic():
    """
    获取股票表的关键统计数据
    """
    return data_manager.get_market_statistics()

def get_stock_list():
    """
    获取数据库中所有股票的代码
    """
    return data_manager.get_stock_list()

def get_kline_data(stock_code):
    """
    获取指定股票的 K 线数据（日期、开盘、收盘、最低、最高）
    """
    kline_data_result = data_manager.get_kline_data(stock_code)
    # 添加日志记录
    # 为了避免日志过长，可以考虑只记录部分数据或数据的摘要
    if kline_data_result and 'kline' in kline_data_result and kline_data_result['kline']:
        summary = {
            "dates_count": len(kline_data_result.get('dates', [])),
            "kline_data_count": len(kline_data_result['kline']),
            "first_kline_point": kline_data_result['kline'][0] if kline_data_result['kline'] else None,
            "last_kline_point": kline_data_result['kline'][-1] if kline_data_result['kline'] else None
        }
        print(f"DEBUG: get_kline_data for {stock_code} - Summary: {json.dumps(summary)}")
    else:
        print(f"DEBUG: get_kline_data for {stock_code} - No data or empty kline data.")
    return kline_data_result

def get_words():
    """
    返回股票代码列表构造的词云数据
    """
    stock_list = get_stock_list()
    colors = ["#FF0000", "#00FF00", "#0000FF", "#FF6600", "#0099CC"]
    words3d = [{
        "label": w,
        "labelNum": random.randint(100, 1000),
        "color": random.choice(colors),
        "fontSize": random.randint(16, 42),
        "url": "",
        "target": ""
    } for w in stock_list[:20]]  # 只取前20个股票
    return {'words3d': words3d}

def strNum(num):
    if num < 1000:
        return str(num)
    elif num < 10000:
        return '{:.0f}K'.format(num / 1000)
    elif num < 1000000:
        return '{:.0f}W'.format(num / 10000)
    elif num < 1000000000:
        return '{:.0f}M'.format(num / 1000000)
    else:
        return '{:.0f}B'.format(num / 1000000000)

def get_data1():
    """
    获取最新交易日的涨跌幅（pct_change）分布
    """
    stock_list = get_stock_list()
    pct_changes = []
    
    for symbol in stock_list[:20]:  # 取前20个股票
        data = data_manager.load_stock_data(symbol)
        if data is not None and len(data) > 1:
            latest = data.iloc[-1]
            prev = data.iloc[-2]
            pct_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
            pct_changes.append(pct_change)
    
    # 分类统计
    ranges = {
        '< -5%': 0,
        '-5% ~ 0%': 0,
        '0% ~ 5%': 0,
        '> 5%': 0
    }
    
    for pct in pct_changes:
        if pct < -5:
            ranges['< -5%'] += 1
        elif -5 <= pct < 0:
            ranges['-5% ~ 0%'] += 1
        elif 0 <= pct < 5:
            ranges['0% ~ 5%'] += 1
        else:
            ranges['> 5%'] += 1
    
    return [{"pct_range": k, "count": v} for k, v in ranges.items()]

def get_data2():
    """
    获取最新交易日的成交量（vol）分布
    """
    stock_list = get_stock_list()
    volumes = []
    
    for symbol in stock_list[:20]:  # 取前20个股票
        data = data_manager.load_stock_data(symbol)
        if data is not None and not data.empty:
            latest_volume = data.iloc[-1]['Volume']
            volumes.append(latest_volume)
    
    # 分类统计
    ranges = {
        'Low': 0,
        'Medium': 0,
        'High': 0
    }
    
    if volumes:
        avg_volume = np.mean(volumes)
        for vol in volumes:
            if vol < avg_volume * 0.5:
                ranges['Low'] += 1
            elif vol < avg_volume * 1.5:
                ranges['Medium'] += 1
            else:
                ranges['High'] += 1
    
    return [{"vol_range": k, "count": v} for k, v in ranges.items()]

def get_data3():
    """
    获取最新交易日的股票代码、交易日期、开盘价、收盘价
    """
    stock_list = get_stock_list()
    result = []
    
    for symbol in stock_list[:20]:  # 取前20个股票
        data = data_manager.load_stock_data(symbol)
        if data is not None and not data.empty:
            latest = data.iloc[-1]
            result.append({
                'trade_date': latest['Date'],
                'ts_code': symbol,
                'open': latest['Open'],
                'close': latest['Close']
            })
    
    return result

def get_data4():
    """
    计算每只股票的 Beta 系数
    """
    stock_list = get_stock_list()
    beta_values = []
    
    # 计算市场收益率（所有股票的平均收益率）
    market_returns = []
    all_stock_returns = {}
    
    for symbol in stock_list[:10]:  # 取前10个股票计算Beta
        data = data_manager.load_stock_data(symbol)
        if data is not None and len(data) > 20:  # 需要足够的数据点
            returns = data['Close'].pct_change().dropna()
            if len(returns) > 0:
                all_stock_returns[symbol] = returns
                if len(market_returns) == 0:
                    market_returns = returns.copy()
                else:
                    # 对齐长度并计算平均
                    min_len = min(len(market_returns), len(returns))
                    market_returns = (market_returns.iloc[-min_len:] + returns.iloc[-min_len:]) / 2
    
    # 计算每只股票的Beta
    for symbol, returns in all_stock_returns.items():
        if len(returns) > 1 and len(market_returns) > 1:
            # 对齐数据长度
            min_len = min(len(returns), len(market_returns))
            stock_ret = returns.iloc[-min_len:]
            market_ret = market_returns.iloc[-min_len:]
            
            # 计算协方差和方差
            covariance = np.cov(stock_ret, market_ret)[0, 1]
            market_variance = np.var(market_ret)
            
            if market_variance > 0:
                beta = covariance / market_variance
                beta_values.append({"ts_code": symbol, "beta": round(beta, 2)})
    
    return beta_values

# PyTorch LSTM模型定义
class LSTMModel(nn.Module):
    """
    PyTorch LSTM模型类
    """
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, output_size)
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 只取最后一个时间步的输出
        out = out[:, -1, :]
        
        # Dropout
        out = self.dropout(out)
        
        # 全连接层
        out = self.fc1(out)
        out = self.fc2(out)
        
        return out

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型和归一化器路径
MODEL_PATH = "lstm_model.pth"
SCALER_PATH = "scaler.pkl"

def get_stock_data(stock_code):
    """ 从本地文件获取指定股票的收盘价 """
    refresh = True if data_manager._get_tiingo_token() else False
    data = data_manager.load_stock_data(stock_code, refresh_if_stale=refresh, max_age_days=2, period="2y")
    
    if data is None or data.empty:
        return pd.DataFrame()
    
    # 设置日期为索引
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # 确保返回所有必要的列，如果它们存在的话
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    # 筛选出实际存在于 data DataFrame 中的列
    cols_to_return = [col for col in required_cols if col in data.columns]
    if not cols_to_return: # 如果连这些基本列都没有，则返回空 DataFrame
        return pd.DataFrame()
    return data[cols_to_return]

def get_data5(stock_code="AAPL"):
    """
    使用 PyTorch LSTM 预测未来 7 天的收盘价
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        # 如果模型不存在，返回模拟数据
        from datetime import datetime, timedelta
        future_dates = []
        future_prices = []
        
        # 获取当前股票价格作为基准
        data = get_stock_data(stock_code)
        if not data.empty:
            base_price = data['Close'].iloc[-1]
        else:
            base_price = 100.0  # 默认价格
        
        # 生成7天预测数据
        for i in range(7):
            date = datetime.now() + timedelta(days=i+1)
            future_dates.append(date.strftime('%Y-%m-%d'))
            # 模拟价格变化 (±5% 随机波动)
            price_change = random.uniform(-0.05, 0.05)
            future_prices.append(base_price * (1 + price_change))
            base_price = future_prices[-1]  # 下一天以上一天为基准
        
        return [{"date": date, "predicted_close": price} 
                for date, price in zip(future_dates, future_prices)]

    # 加载数据
    df = get_stock_data(stock_code)
    if df.empty:
        return {"error": f"未找到 {stock_code} 的数据"}

    # 加载归一化器
    with open(SCALER_PATH, 'rb') as f:
        scaler = joblib.load(f)

    data_scaled = scaler.transform(df["Close"].values.reshape(-1, 1))

    # 取最近 60 天数据进行预测
    if len(data_scaled) < 60:
        return {"error": "数据不足，需要至少60天的历史数据"}
        
    last_60_days = data_scaled[-60:].reshape(1, 60, 1)

    # 加载PyTorch模型
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # 预测未来 7 天
    future_predictions = []
    
    # 将数据转换为PyTorch张量
    input_tensor = torch.FloatTensor(last_60_days).to(device)
    
    with torch.no_grad():
        for i in range(7):
            # 进行预测
            next_pred = model(input_tensor)
            next_pred_value = next_pred.squeeze().cpu().numpy()
            
            # 将预测值转换为标量
            if isinstance(next_pred_value, np.ndarray):
                next_pred_value = float(next_pred_value.item())
            else:
                next_pred_value = float(next_pred_value)
                
            future_predictions.append(next_pred_value)
            
            # 更新输入序列（滑动窗口）
            new_input = torch.cat([
                input_tensor[:, 1:, :], 
                torch.tensor([[[next_pred_value]]]).float().to(device)
            ], dim=1)
            input_tensor = new_input

    # 逆归一化
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
    future_predictions = [float(price) for price in future_predictions]

    # 生成预测日期
    last_date = df.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)

    # 返回 JSON 结构
    result = [{"date": str(date.date()), "predicted_close": price} 
              for date, price in zip(future_dates, future_predictions)]

    return result

def compute_dcf(fcf, wacc, g, years=5):
    """
    计算 DCF 估值
    """
    dcf_value = sum(fcf / (1 + wacc) ** (i + 1) for i in range(years))
    terminal_value = (fcf * (1 + g)) / (wacc - g)
    tv_discounted = terminal_value / (1 + wacc) ** years
    intrinsic_value = dcf_value + tv_discounted
    return round(intrinsic_value / 1e9, 2)  # 以十亿美元（B）为单位返回

def get_data6(wacc=0.1, g=0.02, reinvestment_rate=0.4):
    """
    计算 DCF 估值（基于交易数据）
    """
    stock_list = get_stock_list()
    results = []
    
    for symbol in stock_list[:10]:  # 取前10个股票
        data = data_manager.load_stock_data(symbol)
        if data is not None and not data.empty:
            latest = data.iloc[-1]
            close_price = latest['Close']
            open_price = latest['Open']
            volume = latest['Volume']
            
            eps = abs(close_price - open_price) / 100
            if eps < 0.01:
                eps = 0.01
            
            net_income = close_price * volume * 0.05  # 简化估算
            fcf = net_income * (1 - reinvestment_rate)
            intrinsic_value = compute_dcf(fcf, wacc, g)
            
            results.append({"ts_code": symbol, "dcf_value": intrinsic_value})
    
    return results

def get_market_statistics():
    """获取市场统计数据"""
    try:
        # 获取所有股票列表
        stock_list = get_stock_list()
        if not stock_list:
            return None

        highest_close = {'price': 0, 'stock': None}
        highest_change = {'pct': 0, 'stock': None}
        total_close = 0
        valid_stocks = 0
        latest_date = None

        # 遍历所有股票计算统计数据
        for stock in stock_list:
            stock_data = get_stock_data(stock)
            if stock_data is None or stock_data.empty:
                continue

            # 确保数据类型正确
            stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
            stock_data = stock_data.dropna(subset=['Close'])

            if len(stock_data) == 0:
                continue

            # 获取最新收盘价
            latest_close = stock_data['Close'].iloc[-1]
            total_close += latest_close
            valid_stocks += 1

            # 更新最高收盘价
            if latest_close > highest_close['price']:
                highest_close['price'] = latest_close
                highest_close['stock'] = stock

            # 计算涨跌幅
            if len(stock_data) > 1:
                prev_close = stock_data['Close'].iloc[-2]
                pct_change = ((latest_close - prev_close) / prev_close) * 100
                if pct_change > highest_change['pct']:
                    highest_change['pct'] = pct_change
                    highest_change['stock'] = stock

            # 更新最新交易日期
            current_date = stock_data.index[-1]
            if latest_date is None or current_date > latest_date:
                latest_date = current_date

        # 计算平均收盘价
        avg_close = total_close / valid_stocks if valid_stocks > 0 else 0

        return {
            'highest_close_stock': highest_close['stock'],
            'highest_close_price': highest_close['price'],
            'avg_close_price': avg_close,
            'highest_pct_change_stock': highest_change['stock'],
            'highest_pct_change': highest_change['pct'],
            'latest_date': latest_date.strftime('%Y-%m-%d') if latest_date else None
        }

    except Exception as e:
        print(f"Error calculating market statistics: {str(e)}")
        return None
