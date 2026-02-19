# -*- coding: utf-8 -*-
import os
import json
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from database import db


class StockDataManager:
    """Stock data manager: fetches and caches data locally."""

    def __init__(self, data_dir="stock_data"):
        self.data_dir = data_dir
        self.ensure_data_dir()
        self.scheduler = BackgroundScheduler()
        self.last_refresh = {}
        self.refresh_interval_seconds = 300
        self.stock_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
            'UNH', 'JNJ', 'V', 'PG', 'HD', 'MA', 'BAC', 'ABBV', 'PFE', 'KO',
            'AVGO', 'PEP', 'TMO', 'COST', 'DIS', 'ABT', 'DHR', 'VZ', 'ADBE',
            'CRM', 'NFLX', 'XOM', 'NKE', 'LIN', 'CSCO', 'ACN', 'CVX', 'INTC'
        ]


    def ensure_data_dir(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def _get_tiingo_token(self):
        return os.environ.get("TIINGO_TOKEN")

    def _get_date_range_from_period(self, period):
        # supported: 1y, 2y, 5y, max
        end = datetime.utcnow().date()
        if period == "1y":
            start = end - timedelta(days=365)
        elif period == "2y":
            start = end - timedelta(days=365 * 2)
        elif period == "5y":
            start = end - timedelta(days=365 * 5)
        elif period == "max":
            start = end - timedelta(days=365 * 20)
        else:
            # fallback to 2y
            start = end - timedelta(days=365 * 2)
        return start.isoformat(), end.isoformat()

    def _fetch_tiingo(self, symbol, period="2y"):
        token = self._get_tiingo_token()
        if not token:
            raise RuntimeError("TIINGO_TOKEN not set")
        start_date, end_date = self._get_date_range_from_period(period)
        url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
        params = {
            "startDate": start_date,
            "endDate": end_date,
            "format": "json",
            "token": token
        }
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(f"Tiingo API error {resp.status_code}: {resp.text}")
        data = resp.json()
        if not data:
            return None
        return self._normalize_df(pd.DataFrame(data))

    def _fetch_tiingo_range(self, symbol, start_date, end_date):
        token = self._get_tiingo_token()
        if not token:
            raise RuntimeError("TIINGO_TOKEN not set")
        url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
        params = {
            "startDate": start_date,
            "endDate": end_date,
            "format": "json",
            "token": token
        }
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(f"Tiingo API error {resp.status_code}: {resp.text}")
        data = resp.json()
        if not data:
            return None
        return self._normalize_df(pd.DataFrame(data))


    def _normalize_df(self, df):
        df = df.copy()
        rename_map = {
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df.rename(columns=rename_map, inplace=True)
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
        df = df.drop_duplicates(subset=["Date"]).sort_values("Date")
        return df[required_cols]

    def get_stock_data(self, symbol, period="2y"):
        return self._fetch_tiingo(symbol, period=period)

    def get_stock_data_with_retry(self, symbol, period="2y", max_retries=5, base_delay=2):
        attempt = 0
        while True:
            try:
                return self.get_stock_data(symbol, period=period)
            except (requests.exceptions.RequestException, RuntimeError):
                attempt += 1
                if attempt > max_retries:
                    raise
                sleep_time = base_delay * (2 ** (attempt - 1))
                print(f"Fetch failed for {symbol}, retrying in {sleep_time}s...")
                time.sleep(sleep_time)

    def save_stock_data(self, symbol, data):
        if data is not None:
            # Save to Database
            db.save_daily_data(symbol, data)
            
            # Save CSV as backup (optional, keeping for now)
            file_path = os.path.join(self.data_dir, f"{symbol}.csv")
            data.to_csv(file_path, index=False)
            print(f"Data saved: {symbol}")

    def load_stock_data(self, symbol, refresh_if_stale=True, max_age_days=2, period="2y"):
        # Try loading from DB first
        data = db.load_data(symbol)
        
        if data is None or data.empty:
            # Fallback to CSV if DB empty (during migration) or Fetch API
            file_path = os.path.join(self.data_dir, f"{symbol}.csv")
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                # Migrate to DB immediately
                db.save_daily_data(symbol, data)
            else:
                data = self.get_stock_data(symbol, period=period)
                if data is not None:
                    self.save_stock_data(symbol, data)
                    return data
                return None
        
        if not refresh_if_stale:
            return data
            
        return self.refresh_stock_data(symbol, existing_df=data, max_age_days=max_age_days, period=period)

    def refresh_stock_data(self, symbol, existing_df=None, max_age_days=2, period="2y"):
        now_ts = time.time()
        last_ts = self.last_refresh.get(symbol)
        if last_ts and (now_ts - last_ts) < self.refresh_interval_seconds:
            if existing_df is not None:
                return existing_df
        if existing_df is None:
            file_path = os.path.join(self.data_dir, f"{symbol}.csv")
            if not os.path.exists(file_path):
                data = self.get_stock_data(symbol, period=period)
                if data is not None:
                    self.save_stock_data(symbol, data)
                    self.last_refresh[symbol] = now_ts
                return data
            existing_df = pd.read_csv(file_path)

        if existing_df is None or existing_df.empty or "Date" not in existing_df.columns:
            data = self.get_stock_data(symbol, period=period)
            if data is not None:
                self.save_stock_data(symbol, data)
                self.last_refresh[symbol] = now_ts
            return data

        try:
            last_date = pd.to_datetime(existing_df["Date"]).max().date()
        except Exception:
            last_date = None

        today = datetime.utcnow().date()
        if last_date and (today - last_date).days <= max_age_days:
            return existing_df

        start_date = (last_date + timedelta(days=1)).isoformat() if last_date else None
        end_date = today.isoformat()

        new_data = None
        if start_date:
            new_data = self._fetch_tiingo_range(symbol, start_date, end_date)
        else:
            new_data = self._fetch_tiingo(symbol, period=period)

        if new_data is None or new_data.empty:
            self.last_refresh[symbol] = now_ts
            return existing_df

        merged = pd.concat([existing_df, new_data], ignore_index=True)
        merged = self._normalize_df(merged)
        if merged is not None:
            self.save_stock_data(symbol, merged)
            self.last_refresh[symbol] = now_ts
            return merged
        return existing_df

    def update_all_stocks(self, period="2y", batch_size=5):
        print(f"Start updating stock data... {datetime.now()}")
        symbols = list(self.stock_symbols)
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            for symbol in batch:
                try:
                    if period in ("refresh", "latest"):
                        df = self.refresh_stock_data(symbol, max_age_days=1, period="2y")
                    else:
                        df = self.get_stock_data_with_retry(symbol, period=period)
                    if df is not None and not df.empty:
                        self.save_stock_data(symbol, df)
                except Exception as exc:
                    print(f"Failed to update {symbol}: {exc}")

            time.sleep(2)

        self.save_stock_list()
        print("All stock data updated!")

    def save_stock_list(self):
        list_file = os.path.join(self.data_dir, "stock_list.json")
        with open(list_file, 'w') as f:
            json.dump(self.stock_symbols, f)

    def get_stock_list(self):
        list_file = os.path.join(self.data_dir, "stock_list.json")
        if os.path.exists(list_file):
            with open(list_file, 'r') as f:
                return json.load(f)
        self.save_stock_list()
        return self.stock_symbols

    def get_kline_data(self, symbol):
        data = self.load_stock_data(symbol)
        if data is None or data.empty:
            return {"dates": [], "kline": []}
        kline_data = data[['Open', 'Close', 'Low', 'High']].values.tolist()
        dates = data['Date'].tolist()
        return {"dates": dates, "kline": kline_data}

    def get_market_statistics(self):
        stats = {}
        if not self.stock_symbols:
            return stats

        latest_prices = []
        volumes = []
        price_changes = []
        amounts = []

        for symbol in self.stock_symbols[:10]:
            data = self.load_stock_data(symbol)
            if data is not None and not data.empty:
                latest_data = data.iloc[-1]
                prev_data = data.iloc[-2] if len(data) > 1 else latest_data
                latest_prices.append((symbol, latest_data['Close']))
                volumes.append((symbol, latest_data['Volume']))
                pct_change = ((latest_data['Close'] - prev_data['Close']) / prev_data['Close']) * 100
                price_changes.append((symbol, pct_change))
                amount = latest_data['Close'] * latest_data['Volume']
                amounts.append((symbol, amount))

        if latest_prices:
            max_close_symbol, max_close = max(latest_prices, key=lambda x: x[1])
            min_close_symbol, min_close = min(latest_prices, key=lambda x: x[1])
            max_vol_symbol, max_vol = max(volumes, key=lambda x: x[1]) if volumes else ("", 0)
            max_pct_symbol, max_pct = max(price_changes, key=lambda x: x[1]) if price_changes else ("", 0)
            max_amt_symbol, max_amt = max(amounts, key=lambda x: x[1]) if amounts else ("", 0)
            stats = {
                'latest_date': datetime.now().strftime('%Y-%m-%d'),
                'highest_close_price': max_close,
                'lowest_open_price': min_close,
                'avg_close_price': float(np.mean([v for _, v in latest_prices])),
                'highest_volume': max_vol,
                'highest_pct_change': max_pct,
                'highest_amount': max_amt,
                'highest_close_stock': max_close_symbol,
                'lowest_open_stock': min_close_symbol,
                'highest_volume_stock': max_vol_symbol,
                'highest_pct_change_stock': max_pct_symbol,
                'highest_amount_stock': max_amt_symbol
            }
        return stats

    def start_scheduler(self):
        self.scheduler.add_job(
            func=self.update_all_stocks,
            trigger="interval",
            hours=1,
            id='update_stocks',
            kwargs={"period": "refresh", "batch_size": 5}
        )
        self.scheduler.start()
        print("The data update scheduler has been started and updates once every hour")

    def stop_scheduler(self):
        if self.scheduler.running:
            self.scheduler.shutdown()
            print("The data update scheduler has stopped")

    def migrate_csv_to_db(self):
        print("Checking for CSV to DB migration...")
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]
        for csv_file in csv_files:
            symbol = csv_file.replace(".csv", "")
            # Check if already in DB
            latest = db.get_latest_date(symbol)
            if not latest:
                print(f"Migrating {symbol} to DB...")
                try:
                    df = pd.read_csv(os.path.join(self.data_dir, csv_file))
                    db.save_daily_data(symbol, df)
                except Exception as e:
                    print(f"Failed to migrate {symbol}: {e}")


# Global instance
data_manager = StockDataManager()

def initialize_data():
    print("Initializing stock data...")
    stock_list = data_manager.get_stock_list()
    has_data = False
    for symbol in stock_list[:3]:
        if os.path.exists(os.path.join(data_manager.data_dir, f"{symbol}.csv")):
            has_data = True
            break
    if not has_data:
        print("No local data found, start downloading...")
        data_manager.update_all_stocks()
    else:
        print("Found local data, skipping initial download")
    data_manager.start_scheduler()


if __name__ == "__main__":
    data_manager.migrate_csv_to_db()
    data_manager.update_all_stocks()
