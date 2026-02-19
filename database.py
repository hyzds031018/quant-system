import sqlite3
import pandas as pd
import os
from contextlib import contextmanager

DB_NAME = "stock_data.db"

class DatabaseManager:
    def __init__(self, db_path=None):
        if db_path is None:
            # Default to current directory or a data directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(base_dir, "stock_data", DB_NAME)
        else:
            self.db_path = db_path
            
        self._ensure_db_dir()
        self._create_tables()

    def _ensure_db_dir(self):
        directory = os.path.dirname(self.db_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _create_tables(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Stocks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stocks (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    sector TEXT,
                    last_updated TIMESTAMP
                )
            ''')
            
            # Daily Prices table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_prices (
                    symbol TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, date),
                    FOREIGN KEY (symbol) REFERENCES stocks (symbol)
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON daily_prices (date)')
            conn.commit()

    def upsert_stock(self, symbol, name=None, sector=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO stocks (symbol, name, sector, last_updated)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (symbol, name, sector))
            conn.commit()

    def save_daily_data(self, symbol, df):
        """
        Save DataFrame to database.
        df should have columns: Date, Open, High, Low, Close, Volume
        """
        if df is None or df.empty:
            return

        # Ensure minimal stock info exists
        self.upsert_stock(symbol)

        # Prepare data for insertion (batch)
        records = []
        for _, row in df.iterrows():
            # Ensure Date is string YYYY-MM-DD
            d = row['Date']
            if hasattr(d, 'strftime'):
                date_str = d.strftime('%Y-%m-%d')
            else:
                date_str = str(d).split(' ')[0]

            records.append((
                symbol,
                date_str,
                row['Open'],
                row['High'],
                row['Low'],
                row['Close'],
                row['Volume']
            ))

        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Use INTO ... ON CONFLICT REPLACE to handle duplicates (update existing)
            cursor.executemany('''
                INSERT OR REPLACE INTO daily_prices (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', records)
            conn.commit()

    def load_data(self, symbol, start_date=None, end_date=None):
        """
        Load data from database into DataFrame
        """
        query = "SELECT date as Date, open as Open, high as High, low as Low, close as Close, volume as Volume FROM daily_prices WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
            
        query += " ORDER BY date ASC"
        
        with self.get_connection() as conn:
            # Using pandas read_sql is easier but requires connection object (which context manager handles)
            # But context manager closes connection on exit.
            # pd.read_sql expects connection to be open.
            df = pd.read_sql_query(query, conn, params=params)
            
        if df.empty:
            return None
            
        # Ensure proper types
        df['Date'] = pd.to_datetime(df['Date'])
        # Return index as Date? No, keep Date column as string/datetime for compatibility with CSV format
        # existing code expects 'Date' column and 'Open'... as float.
        # But 'Date' in CSV load was string. And processed later.
        # Let's keep it consistent: 'Date' column exists.
        return df

    def get_latest_date(self, symbol):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(date) FROM daily_prices WHERE symbol = ?", (symbol,))
            result = cursor.fetchone()
            return result[0] if result else None
    
    def get_all_symbols(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT symbol FROM stocks")
            return [row[0] for row in cursor.fetchall()]

db = DatabaseManager()
