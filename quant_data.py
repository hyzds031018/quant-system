import os
import requests
from datetime import datetime, timedelta

class SentimentAnalyzer:
    def __init__(self):
        # Extended lexicon
        self.positive_words = {
            'surge', 'jump', 'gain', 'rise', 'rising', 'bull', 'optimistic', 'positive', 
            'grow', 'growth', 'profit', 'beat', 'up', 'high', 'strong', 'rally', 'record',
            'outperform', 'upgrade', 'buy', 'revenue', 'earnings', 'climb', 'soar'
        }
        self.negative_words = {
            'drop', 'fall', 'falling', 'lose', 'loss', 'bear', 'pessimistic', 'negative', 
            'slump', 'miss', 'weak', 'down', 'low', 'crash', 'decline', 'plunge', 
            'underperform', 'downgrade', 'sell', 'risk', 'fail', 'concern', 'worry'
        }

    def analyze(self, text):
        if not text:
            return 0
        words = text.lower().replace('.', ' ').replace(',', ' ').split()
        score = 0
        for word in words:
            word = word.strip('!?:;"\'')
            if word in self.positive_words:
                score += 1
            elif word in self.negative_words:
                score -= 1
        
        # Normalize to -1 to 1
        # If score is > 5, it's very positive (1.0). If < -5, very negative (-1.0).
        if score > 0:
            return min(score * 0.2, 1.0)
        elif score < 0:
            return max(score * 0.2, -1.0)
        return 0

class FinnhubClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("FINNHUB_TOKEN")
        # Fallback if provided directly
        if not self.api_key and api_key: 
            self.api_key = api_key
            
        if not self.api_key:
            # Try to grab from global env if not passed
            self.api_key = os.environ.get("FINNHUB_TOKEN")
            
        self.base_url = "https://finnhub.io/api/v1"
        self.analyzer = SentimentAnalyzer()

    def get_company_news(self, symbol):
        if not self.api_key: return []
        
        # Default to last 5 days to increase chance of finding news
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        url = f"{self.base_url}/company-news?symbol={symbol}&from={start}&to={end}&token={self.api_key}"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                news = r.json()
                if isinstance(news, list):
                    # Sort by time desc
                    news.sort(key=lambda x: x['datetime'], reverse=True)
                    return self._process_news(news[:15]) # Top 15
            return []
        except Exception as e:
            print(f"Error fetching company news for {symbol}: {e}")
            return []

    def get_market_news(self):
        if not self.api_key: return []
        
        url = f"{self.base_url}/news?category=general&token={self.api_key}"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                news = r.json()
                if isinstance(news, list):
                    return self._process_news(news[:15])
            return []
        except Exception as e:
            print(f"Error fetching market news: {e}")
            return []

    def _process_news(self, news_list):
        processed = []
        for item in news_list:
            headline = item.get('headline', '')
            summary = item.get('summary', '')
            text = f"{headline} {summary}"
            sentiment = self.analyzer.analyze(text)
            
            # Format datetime
            dt = datetime.fromtimestamp(item.get('datetime', 0))
            dt_str = dt.strftime("%Y-%m-%d %H:%M")
            
            processed.append({
                'id': item.get('id'),
                'datetime': dt_str,
                'timestamp': item.get('datetime'),
                'headline': headline,
                'summary': summary,
                'source': item.get('source'),
                'url': item.get('url'),
                'image': item.get('image'),
                'sentiment': sentiment,
                'sentiment_label': 'Bullish' if sentiment > 0.1 else 'Bearish' if sentiment < -0.1 else 'Neutral',
                'sentiment_color': 'var(--green)' if sentiment > 0.1 else 'var(--red)' if sentiment < -0.1 else 'var(--text-muted)'
            })
        return processed
