from __future__ import annotations

import math
import random
import os
import warnings
import pickle
import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

def _select_device(prefer: str | None = None) -> torch.device:
    if prefer is None or prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefer = prefer.lower()
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = _select_device()
print(f"Use device: {device}")

def _set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StockDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray, feature_names=None):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.feature_names = feature_names

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.targets[idx]


class AttentionLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2, attention_heads: int = 8):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size * 2),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        x1 = self.layer_norm1(lstm_out + attn_out)
        ff_out = self.feed_forward(x1)
        x2 = self.layer_norm2(x1 + ff_out)
        output = self.fc_out(x2[:, -1, :])
        return output, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    def __init__(self, input_size: int, d_model: int = 256, nhead: int = 8, num_layers: int = 4, output_size: int = 1, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size),
        )

    def forward(self, x: torch.Tensor):
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        pooled = torch.mean(encoded, dim=1)
        return self.output_projection(pooled)


class EnhancedGRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.attention_weights = nn.Linear(hidden_size * 2, 1)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor):
        gru_out, _ = self.gru(x)
        attention_scores = self.attention_weights(gru_out)
        attention_weights = F.softmax(attention_scores, dim=1)
        context = torch.sum(gru_out * attention_weights, dim=1)
        return self.fc_out(context)


@dataclass
class SequenceBundle:
    X_raw: np.ndarray
    y_raw: np.ndarray
    prev_closes: np.ndarray
    dates: pd.DatetimeIndex


class EnhancedStockPredictor:
    def __init__(self, symbol: str = "AAPL", period: str = "2y", sequence_length: int = 60):
        self.symbol = symbol
        self.period = period
        self.sequence_length = sequence_length
        self.data: pd.DataFrame | None = None
        self.models: Dict[str, Dict] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.predictions: Dict = {}
        self.feature_columns: List[str] = []
        self.sequence_metadata: Dict = {}

    def fetch_and_prepare_data(self):
        print(f"Reading {self.symbol} stock data...")
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_data")
        file_path = os.path.join(data_dir, f"{self.symbol}.csv")
        self.data = pd.read_csv(file_path)
        self.data["Date"] = pd.to_datetime(self.data["Date"])
        self.data.set_index("Date", inplace=True)
        if self.data.empty:
            raise ValueError(f"Unable to read {self.symbol} data")
        self._calculate_advanced_features()
        return self.data

    def _calculate_advanced_features(self):
        df = self.data
        df["Returns"] = df["Close"].pct_change()
        df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["High_Low_Pct"] = (df["High"] - df["Low"]) / df["Close"]
        df["Open_Close_Pct"] = (df["Close"] - df["Open"]) / df["Open"]

        for p in [5, 10, 20, 50, 200]:
            df[f"MA_{p}"] = df["Close"].rolling(window=p).mean()
            df[f"MA_{p}_ratio"] = df["Close"] / df[f"MA_{p}"]

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        df["BB_Upper"] = sma20 + (2 * std20)
        df["BB_Lower"] = sma20 - (2 * std20)
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / sma20
        df["BB_Position"] = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

        low14 = low.rolling(window=14).min()
        high14 = high.rolling(window=14).max()
        df["Stoch_K"] = ((close - low14) / (high14 - low14)) * 100
        df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()
        df["Williams_R"] = ((high14 - close) / (high14 - low14)) * -100

        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["ATR"] = tr.ewm(span=14, adjust=False).mean()

        df["Volume_MA"] = df["Volume"].rolling(window=20).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"]
        df["Momentum"] = close.diff(10)
        df["ROC"] = (close.diff(10) / close.shift(10)) * 100
        df["Upper_Shadow"] = df["High"] - np.maximum(df["Open"], df["Close"])
        df["Lower_Shadow"] = np.minimum(df["Open"], df["Close"]) - df["Low"]
        df["Body_Size"] = (df["Close"] - df["Open"]).abs()

        for p in [5, 10, 20]:
            df[f"Volatility_{p}"] = df["Returns"].rolling(window=p).std() * np.sqrt(252)

        df["Price_Position"] = (close - close.rolling(window=50).min()) / (
            close.rolling(window=50).max() - close.rolling(window=50).min()
        )

        self.data = df.dropna().copy()
        self.feature_columns = [
            "Open", "High", "Low", "Volume",
            "Returns", "High_Low_Pct", "Open_Close_Pct",
            "MA_5_ratio", "MA_10_ratio", "MA_20_ratio", "MA_50_ratio",
            "RSI", "MACD", "MACD_Signal", "MACD_Hist",
            "BB_Width", "BB_Position", "Stoch_K", "Stoch_D", "Williams_R",
            "ATR", "Volume_Ratio", "Momentum", "ROC",
            "Upper_Shadow", "Lower_Shadow", "Body_Size",
            "Volatility_5", "Volatility_10", "Volatility_20", "Price_Position",
        ]

    def _build_raw_sequences(self, target_col: str = "Log_Returns") -> SequenceBundle:
        features = self.data[self.feature_columns].values
        targets = self.data[target_col].values
        closes = self.data["Close"].values
        dates = self.data.index

        X_raw, y_raw, prev_closes, target_dates = [], [], [], []
        for i in range(self.sequence_length, len(features)):
            X_raw.append(features[i - self.sequence_length : i])
            y_raw.append(targets[i])
            prev_closes.append(closes[i - 1])
            target_dates.append(dates[i])

        return SequenceBundle(
            X_raw=np.array(X_raw),
            y_raw=np.array(y_raw),
            prev_closes=np.array(prev_closes),
            dates=pd.DatetimeIndex(target_dates),
        )

    @staticmethod
    def _returns_to_prices(log_returns: np.ndarray, prev_closes: np.ndarray) -> np.ndarray:
        return prev_closes * np.exp(log_returns)

    def create_sequences(self, target_col: str = "Log_Returns"):
        bundle = self._build_raw_sequences(target_col=target_col)
        train_size = int(len(bundle.X_raw) * 0.8)

        X_train_raw = bundle.X_raw[:train_size]
        X_test_raw = bundle.X_raw[train_size:]
        y_train_raw = bundle.y_raw[:train_size]
        y_test_raw = bundle.y_raw[train_size:]

        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

        n_features = X_train_raw.shape[-1]
        feature_scaler.fit(X_train_raw.reshape(-1, n_features))
        X_train = feature_scaler.transform(X_train_raw.reshape(-1, n_features)).reshape(X_train_raw.shape)
        X_test = feature_scaler.transform(X_test_raw.reshape(-1, n_features)).reshape(X_test_raw.shape)

        y_train = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
        y_test = target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()

        self.scalers["feature"] = feature_scaler
        self.scalers["target"] = target_scaler
        self.sequence_metadata = {
            "train_size": train_size,
            "total_sequences": len(bundle.X_raw),
            "prev_closes": bundle.prev_closes,
            "target_dates": bundle.dates,
            "target_col": target_col,
        }

        return X_train, X_test, y_train, y_test

    def _model_configs(self):
        return {
            "attention_lstm": {"model": AttentionLSTM(len(self.feature_columns), 128, 2, 1, 0.2), "name": "Attention LSTM"},
            "gru": {"model": EnhancedGRUModel(len(self.feature_columns), 128, 2, 1, 0.2), "name": "Enhanced GRU"},
            "transformer": {"model": TransformerPredictor(len(self.feature_columns), 256, 8, 4, 1, 0.1), "name": "Transformer"},
        }

    def save_artifacts(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        for model_name, info in self.models.items():
            torch.save(info["model"].state_dict(), os.path.join(model_dir, f"{model_name}.pth"))
        meta = {
            "feature_columns": self.feature_columns,
            "sequence_length": self.sequence_length,
            "target_col": self.sequence_metadata.get("target_col", "Log_Returns"),
            "feature_scaler": self.scalers.get("feature"),
            "target_scaler": self.scalers.get("target"),
        }
        with open(os.path.join(model_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        self._save_model_info(model_dir)
        self._save_metrics(model_dir)

    def load_artifacts(self, model_dir: str):
        meta_path = os.path.join(model_dir, "meta.pkl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Model artifacts not found: {model_dir}")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.feature_columns = meta["feature_columns"]
        self.sequence_length = meta["sequence_length"]
        self.scalers["feature"] = meta["feature_scaler"]
        self.scalers["target"] = meta["target_scaler"]

        self.models = {}
        for model_name, cfg in self._model_configs().items():
            model_path = os.path.join(model_dir, f"{model_name}.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Missing model weight: {model_path}")
            model = cfg["model"].to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            self.models[model_name] = {
                "model": model,
                "name": cfg["name"],
                "train_losses": [],
                "val_losses": [],
            }

        return meta

    def _save_model_info(self, model_dir: str):
        info = {
            "symbol": self.symbol,
            "sequence_length": self.sequence_length,
            "feature_count": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "target_col": self.sequence_metadata.get("target_col", "Log_Returns"),
            "models": {name: self.models[name]["name"] for name in self.models.keys()},
        }
        path = os.path.join(model_dir, "model_info.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    def _save_metrics(self, model_dir: str):
        if not self.predictions:
            return
        metrics = {
            "price_rmse_simple": float(np.sqrt(mean_squared_error(
                self.predictions["actual_prices"], self.predictions["ensemble_simple_prices"]
            ))),
            "price_rmse_weighted": float(np.sqrt(mean_squared_error(
                self.predictions["actual_prices"], self.predictions["ensemble_weighted_prices"]
            ))),
            "price_mae_simple": float(mean_absolute_error(
                self.predictions["actual_prices"], self.predictions["ensemble_simple_prices"]
            )),
            "price_mae_weighted": float(mean_absolute_error(
                self.predictions["actual_prices"], self.predictions["ensemble_weighted_prices"]
            )),
        }
        path = os.path.join(model_dir, "metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    def prepare_inference(self, target_col: str = "Log_Returns"):
        bundle = self._build_raw_sequences(target_col=target_col)
        self.sequence_metadata = {
            "train_size": 0,
            "total_sequences": len(bundle.X_raw),
            "prev_closes": bundle.prev_closes,
            "target_dates": bundle.dates,
            "target_col": target_col,
        }

    def train_ensemble_models(self, epochs: int = 100, optimize_hyperparams: bool = False, log_every: int = 10):
        _set_seed(42)
        X_train_full, X_test, y_train_full, y_test = self.create_sequences(target_col="Log_Returns")

        # time-based validation split within training set
        val_size = max(int(len(X_train_full) * 0.1), 1)
        train_end = len(X_train_full) - val_size

        X_train = X_train_full[:train_end]
        y_train = y_train_full[:train_end]
        X_val = X_train_full[train_end:]
        y_val = y_train_full[train_end:]

        train_loader = DataLoader(StockDataset(X_train, y_train, self.feature_columns), batch_size=64, shuffle=True)
        val_loader = DataLoader(StockDataset(X_val, y_val, self.feature_columns), batch_size=64, shuffle=False)

        for model_name, cfg in self._model_configs().items():
            model = cfg["model"].to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            criterion = nn.MSELoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

            train_losses, val_losses = [], []
            best_val = float("inf")
            best_state = None
            patience = 12
            patience_left = patience

            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    if model_name == "attention_lstm":
                        output, _ = model(batch_X)
                    else:
                        output = model(batch_X)
                    loss = criterion(output.squeeze(), batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        if model_name == "attention_lstm":
                            output, _ = model(batch_X)
                        else:
                            output = model(batch_X)
                        val_loss += criterion(output.squeeze(), batch_y).item()

                train_loss = train_loss / max(len(train_loader), 1)
                val_loss = val_loss / max(len(val_loader), 1)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                scheduler.step(val_loss)

                if log_every > 0 and ((epoch + 1) % log_every == 0 or epoch == 0 or epoch + 1 == epochs):
                    print(
                        f"[{model_name}] Epoch {epoch + 1}/{epochs} - Train: {train_loss:.6f} Val: {val_loss:.6f}"
                    )

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience_left = patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            self.models[model_name] = {
                "model": model,
                "name": cfg["name"],
                "train_losses": train_losses,
                "val_losses": val_losses,
            }
            torch.save(model.state_dict(), f"{model_name}_model.pth")

        return X_test, y_test

    def evaluate_ensemble_models(self, X_test, y_test):
        test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=64, shuffle=False)
        individual = {}

        for model_name, info in self.models.items():
            preds = []
            model = info["model"]
            model.eval()
            with torch.no_grad():
                for batch_X, _ in test_loader:
                    batch_X = batch_X.to(device)
                    if model_name == "attention_lstm":
                        out, _ = model(batch_X)
                    else:
                        out = model(batch_X)
                    preds.extend(out.squeeze().cpu().numpy())
            individual[model_name] = np.array(preds)

        all_preds = np.array(list(individual.values()))
        ensemble = np.mean(all_preds, axis=0)

        # weight by price-space RMSE on the validation tail for stability
        val_size = max(int(len(self.sequence_metadata["prev_closes"]) * 0.1), 1)
        val_prev_closes = self.sequence_metadata["prev_closes"][-val_size:]
        val_actual_log = self.scalers["target"].inverse_transform(y_test[-val_size:].reshape(-1, 1)).flatten()
        val_actual_prices = self._returns_to_prices(val_actual_log, val_prev_closes)
        weight_list = []
        for m, v in individual.items():
            val_log = self.scalers["target"].inverse_transform(v[-val_size:].reshape(-1, 1)).flatten()
            val_prices = self._returns_to_prices(val_log, val_prev_closes)
            rmse = np.sqrt(mean_squared_error(val_actual_prices, val_prices))
            weight_list.append(1.0 / max(rmse, 1e-8))
        weights = np.array(weight_list)
        weights = weights / weights.sum()
        weighted = np.average(all_preds, axis=0, weights=weights)

        train_size = self.sequence_metadata["train_size"]
        prev_closes = self.sequence_metadata["prev_closes"][train_size:]

        actual_log = self.scalers["target"].inverse_transform(y_test.reshape(-1, 1)).flatten()
        actual_prices = self._returns_to_prices(actual_log, prev_closes)

        individual_prices = {}
        for k, v in individual.items():
            pred_log = self.scalers["target"].inverse_transform(v.reshape(-1, 1)).flatten()
            individual_prices[k] = self._returns_to_prices(pred_log, prev_closes)

        simple_log = self.scalers["target"].inverse_transform(ensemble.reshape(-1, 1)).flatten()
        weighted_log = self.scalers["target"].inverse_transform(weighted.reshape(-1, 1)).flatten()
        simple_prices = self._returns_to_prices(simple_log, prev_closes)
        weighted_prices = self._returns_to_prices(weighted_log, prev_closes)

        print(f"Price RMSE (simple): {np.sqrt(mean_squared_error(actual_prices, simple_prices)):.4f}")
        print(f"Price RMSE (weighted): {np.sqrt(mean_squared_error(actual_prices, weighted_prices)):.4f}")

        self.predictions = {
            "individual_returns_scaled": individual,
            "ensemble_simple_returns_scaled": ensemble,
            "ensemble_weighted_returns_scaled": weighted,
            "actual_returns_scaled": y_test,
            "individual_prices": individual_prices,
            "ensemble_simple_prices": simple_prices,
            "ensemble_weighted_prices": weighted_prices,
            "actual_prices": actual_prices,
            "weights": weights,
        }

        return individual, ensemble, weighted

    def plot_comprehensive_results(self):
        if not self.predictions:
            print("Run evaluate_ensemble_models first")
            return
        dates = self.sequence_metadata["target_dates"][self.sequence_metadata["train_size"] :]
        plt.figure(figsize=(14, 6))
        plt.plot(dates, self.predictions["actual_prices"], label="Actual")
        plt.plot(dates, self.predictions["ensemble_weighted_prices"], label="Weighted Ensemble")
        plt.title("Prediction vs Actual")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def predict_historical(self):
        if not self.models:
            raise ValueError("Model not trained")

        features = self.data[self.feature_columns].values
        features_scaled = self.scalers["feature"].transform(features)
        sequences = np.array([
            features_scaled[i - self.sequence_length : i] for i in range(self.sequence_length, len(features_scaled))
        ])
        tensor = torch.FloatTensor(sequences).to(device)

        predictions = {}
        prev_closes = self.sequence_metadata["prev_closes"]

        for model_name, info in self.models.items():
            model = info["model"]
            model.eval()
            with torch.no_grad():
                if model_name == "attention_lstm":
                    out, _ = model(tensor)
                else:
                    out = model(tensor)
            scaled = out.squeeze().cpu().numpy()
            log_ret = self.scalers["target"].inverse_transform(scaled.reshape(-1, 1)).flatten()
            predictions[model_name] = self._returns_to_prices(log_ret, prev_closes)

        weights = self.predictions.get("weights")
        if weights is not None:
            ensemble = np.average(np.array(list(predictions.values())), axis=0, weights=weights)
        else:
            ensemble = np.mean(np.array(list(predictions.values())), axis=0)

        dates = self.sequence_metadata.get("target_dates", self.data.index[self.sequence_length :])
        return {
            "dates": pd.DatetimeIndex(dates).strftime("%Y-%m-%d").tolist(),
            "predictions": {
                "ensemble": ensemble.tolist(),
                **{k: v.tolist() for k, v in predictions.items()},
            },
        }

    def predict_future(self, days: int = 30):
        if not self.models:
            raise ValueError("Model not trained")

        recent = self.data[self.feature_columns].iloc[-self.sequence_length :].values
        seq = self.scalers["feature"].transform(recent)
        last_close = float(self.data["Close"].iloc[-1])

        future_prices = {k: [] for k in self.models.keys()}

        for _ in range(days):
            current_input = torch.FloatTensor(seq[-self.sequence_length :]).unsqueeze(0).to(device)
            day_scaled_preds = {}
            for model_name, info in self.models.items():
                model = info["model"]
                model.eval()
                with torch.no_grad():
                    if model_name == "attention_lstm":
                        out, _ = model(current_input)
                    else:
                        out = model(current_input)
                day_scaled_preds[model_name] = float(out.squeeze().cpu().numpy())

            avg_scaled = np.mean(list(day_scaled_preds.values()))
            next_row = seq[-1].copy()
            next_row[0] = avg_scaled
            seq = np.vstack([seq[1:], next_row])

            for model_name, scaled_pred in day_scaled_preds.items():
                log_ret = self.scalers["target"].inverse_transform(np.array([[scaled_pred]])).item()
                base = future_prices[model_name][-1] if future_prices[model_name] else last_close
                future_prices[model_name].append(base * math.exp(log_ret))

        all_preds = np.array(list(future_prices.values()))
        ensemble_future = np.mean(all_preds, axis=0)
        future_dates = pd.date_range(start=self.data.index[-1] + timedelta(days=1), periods=days, freq="D")
        return future_dates, ensemble_future, {k: np.array(v) for k, v in future_prices.items()}

    def predict_future_with_ci(self, days: int = 30, ci: float = 0.95):
        future_dates, ensemble_future, per_model = self.predict_future(days=days)
        all_preds = np.array(list(per_model.values()))
        if all_preds.size == 0:
            raise ValueError("No model predictions available")

        std = np.std(all_preds, axis=0)
        z_map = {0.9: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_map.get(ci, 1.96)
        lower = ensemble_future - z * std
        upper = ensemble_future + z * std
        return future_dates, ensemble_future, lower, upper, per_model

    def generate_trading_signals(self):
        if not self.predictions:
            print("Run evaluate_ensemble_models first")
            return None
        current = float(self.predictions["actual_prices"][-1])
        predicted = float(self.predictions["ensemble_weighted_prices"][-1])
        change = (predicted - current) / max(current, 1e-8)

        if change > 0.02:
            signal = "强烈买入"
        elif change > 0.005:
            signal = "买入"
        elif change < -0.02:
            signal = "强烈卖出"
        elif change < -0.005:
            signal = "卖出"
        else:
            signal = "持有"

        return {
            "signal": signal,
            "confidence": float(min(abs(change) * 10, 1.0) if signal != "持有" else 0.5),
            "current_price": current,
            "predicted_price": predicted,
            "expected_change": float(change),
        }

    def walk_forward_validation(self, folds: int = 5, min_train_size: int = 180, test_size: int = 30, epochs: int = 20):
        bundle = self._build_raw_sequences(target_col="Log_Returns")
        n_features = bundle.X_raw.shape[-1]
        results = []
        start = min_train_size

        for fold in range(folds):
            if start + test_size > len(bundle.X_raw):
                break

            train_end = start
            test_end = start + test_size
            X_train_raw = bundle.X_raw[:train_end]
            y_train_raw = bundle.y_raw[:train_end]
            X_test_raw = bundle.X_raw[train_end:test_end]
            y_test_raw = bundle.y_raw[train_end:test_end]

            f_scaler = StandardScaler()
            t_scaler = StandardScaler()
            f_scaler.fit(X_train_raw.reshape(-1, n_features))
            X_train = f_scaler.transform(X_train_raw.reshape(-1, n_features)).reshape(X_train_raw.shape)
            X_test = f_scaler.transform(X_test_raw.reshape(-1, n_features)).reshape(X_test_raw.shape)
            y_train = t_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
            y_test = t_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()

            train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=64, shuffle=True)
            test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=64, shuffle=False)

            fold_preds = []
            for model_name, cfg in self._model_configs().items():
                model = cfg["model"].to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.MSELoss()

                for _ in range(epochs):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        optimizer.zero_grad()
                        if model_name == "attention_lstm":
                            out, _ = model(batch_X)
                        else:
                            out = model(batch_X)
                        loss = criterion(out.squeeze(), batch_y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                model.eval()
                preds = []
                with torch.no_grad():
                    for batch_X, _ in test_loader:
                        batch_X = batch_X.to(device)
                        if model_name == "attention_lstm":
                            out, _ = model(batch_X)
                        else:
                            out = model(batch_X)
                        preds.extend(out.squeeze().cpu().numpy())
                fold_preds.append(np.array(preds))

            ensemble_scaled = np.mean(np.array(fold_preds), axis=0)
            y_test_log = t_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            pred_log = t_scaler.inverse_transform(ensemble_scaled.reshape(-1, 1)).flatten()

            prev_close = bundle.prev_closes[train_end:test_end]
            y_test_price = self._returns_to_prices(y_test_log, prev_close)
            pred_price = self._returns_to_prices(pred_log, prev_close)

            rmse = float(np.sqrt(mean_squared_error(y_test_price, pred_price)))
            mae = float(mean_absolute_error(y_test_price, pred_price))
            direction = float(np.mean((y_test_log > 0) == (pred_log > 0)))

            results.append(
                {
                    "fold": fold + 1,
                    "train_end_date": str(bundle.dates[train_end - 1].date()),
                    "test_end_date": str(bundle.dates[test_end - 1].date()),
                    "price_rmse": rmse,
                    "price_mae": mae,
                    "direction_acc": direction,
                }
            )

            print(f"Fold {fold + 1}: RMSE={rmse:.4f}, MAE={mae:.4f}, DirectionAcc={direction:.4f}")
            start += test_size

        if results:
            print(
                "Walk-forward avg -> "
                f"RMSE={np.mean([r['price_rmse'] for r in results]):.4f}, "
                f"MAE={np.mean([r['price_mae'] for r in results]):.4f}, "
                f"DirectionAcc={np.mean([r['direction_acc'] for r in results]):.4f}"
            )
        return results


def main():
    predictor = EnhancedStockPredictor(symbol="AAPL", period="2y", sequence_length=60)
    predictor.fetch_and_prepare_data()
    X_test, y_test = predictor.train_ensemble_models(epochs=50)
    predictor.evaluate_ensemble_models(X_test, y_test)
    predictor.plot_comprehensive_results()
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", predictor.symbol)
    predictor.save_artifacts(model_dir)


if __name__ == "__main__":
    main()
