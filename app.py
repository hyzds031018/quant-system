
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from werkzeug.security import check_password_hash, generate_password_hash
import utils
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from data_manager import initialize_data, data_manager
import threading
import time
import uuid
import json
import os

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET", "change-me")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "0") == "1",
)

ENABLE_LOGIN = os.environ.get("ENABLE_LOGIN", "1") != "0"
APP_USER = os.environ.get("APP_USER", "admin")
APP_PASSWORD = os.environ.get("APP_PASSWORD")
APP_PASSWORD_HASH = os.environ.get("APP_PASSWORD_HASH")
AUTH_FILE = os.environ.get("AUTH_FILE", ".auth.json")


def _password_configured():
    return bool(_get_auth_password_hash() or APP_PASSWORD_HASH or APP_PASSWORD)


def _check_login(username, password):
    if username != APP_USER:
        return False
    file_hash = _get_auth_password_hash()
    if file_hash:
        return check_password_hash(file_hash, password or "")
    if APP_PASSWORD_HASH:
        return check_password_hash(APP_PASSWORD_HASH, password or "")
    if APP_PASSWORD:
        return password == APP_PASSWORD
    return False


def _load_auth_file():
    try:
        if not AUTH_FILE:
            return {}
        if not os.path.exists(AUTH_FILE):
            return {}
        with open(AUTH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _get_auth_password_hash():
    data = _load_auth_file()
    return data.get("password_hash")


def _save_auth_password_hash(password_hash):
    if not AUTH_FILE:
        raise RuntimeError("AUTH_FILE not configured")
    data = {
        "password_hash": password_hash,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(AUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


jobs = {}
jobs_lock = threading.Lock()
cache_lock = threading.Lock()
response_cache = {}
CACHE_TTL_SECONDS = 300


def init_data_background():
    initialize_data()


_init_started = False


def _start_init_once():
    global _init_started
    if _init_started:
        return
    _init_started = True
    t = threading.Thread(target=init_data_background, daemon=True)
    t.start()


_start_init_once()


def _parse_int(value, default, min_value=None, max_value=None):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    if min_value is not None:
        v = max(min_value, v)
    if max_value is not None:
        v = min(max_value, v)
    return v


def _parse_float(value, default, min_value=None, max_value=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if min_value is not None:
        v = max(min_value, v)
    if max_value is not None:
        v = min(max_value, v)
    return v


def _parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _prune_jobs(max_jobs=50, max_age_seconds=86400):
    now = time.time()
    with jobs_lock:
        to_delete = []
        for job_id, job in jobs.items():
            created_at = job.get("created_at", now)
            if (now - created_at) > max_age_seconds:
                to_delete.append(job_id)
        for job_id in to_delete:
            jobs.pop(job_id, None)
        if len(jobs) > max_jobs:
            ordered = sorted(jobs.items(), key=lambda kv: kv[1].get("created_at", now))
            for job_id, _ in ordered[: max(0, len(jobs) - max_jobs)]:
                jobs.pop(job_id, None)


@app.before_request
def _require_login():
    if not ENABLE_LOGIN:
        return None
    path = request.path or ""
    if path.startswith("/static/"):
        return None
    if path in ("/login", "/logout", "/healthz"):
        return None
    if session.get("user"):
        return None
    if path.startswith("/api/"):
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    return redirect(url_for("login", next=path))


@app.route("/login", methods=["GET", "POST"])
def login():
    if not ENABLE_LOGIN:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        if not _password_configured():
            error = "未配置登录密码，请设置 APP_PASSWORD 或 APP_PASSWORD_HASH"
        else:
            username = (request.form.get("username") or "").strip()
            password = request.form.get("password") or ""
            if _check_login(username, password):
                session["user"] = username
                next_url = request.args.get("next") or url_for("index")
                return redirect(next_url)
            error = "账号或密码错误"
    return render_template("login.html", error=error, username=APP_USER or "")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/admin", methods=["GET", "POST"])
def admin():
    message = None
    error = None
    if request.method == "POST":
        if APP_PASSWORD_HASH:
            error = "已设置 APP_PASSWORD_HASH，在线修改已禁用"
        else:
            current_pw = request.form.get("current_password") or ""
            new_pw = request.form.get("new_password") or ""
            confirm_pw = request.form.get("confirm_password") or ""
            if not _password_configured():
                error = "未配置登录密码，请先设置 APP_PASSWORD 或 APP_PASSWORD_HASH"
            elif not _check_login(APP_USER, current_pw):
                error = "当前密码错误"
            elif len(new_pw) < 6:
                error = "新密码至少 6 位"
            elif new_pw != confirm_pw:
                error = "两次输入的新密码不一致"
            else:
                _save_auth_password_hash(generate_password_hash(new_pw))
                message = "密码已更新"
    return render_template(
        "admin.html",
        message=message,
        error=error,
        username=APP_USER or "",
        auth_file=AUTH_FILE,
        hash_locked=bool(APP_PASSWORD_HASH),
    )


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stocks")
def get_stocks():
    return jsonify(utils.get_stock_list())


@app.route("/api/stock/<stock_code>")
def get_stock_kline(stock_code):
    try:
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "stock_data",
            f"{stock_code}.csv",
        )
        data_mtime = os.path.getmtime(data_path) if os.path.exists(data_path) else None
        cache_key = f"kline:{stock_code}"
        now = time.time()
        with cache_lock:
            cached = response_cache.get(cache_key)
            if cached:
                if (
                    (now - cached["ts"]) < CACHE_TTL_SECONDS
                    and cached["data_mtime"] == data_mtime
                ):
                    return jsonify(cached["payload"])

        stock_data = utils.get_stock_data(stock_code)
        if stock_data is None or stock_data.empty:
            return jsonify({"error": "Unable to obtain stock data", "success": False})

        numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_columns:
            if col not in stock_data.columns:
                return jsonify({"error": f"Data missing {col} column", "success": False})
            stock_data[col] = pd.to_numeric(stock_data[col], errors="coerce")

        stock_data = stock_data.dropna(subset=numeric_columns)
        if len(stock_data) == 0:
            return jsonify({"error": "No valid data after data cleaning", "success": False})

        stock_data["MA5"] = stock_data["Close"].rolling(window=5).mean()
        stock_data["MA10"] = stock_data["Close"].rolling(window=10).mean()
        stock_data["MA20"] = stock_data["Close"].rolling(window=20).mean()

        from main_lstm import EnhancedStockPredictor

        predictor = EnhancedStockPredictor(symbol=stock_code)
        predictor.fetch_and_prepare_data()
        model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models", stock_code
        )
        if not os.path.exists(model_dir):
            return jsonify({"error": "Model not trained. Train offline and save artifacts first.", "success": False})
        predictor.load_artifacts(model_dir)
        predictor.prepare_inference()
        historical_predictions = predictor.predict_historical()

        kline_data = []
        for idx, row in stock_data.iterrows():
            date_str = idx.strftime("%Y-%m-%d")
            data_point = {
                "time": date_str,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
                "ma5": float(row["MA5"]) if not pd.isna(row["MA5"]) else None,
                "ma10": float(row["MA10"]) if not pd.isna(row["MA10"]) else None,
                "ma20": float(row["MA20"]) if not pd.isna(row["MA20"]) else None,
            }

            if date_str in historical_predictions["dates"]:
                pred_idx = historical_predictions["dates"].index(date_str)
                data_point["predicted"] = {
                    "ensemble": historical_predictions["predictions"]["ensemble"][pred_idx],
                    "attention_lstm": historical_predictions["predictions"]["attention_lstm"][pred_idx],
                    "gru": historical_predictions["predictions"]["gru"][pred_idx],
                    "transformer": historical_predictions["predictions"]["transformer"][pred_idx],
                }

            kline_data.append(data_point)

        result = {
            "success": True,
            "code": stock_code,
            "dates": [item["time"] for item in kline_data],
            "kline": kline_data,
        }

        with cache_lock:
            response_cache[cache_key] = {
                "ts": now,
                "data_mtime": data_mtime,
                "payload": result,
            }

        app.logger.info(f"Successfully processed data for {stock_code} with {len(kline_data)} records")
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error processing stock data for {stock_code}: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/prediction/<stock_code>")
def get_prediction(stock_code):
    prediction_data = utils.get_data5(stock_code)
    return jsonify(prediction_data)


@app.route("/api/future/<stock_code>")
def get_future_prediction(stock_code):
    try:
        days = _parse_int(request.args.get("days"), 7, min_value=1, max_value=365)
        ci = _parse_float(request.args.get("ci"), 0.95, min_value=0.5, max_value=0.999)
        from main_lstm import EnhancedStockPredictor

        predictor = EnhancedStockPredictor(symbol=stock_code)
        predictor.fetch_and_prepare_data()
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", stock_code)
        if not os.path.exists(model_dir):
            return jsonify({"success": False, "error": "Model not trained"})
        predictor.load_artifacts(model_dir)
        predictor.prepare_inference()

        dates, ensemble, lower, upper, per_model = predictor.predict_future_with_ci(days=days, ci=ci)

        return jsonify({
            "success": True,
            "dates": dates.strftime("%Y-%m-%d").tolist(),
            "ensemble": ensemble.tolist(),
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "models": {k: v.tolist() for k, v in per_model.items()},
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/dcf/<stock_code>")
def get_dcf_valuation(stock_code):
    try:
        wacc = _parse_float(request.args.get("wacc"), 0.1, min_value=0.0, max_value=0.5)
        g = _parse_float(request.args.get("g"), 0.02, min_value=-0.1, max_value=0.2)
        reinvestment_rate = _parse_float(request.args.get("reinvestment_rate"), 0.4, min_value=0.0, max_value=1.0)
        dcf_data = utils.get_data6(wacc, g, reinvestment_rate)
        return jsonify(dcf_data)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/technical_indicators/<stock_code>")
def get_technical_indicators(stock_code):
    try:
        stock_data = utils.get_stock_data(stock_code)

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        if (
            stock_data is None
            or stock_data.empty
            or not all(col in stock_data.columns for col in required_columns)
        ):
            error_msg = f"stock {stock_code} data missing or invalid"
            return jsonify({"error": error_msg})

        min_rows_required = 60
        if len(stock_data) < min_rows_required:
            error_msg = f"stock {stock_code} Insufficient number of data rows"
            return jsonify({"error": error_msg})

        df = stock_data.copy()
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=required_columns, inplace=True)
        if len(df) < min_rows_required:
            error_msg = f"stock {stock_code} Insufficient number of data rows"
            return jsonify({"error": error_msg})

        df["MA5"] = df["Close"].rolling(window=5).mean()
        df["MA10"] = df["Close"].rolling(window=10).mean()
        df["MA20"] = df["Close"].rolling(window=20).mean()

        high_low = df["High"] - df["Low"]
        high_prev_close = np.abs(df["High"] - df["Close"].shift(1))
        low_prev_close = np.abs(df["Low"] - df["Close"].shift(1))
        tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        df["ATR"] = tr.ewm(span=14, adjust=False).mean()

        low_14 = df["Low"].rolling(window=14).min()
        high_14 = df["High"].rolling(window=14).max()
        df["STOCH_K"] = ((df["Close"] - low_14) / (high_14 - low_14)) * 100
        df["STOCH_D"] = df["STOCH_K"].rolling(window=3).mean()
        df["WILLR"] = ((high_14 - df["Close"]) / (high_14 - low_14)) * -100
        df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        df["RSI"] = df["RSI"].fillna(50)

        def process_series(series, round_digits=2):
            return [round(x, round_digits) if pd.notnull(x) else None for x in series.tolist()]

        result = {
            "dates": df.index.strftime("%Y-%m-%d").tolist(),
            "price": {
                "close": process_series(df["Close"]),
                "ma5": process_series(df["MA5"]),
                "ma10": process_series(df["MA10"]),
                "ma20": process_series(df["MA20"]),
            },
            "indicators": {
                "rsi": process_series(df["RSI"]),
                "atr": process_series(df["ATR"]),
                "stoch": {"k": process_series(df["STOCH_K"]), "d": process_series(df["STOCH_D"])},
                "willr": process_series(df["WILLR"]),
                "obv": process_series(df["OBV"], 0),
            },
        }

        return jsonify(result)
    except Exception as e:
        error_message = f"Calculation of technical indicators is wrong: {str(e)}"
        return jsonify({"error": error_message})


@app.route("/api/risk_assessment/<stock_code>")
def get_risk_assessment(stock_code):
    risk_data = {
        "volatility": 15.6,
        "sharpe_ratio": 1.2,
        "beta": 1.15,
        "max_drawdown": 12.5,
        "risk_level": "Medium",
    }
    return jsonify(risk_data)


@app.route("/api/market_overview")
def get_market_overview():
    try:
        statistics = utils.get_market_statistics()
        if not statistics:
            return jsonify({
                "error": "Failed to get market statistics",
                "highest_close_stock": "N/A",
                "highest_close_price": 0,
                "avg_close_price": 0,
                "highest_pct_change_stock": "N/A",
                "highest_pct_change": 0,
                "latest_date": datetime.now().strftime("%Y-%m-%d"),
            })

        result = {
            "highest_close_stock": statistics.get("highest_close_stock", "N/A"),
            "highest_close_price": float(statistics.get("highest_close_price", 0)),
            "avg_close_price": float(statistics.get("avg_close_price", 0)),
            "highest_pct_change_stock": statistics.get("highest_pct_change_stock", "N/A"),
            "highest_pct_change": float(statistics.get("highest_pct_change", 0)),
            "latest_date": statistics.get("latest_date", datetime.now().strftime("%Y-%m-%d")),
        }
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error getting market overview: {str(e)}")
        return jsonify({
            "error": str(e),
            "highest_close_stock": "N/A",
            "highest_close_price": 0,
            "avg_close_price": 0,
            "highest_pct_change_stock": "N/A",
            "highest_pct_change": 0,
            "latest_date": datetime.now().strftime("%Y-%m-%d"),
        })


@app.route("/api/opportunities")
def get_opportunities():
    try:
        top_n = _parse_int(request.args.get("top_n"), 5, min_value=1, max_value=50)
        bottom_n = _parse_int(request.args.get("bottom_n"), 5, min_value=1, max_value=50)
        momentum_window = _parse_int(request.args.get("momentum"), 5, min_value=1, max_value=60)
        momentum_min = _parse_float(request.args.get("momentum_min"), 0.0, min_value=0.0, max_value=1.0)
        momentum_abs = _parse_bool(request.args.get("momentum_abs"), False)
        lb1 = _parse_int(request.args.get("lb1"), 5, min_value=2, max_value=60)
        lb2 = _parse_int(request.args.get("lb2"), 10, min_value=2, max_value=120)
        lb3 = _parse_int(request.args.get("lb3"), 20, min_value=2, max_value=252)
        vol_series_window = _parse_int(request.args.get("vol_window"), 20, min_value=2, max_value=252)
        vol_series_length = _parse_int(request.args.get("vol_len"), 20, min_value=1, max_value=252)

        stocks_raw = request.args.get("stocks") or request.args.get("pool")
        industry_filter = request.args.get("industry")
        stock_list = None
        if stocks_raw:
            stock_list = [s.strip().upper() for s in stocks_raw.split(",") if s.strip()]
            stock_list = list(dict.fromkeys(stock_list))
            if not stock_list:
                return jsonify({"success": False, "error": "股票池为空"})

        if industry_filter:
            industry_key = industry_filter.strip().lower()
            if industry_key:
                try:
                    with open("stock_info.json", "r", encoding="utf-8") as f:
                        stock_info = json.load(f)
                except Exception as exc:
                    return jsonify({"success": False, "error": f"读取 stock_info.json 失败: {exc}"})

                if stock_list is None:
                    stock_list = data_manager.get_stock_list()

                filtered = []
                for symbol in stock_list:
                    info = stock_info.get(symbol, {})
                    industry = str(info.get("industry", "")).lower()
                    if industry_key in industry:
                        filtered.append(symbol)
                stock_list = filtered

        from strategies import find_volatility_opportunities

        data = find_volatility_opportunities(
            stock_list=stock_list,
            top_n=top_n,
            bottom_n=bottom_n,
            lookbacks=(lb1, lb2, lb3),
            momentum_window=momentum_window,
            momentum_min=momentum_min,
            momentum_abs=momentum_abs,
            vol_series_window=vol_series_window,
            vol_series_length=vol_series_length,
        )
        filters = {}
        if stocks_raw:
            filters["pool"] = stocks_raw
        if industry_filter:
            filters["industry"] = industry_filter
        pool_size = None
        if stock_list is not None and (stocks_raw or industry_filter):
            pool_size = len(stock_list)
        return jsonify(
            {
                "success": True,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "long": data["long"],
                "short": data["short"],
                "meta": {
                    "universe": data["universe"],
                    "scanned": data["scanned"],
                    "lookbacks": data["lookbacks"],
                    "momentum_window": data["momentum_window"],
                    "momentum_min": data.get("momentum_min", momentum_min),
                    "momentum_abs": data.get("momentum_abs", momentum_abs),
                    "vol_series_window": data.get("vol_series_window", vol_series_window),
                    "vol_series_length": data.get("vol_series_length", vol_series_length),
                    "filters": filters,
                    "pool_size": pool_size,
                },
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/model_evaluation/<stock_code>")
def get_model_evaluation(stock_code):
    hash_val = hash(stock_code)
    arima_rmse = round(0.5 + (hash_val % 10) * 0.01, 4)
    arima_mae = round(0.3 + (hash_val % 10) * 0.01, 4)
    arima_mape = round(1.5 + (hash_val % 10) * 0.05, 2)
    lstm_rmse = round(0.3 + (hash_val % 10) * 0.005, 4)
    lstm_mae = round(0.2 + (hash_val % 10) * 0.005, 4)
    lstm_mape = round(1.0 + (hash_val % 10) * 0.03, 2)

    evaluation_data = {
        "arima": {"rmse": arima_rmse, "mae": arima_mae, "mape": f"{arima_mape}%"},
        "lstm_attention": {"rmse": lstm_rmse, "mae": lstm_mae, "mape": f"{lstm_mape}%"},
        "transformer": {"rmse": round(lstm_rmse * 0.9, 4), "mae": round(lstm_mae * 0.9, 4), "mape": f"{round(lstm_mape * 0.9, 2)}%"},
    }
    return jsonify(evaluation_data)

@app.route("/api/update_data", methods=["GET", "POST"])
def update_data():
    period = request.args.get("period", "refresh")
    batch_size = _parse_int(request.args.get("batch_size"), 5, min_value=1, max_value=50)
    data_manager.update_all_stocks(period=period, batch_size=batch_size)
    return jsonify({"message": "Data updated"})


@app.route("/api/update_and_train/<stock_code>", methods=["GET", "POST"])
def update_and_train(stock_code):
    try:
        from main_lstm import EnhancedStockPredictor

        period = request.args.get("period", "max")
        epochs = _parse_int(request.args.get("epochs"), 50, min_value=1, max_value=500)
        sequence_length = _parse_int(request.args.get("sequence_length"), 60, min_value=10, max_value=500)

        data = data_manager.get_stock_data(stock_code, period=period)
        if data is None or data.empty:
            return jsonify({"success": False, "error": "Failed to fetch stock data"})
        data_manager.save_stock_data(stock_code, data)

        predictor = EnhancedStockPredictor(symbol=stock_code, sequence_length=sequence_length)
        predictor.fetch_and_prepare_data()
        X_test, y_test = predictor.train_ensemble_models(epochs=epochs)
        predictor.evaluate_ensemble_models(X_test, y_test)

        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", stock_code)
        predictor.save_artifacts(model_dir)

        return jsonify({"success": True, "message": "Data updated and model retrained", "model_dir": model_dir})
    except Exception as e:
        app.logger.error(f"Update and train error for {stock_code}: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/update_and_train_all", methods=["GET", "POST"])
def update_and_train_all():
    try:
        from main_lstm import EnhancedStockPredictor

        period = request.args.get("period", "max")
        epochs = _parse_int(request.args.get("epochs"), 50, min_value=1, max_value=500)
        sequence_length = _parse_int(request.args.get("sequence_length"), 60, min_value=10, max_value=500)

        results = []
        for stock_code in data_manager.get_stock_list():
            try:
                data = data_manager.get_stock_data(stock_code, period=period)
                if data is None or data.empty:
                    results.append({"stock": stock_code, "success": False, "error": "No data"})
                    continue
                data_manager.save_stock_data(stock_code, data)

                predictor = EnhancedStockPredictor(symbol=stock_code, sequence_length=sequence_length)
                predictor.fetch_and_prepare_data()
                X_test, y_test = predictor.train_ensemble_models(epochs=epochs)
                predictor.evaluate_ensemble_models(X_test, y_test)

                model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", stock_code)
                predictor.save_artifacts(model_dir)
                results.append({"stock": stock_code, "success": True, "model_dir": model_dir})
            except Exception as exc:
                results.append({"stock": stock_code, "success": False, "error": str(exc)})

        return jsonify({"success": True, "results": results})
    except Exception as e:
        app.logger.error(f"Batch update/train error: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


def _run_batch_job(job_id, period, epochs, sequence_length):
    from main_lstm import EnhancedStockPredictor

    stock_list = data_manager.get_stock_list()
    results = []

    for idx, stock_code in enumerate(stock_list, start=1):
        try:
            data = data_manager.get_stock_data(stock_code, period=period)
            if data is None or data.empty:
                result = {"stock": stock_code, "success": False, "error": "No data"}
            else:
                data_manager.save_stock_data(stock_code, data)
                predictor = EnhancedStockPredictor(symbol=stock_code, sequence_length=sequence_length)
                predictor.fetch_and_prepare_data()
                X_test, y_test = predictor.train_ensemble_models(epochs=epochs)
                predictor.evaluate_ensemble_models(X_test, y_test)
                model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", stock_code)
                predictor.save_artifacts(model_dir)
                result = {"stock": stock_code, "success": True, "model_dir": model_dir}
        except Exception as exc:
            result = {"stock": stock_code, "success": False, "error": str(exc)}

        results.append(result)
        with jobs_lock:
            jobs[job_id]["completed"] = idx
            jobs[job_id]["results"] = results
            jobs[job_id]["updated_at"] = time.time()

    with jobs_lock:
        jobs[job_id]["status"] = "finished"


@app.route("/api/update_and_train_all_async", methods=["GET", "POST"])
def update_and_train_all_async():
    period = request.args.get("period", "max")
    epochs = _parse_int(request.args.get("epochs"), 50, min_value=1, max_value=500)
    sequence_length = _parse_int(request.args.get("sequence_length"), 60, min_value=10, max_value=500)

    _prune_jobs()

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "status": "running",
            "total": 0,
            "completed": 0,
            "results": [],
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    def _init_and_run():
        total = len(data_manager.get_stock_list())
        with jobs_lock:
            jobs[job_id]["total"] = total
        _run_batch_job(job_id, period, epochs, sequence_length)

    t = threading.Thread(target=_init_and_run, daemon=True)
    t.start()
    return jsonify({"success": True, "job_id": job_id})


@app.route("/api/job/<job_id>")
def get_job_status(job_id):
    _prune_jobs()
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"})
    return jsonify({"success": True, "job": job})


@app.route("/api/stock_info/<stock_code>")
def get_stock_basic_info(stock_code):
    try:
        with open("stock_info.json", "r", encoding="utf-8") as f:
            stock_info = json.load(f)

        if stock_code in stock_info:
            return jsonify(stock_info[stock_code])
        else:
            return jsonify({
                "error": "Stock information not found",
                "company_name": "N/A",
                "industry": "N/A",
                "market_cap": "N/A",
                "pe_ratio": "N/A",
                "dividend_yield": "N/A",
                "eps": "N/A",
                "revenue": "N/A",
                "employees": "N/A",
                "founded": "N/A",
                "headquarters": "N/A",
                "description": "No company description available.",
            })
    except Exception as e:
        app.logger.error(f"Error reading stock information: {str(e)}")
        return jsonify({
            "error": f"Failed to read stock information: {str(e)}",
            "company_name": "Error",
            "industry": "N/A",
            "market_cap": "N/A",
            "pe_ratio": "N/A",
            "dividend_yield": "N/A",
            "eps": "N/A",
            "revenue": "N/A",
            "employees": "N/A",
            "founded": "N/A",
            "headquarters": "N/A",
            "description": "Error loading company information.",
        })


@app.route("/api/metrics/<stock_code>")
def get_model_metrics(stock_code):
    try:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", stock_code)
        metrics_path = os.path.join(model_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            return jsonify({"error": "metrics.json not found", "success": False})
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        return jsonify({"success": True, "metrics": metrics})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/model_info/<stock_code>")
def get_model_info(stock_code):
    try:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", stock_code)
        info_path = os.path.join(model_dir, "model_info.json")
        if not os.path.exists(info_path):
            return jsonify({"error": "model_info.json not found", "success": False})
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        return jsonify({"success": True, "info": info})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/strategy/ma_backtest", methods=["POST"])
def ma_backtest():
    try:
        data = request.json or {}
        stock = data.get("stock")
        short_window = _parse_int(data.get("short_window"), 20, min_value=1, max_value=500)
        long_window = _parse_int(data.get("long_window"), 50, min_value=1, max_value=1000)
        initial_capital = _parse_float(data.get("initial_capital"), 100000.0, min_value=0.0)
        commission = _parse_float(data.get("commission"), 0.0003, min_value=0.0, max_value=0.1)

        stock_data = data_manager.load_stock_data(stock)
        if stock_data is None or stock_data.empty:
            return jsonify({"success": False, "error": "No data found"})

        prices = stock_data["Close"].copy()
        prices.index = pd.to_datetime(stock_data["Date"])

        from strategies import MovingAverageStrategy

        strategy = MovingAverageStrategy(short_window, long_window)
        results = strategy.backtest(prices, initial_capital=initial_capital, commission=commission)

        return jsonify({
            "success": True,
            "metrics": results.metrics,
            "dates": results.signals.index.strftime("%Y-%m-%d").tolist(),
            "cumulative_returns": results.cumulative_returns.fillna(1.0).tolist(),
            "signals": results.signals["signal"].fillna(0).tolist(),
            "price": stock_data["Close"].tolist(),
            "short_ma": results.signals["short_mavg"].fillna(0).tolist(),
            "long_ma": results.signals["long_mavg"].fillna(0).tolist(),
            "trades": results.trades,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/strategy/macd_backtest", methods=["POST"])
def macd_backtest():
    try:
        data = request.json or {}
        stock = data.get("stock")
        fast = _parse_int(data.get("fast"), 12, min_value=1, max_value=200)
        slow = _parse_int(data.get("slow"), 26, min_value=1, max_value=400)
        signal = _parse_int(data.get("signal"), 9, min_value=1, max_value=200)
        initial_capital = _parse_float(data.get("initial_capital"), 100000.0, min_value=0.0)
        commission = _parse_float(data.get("commission"), 0.0003, min_value=0.0, max_value=0.1)

        stock_data = data_manager.load_stock_data(stock)
        if stock_data is None or stock_data.empty:
            return jsonify({"success": False, "error": "No data found"})

        prices = stock_data["Close"].copy()
        prices.index = pd.to_datetime(stock_data["Date"])

        from strategies import MACDStrategy

        strategy = MACDStrategy(fast, slow, signal)
        results = strategy.backtest(prices, initial_capital, commission)

        return jsonify({
            "success": True,
            "metrics": results.metrics,
            "dates": results.signals.index.strftime("%Y-%m-%d").tolist(),
            "cumulative_returns": results.cumulative_returns.fillna(1.0).tolist(),
            "signals": results.signals["signal"].fillna(0).tolist(),
            "price": stock_data["Close"].tolist(),
            "indicator_1": results.signals["macd"].fillna(0).tolist(),
            "indicator_2": results.signals["signal_line"].fillna(0).tolist(),
            "indicator_3": results.signals["histogram"].fillna(0).tolist(),
            "trades": results.trades,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/strategy/rsi_backtest", methods=["POST"])
def rsi_backtest():
    try:
        data = request.json or {}
        stock = data.get("stock")
        window = _parse_int(data.get("window"), 14, min_value=1, max_value=200)
        overbought = _parse_int(data.get("overbought"), 70, min_value=1, max_value=100)
        oversold = _parse_int(data.get("oversold"), 30, min_value=0, max_value=99)
        initial_capital = _parse_float(data.get("initial_capital"), 100000.0, min_value=0.0)
        commission = _parse_float(data.get("commission"), 0.0003, min_value=0.0, max_value=0.1)

        stock_data = data_manager.load_stock_data(stock)
        if stock_data is None or stock_data.empty:
            return jsonify({"success": False, "error": "No data found"})

        prices = stock_data["Close"].copy()
        prices.index = pd.to_datetime(stock_data["Date"])

        from strategies import RSIStrategy

        strategy = RSIStrategy(window, overbought, oversold)
        results = strategy.backtest(prices, initial_capital, commission)

        return jsonify({
            "success": True,
            "metrics": results.metrics,
            "dates": results.signals.index.strftime("%Y-%m-%d").tolist(),
            "cumulative_returns": results.cumulative_returns.fillna(1.0).tolist(),
            "signals": results.signals["signal"].fillna(0).tolist(),
            "price": stock_data["Close"].tolist(),
            "indicator_1": results.signals["rsi"].fillna(50).tolist(),
            "indicator_2": [overbought] * len(stock_data),
            "indicator_3": [oversold] * len(stock_data),
            "trades": results.trades,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/portfolio/optimize", methods=["POST"])
def optimize_portfolio():
    try:
        data = request.json or {}
        stocks = data.get("stocks", [])
        if not stocks or len(stocks) < 2:
            return jsonify({"success": False, "error": "Need at least 2 stocks"})

        close_prices = pd.DataFrame()
        for stock in stocks:
            df = utils.get_stock_data(stock)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                close_prices[stock] = df["Close"]

        if close_prices.empty:
            return jsonify({"success": False, "error": "No valid data fetched"})

        close_prices = close_prices.dropna()
        returns = close_prices.pct_change().dropna()

        from strategies import PortfolioOptimizer

        optimizer = PortfolioOptimizer(returns)
        result = optimizer.maximize_sharpe_ratio()

        if result:
            cleaned_result = {
                "weights": result["weights"],
                "return": result["return"],
                "volatility": result["volatility"],
                "sharpe": result["sharpe"],
            }
            return jsonify({"success": True, "result": cleaned_result})
        else:
            return jsonify({"success": False, "error": "Optimization failed"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/volatility/target/<stock_code>")
def vol_targeting(stock_code):
    try:
        target_vol = _parse_float(request.args.get("target"), 0.15, min_value=0.01, max_value=1.0)
        stock_data = utils.get_stock_data(stock_code)
        if stock_data is None or stock_data.empty:
            return jsonify({"success": False, "error": "No data"})

        from strategies import VolatilityTargetStrategy

        strategy = VolatilityTargetStrategy(target_vol)
        res = strategy.backtest(stock_data["Close"])

        return jsonify({
            "success": True,
            "metrics": res["metrics"],
            "dates": res["cumulative_returns"].index.strftime("%Y-%m-%d").tolist(),
            "cumulative_returns": res["cumulative_returns"].tolist(),
            "leverage": res["leverage"].fillna(0).tolist(),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

from quant_data import FinnhubClient

finnhub_client = FinnhubClient(api_key=os.environ.get("FINNHUB_TOKEN"))


@app.route("/api/news/<stock_code>")
def get_company_news(stock_code):
    try:
        if not finnhub_client.api_key:
            return jsonify({"success": False, "error": "未配置 FINNHUB_TOKEN，请在环境变量中设置"})
        news = finnhub_client.get_company_news(stock_code)
        return jsonify({"success": True, "news": news})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/news/market")
def get_market_news_api():
    try:
        if not finnhub_client.api_key:
            return jsonify({"success": False, "error": "未配置 FINNHUB_TOKEN，请在环境变量中设置"})
        news = finnhub_client.get_market_news()
        return jsonify({"success": True, "news": news})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/sentiment/<stock_code>")
def get_sentiment(stock_code):
    try:
        news = finnhub_client.get_company_news(stock_code)
        if not news:
            return jsonify({
                "overall_sentiment": "Neutral",
                "confidence": 0.0,
                "news_volume": [0, 0, 0, 0, 0, 0, 0],
            })

        scores = [n["sentiment"] for n in news]
        avg_score = sum(scores) / len(scores)

        if avg_score > 0.05:
            label = "Bullish"
        elif avg_score < -0.05:
            label = "Bearish"
        else:
            label = "Neutral"

        confidence = min(len(news) / 10.0, 1.0)
        if avg_score != 0:
            confidence = (abs(avg_score) * 0.5 + confidence * 0.5)

        volume = {}
        today = datetime.now().date()
        for i in range(7):
            d = (today - timedelta(days=6 - i)).strftime("%Y-%m-%d")
            volume[d] = 0

        for n in news:
            d_str = datetime.fromtimestamp(n["timestamp"]).strftime("%Y-%m-%d")
            if d_str in volume:
                volume[d_str] += 1

        vol_list = list(volume.values())

        return jsonify({
            "overall_sentiment": label,
            "confidence": round(confidence, 2),
            "news_volume": vol_list,
            "score": round(avg_score, 3),
        })
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1")
