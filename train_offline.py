import argparse
import os

import torch

import main_lstm
from main_lstm import EnhancedStockPredictor
from data_manager import data_manager


def main():
    parser = argparse.ArgumentParser(description="Train offline models and save artifacts.")
    parser.add_argument("--symbol", help="Stock symbol, e.g. AAPL")
    parser.add_argument("--all", action="store_true", help="Train all stocks from data_manager")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-every", type=int, default=10, help="Print train/val loss every N epochs")
    args = parser.parse_args()

    # Override global device used by main_lstm
    main_lstm.device = main_lstm._select_device(args.device)
    print(f"Using device: {main_lstm.device}")
    if main_lstm.device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"CUDA total memory: {total_mem:.2f} GB")
        except Exception:
            pass

    if not args.all and not args.symbol:
        parser.error("Specify --symbol or --all")

    symbols = data_manager.get_stock_list() if args.all else [args.symbol]

    for symbol in symbols:
        try:
            predictor = EnhancedStockPredictor(symbol=symbol, sequence_length=args.sequence_length)
            predictor.fetch_and_prepare_data()
            X_test, y_test = predictor.train_ensemble_models(epochs=args.epochs, log_every=args.log_every)
            predictor.evaluate_ensemble_models(X_test, y_test)

            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", symbol)
            predictor.save_artifacts(model_dir)
            print(f"Saved model artifacts to {model_dir}")
        except Exception as exc:
            print(f"Failed to train {symbol}: {exc}")


if __name__ == "__main__":
    main()
