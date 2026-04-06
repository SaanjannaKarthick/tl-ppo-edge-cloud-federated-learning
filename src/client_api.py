#!/usr/bin/env python3
import argparse
import json
import time
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def safe_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.fillna(0.0)


# =========================================================
# Model
# =========================================================
class TelemetryClassifier1DCNN(nn.Module):
    """
    15 telemetry features -> 4 classes
    Input shape: [B, 1, 15]
    """
    def __init__(self, input_len=15, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def state_dict_to_numpy(sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    return {k: v.detach().cpu().numpy().tolist() for k, v in sd.items()}


def numpy_to_state_dict(d: Dict[str, Any], device="cpu") -> Dict[str, torch.Tensor]:
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in d.items()}


# =========================================================
# Request models
# =========================================================
class TrainRequest(BaseModel):
    global_weights: Dict[str, Any]
    lr: float = 1e-3
    epochs: int = 3
    batch_size: int = 64
    seed: int = 42


class ExportShardRequest(BaseModel):
    max_rows: Optional[int] = None


# =========================================================
# Build app
# =========================================================
def build_app(args):
    app = FastAPI(title=f"Live Telemetry Client API - {args.client_id}")

    class_map = load_json(args.class_map)
    scaler = load_json(args.scaler_path)

    feature_cols = scaler["feature_cols"]
    label_col = "scenario"

    # single-client CSV on each node
    df = pd.read_csv(args.csv)
    if label_col not in df.columns:
        raise RuntimeError(f"CSV missing label column: {label_col}")

    # make sure the client_id column exists even if telemetry file is single-client only
    if "client_id" not in df.columns:
        df["client_id"] = args.client_id

    df["label"] = df[label_col].map(class_map).astype(int)
    df = safe_numeric_df(df, feature_cols + ["label"])

    # time split
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_val = df.iloc[split_idx:].copy().reset_index(drop=True)

    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["scale"], dtype=np.float32)

    def scale_X(dframe: pd.DataFrame):
        X = dframe[feature_cols].to_numpy(dtype=np.float32)
        X = (X - mean) / std
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    X_train = scale_X(df_train)
    y_train = df_train["label"].to_numpy(np.int64)

    X_val = scale_X(df_val)
    y_val = df_val["label"].to_numpy(np.int64)

    raw_train = df_train[feature_cols].to_numpy(dtype=np.float32)
    raw_all = df[feature_cols].to_numpy(dtype=np.float32)

    device = "cpu"

    def make_loader(X, y, batch_size=64, shuffle=True):
        tx = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        ty = torch.tensor(y, dtype=torch.long)
        ds = torch.utils.data.TensorDataset(tx, ty)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def evaluate(model):
        model.eval()
        loader = make_loader(X_val, y_val, batch_size=256, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        total, correct, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                pred = torch.argmax(logits, dim=1)
                total += yb.size(0)
                correct += (pred == yb).sum().item()
                loss_sum += loss.item() * yb.size(0)
        return {
            "val_acc": correct / max(total, 1),
            "val_loss": loss_sum / max(total, 1),
            "n_val": int(total),
        }

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "client_id": args.client_id,
            "rows": int(len(df)),
            "train_rows": int(len(df_train)),
            "val_rows": int(len(df_val)),
        }

    @app.get("/meta")
    def meta():
        return {
            "client_id": args.client_id,
            "n_train": int(len(df_train)),
            "n_val": int(len(df_val)),
            "class_counts": df[label_col].value_counts().to_dict(),
            "feature_cols": feature_cols,
        }

    @app.get("/telemetry/latest")
    def telemetry_latest():
        # use latest row in the client file
        row = df.iloc[-1]
        latest = {}
        for c in feature_cols:
            v = row[c]
            latest[c] = None if pd.isna(v) else float(v)
        return {
            "client_id": args.client_id,
            "latest": latest,
            "n_samples": int(len(df)),
        }

    @app.post("/train_local")
    def train_local(req: TrainRequest):
        set_seed(req.seed)

        model = TelemetryClassifier1DCNN(
            input_len=len(feature_cols),
            num_classes=len(class_map)
        ).to(device)
        model.load_state_dict(numpy_to_state_dict(req.global_weights, device=device))

        loader = make_loader(X_train, y_train, batch_size=req.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=req.lr)

        t0 = time.time()
        model.train()
        for _ in range(req.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
        train_time_s = time.time() - t0

        metrics = evaluate(model)
        return {
            "client_id": args.client_id,
            "mode": "local",
            "weights": state_dict_to_numpy(model.state_dict()),
            "n_train": int(len(df_train)),
            "train_time_s": float(train_time_s),
            "metrics": metrics,
        }

    @app.post("/export_train_shard")
    def export_train_shard(req: ExportShardRequest):
        """
        Real cloud-execution path:
        client exports its train shard to server,
        server trains centrally for offloaded clients.
        """
        if req.max_rows is None:
            X = X_train
            y = y_train
            raw = raw_train
        else:
            n = min(int(req.max_rows), len(X_train))
            X = X_train[:n]
            y = y_train[:n]
            raw = raw_train[:n]

        return {
            "client_id": args.client_id,
            "mode": "offload_export",
            "X": X.tolist(),
            "y": y.tolist(),
            "raw_X": raw.tolist(),
            "n_train": int(len(y)),
        }

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--win", type=int, default=10)  # kept for compatibility with your old command
    parser.add_argument("--class_map", required=True)
    parser.add_argument("--scaler_path", required=True)
    args = parser.parse_args()

    import uvicorn
    app = build_app(args)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
