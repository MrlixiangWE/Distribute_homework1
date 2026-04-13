#!/usr/bin/env python3
"""
Multi-system ML comparison:
1) single machine sklearn
2) distributed Dask (local simulation or real cluster)
3) deep learning PyTorch

Task: binary classification for industrial equipment fault prediction
(synthetic but non-handwritten-digit style scenario).
"""

from __future__ import annotations

import json
import os
import pickle
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import dask.array as da
import joblib
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from dask.distributed import Client, LocalCluster
from dask_ml.linear_model import LogisticRegression as DaskLogisticRegression
from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# Installation config for each system type (fixed in code)
# ============================================================
INSTALLATION_CONFIG: dict[str, Any] = {
    "single_machine_system": {
        "name": "scikit-learn on one node",
        "python": "3.10+",
        "packages": ["numpy", "pandas", "scikit-learn", "joblib", "psutil"],
    },
    "distributed_system": {
        "name": "Dask Distributed",
        "python": "3.10+",
        "packages": ["dask", "distributed", "dask-ml", "numpy", "pandas", "psutil"],
        "cluster_mode": "local_simulation",  # local_simulation or remote_cluster
        "local_simulation": {
            "n_workers": 3,
            "threads_per_worker": 2,
            "processes": True,
            "memory_limit": "2GB",
            "dashboard_address": None,
        },
        "remote_cluster": {
            # Fill these when migrating to real multi-machine deployment.
            "scheduler_address": "tcp://192.168.10.10:8786",
            "worker_nodes": [
                {"name": "worker-1", "ip": "192.168.10.11", "port": 8786},
                {"name": "worker-2", "ip": "192.168.10.12", "port": 8786},
            ],
            "how_to_expand": "Add more worker_nodes entries with target machine IP and port.",
        },
    },
    "deep_learning_system": {
        "name": "PyTorch CPU",
        "python": "3.10+",
        "packages": ["torch", "numpy", "pandas", "scikit-learn", "psutil"],
        "device": "cpu",
        "model": {"hidden_dim": 64, "epochs": 12, "batch_size": 512, "learning_rate": 1e-3},
    },
}


# ============================================================
# Experiment config (fixed in code, no CLI args)
# ============================================================
RANDOM_SEED = 42
OUTPUT_DIR = Path("outputs")
MODEL_DIR = OUTPUT_DIR / "models"

DATASET_CONFIG = {
    "scenario_name": "smart_factory_fault_prediction",
    "n_samples": 80_000,
    "n_features": 48,
    "n_informative": 20,
    "n_redundant": 12,
    "class_sep": 1.1,
    "weights": [0.88, 0.12],
    "flip_y": 0.02,
    "test_size": 0.2,
}


@dataclass
class ExperimentResult:
    system: str
    algorithm: str
    accuracy: float
    f1: float
    roc_auc: float
    train_time_sec: float
    infer_time_sec: float
    peak_rss_mb: float
    model_size_mb: float
    parallelism: str
    notes: str


class ResourceMonitor:
    def __init__(self, interval_sec: float = 0.05) -> None:
        self.interval_sec = interval_sec
        self.process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss = self.process.memory_info().rss

    def _run(self) -> None:
        while not self._stop_event.is_set():
            rss = self.process.memory_info().rss
            if rss > self._peak_rss:
                self._peak_rss = rss
            time.sleep(self.interval_sec)

    def start(self) -> None:
        self._peak_rss = self.process.memory_info().rss
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_peak_mb(self) -> float:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        return self._peak_rss / (1024 ** 2)


class TorchMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(1)


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def metrics_from_outputs(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, float]:
    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred)),
        float(roc_auc_score(y_true, y_prob)),
    )


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=DATASET_CONFIG["n_samples"],
        n_features=DATASET_CONFIG["n_features"],
        n_informative=DATASET_CONFIG["n_informative"],
        n_redundant=DATASET_CONFIG["n_redundant"],
        n_classes=2,
        class_sep=DATASET_CONFIG["class_sep"],
        weights=DATASET_CONFIG["weights"],
        flip_y=DATASET_CONFIG["flip_y"],
        random_state=RANDOM_SEED,
    )
    return train_test_split(
        X, y, test_size=DATASET_CONFIG["test_size"], random_state=RANDOM_SEED, stratify=y
    )


def run_single_machine(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> ExperimentResult:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=300, n_jobs=-1, random_state=RANDOM_SEED)
    monitor = ResourceMonitor()

    monitor.start()
    t0 = time.perf_counter()
    model.fit(X_train_scaled, y_train)
    train_time = time.perf_counter() - t0
    peak_rss_mb = monitor.stop_peak_mb()

    t1 = time.perf_counter()
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    infer_time = time.perf_counter() - t1

    acc, f1, auc = metrics_from_outputs(y_test, y_pred, y_prob)

    out_path = MODEL_DIR / "single_sklearn_logreg.joblib"
    joblib.dump({"scaler": scaler, "model": model}, out_path)

    return ExperimentResult(
        system="single_machine_sklearn",
        algorithm="Logistic Regression",
        accuracy=acc,
        f1=f1,
        roc_auc=auc,
        train_time_sec=train_time,
        infer_time_sec=infer_time,
        peak_rss_mb=peak_rss_mb,
        model_size_mb=file_size_mb(out_path),
        parallelism=f"CPU cores available={os.cpu_count()}",
        notes="Baseline single-node training",
    )


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "compute"):
        value = value.compute()
    return np.asarray(value)


def build_dask_client() -> tuple[Client, LocalCluster | None, str]:
    dist_cfg = INSTALLATION_CONFIG["distributed_system"]
    mode = dist_cfg["cluster_mode"]
    if mode == "local_simulation":
        local_cfg = dist_cfg["local_simulation"]
        cluster = LocalCluster(
            n_workers=local_cfg["n_workers"],
            threads_per_worker=local_cfg["threads_per_worker"],
            processes=local_cfg["processes"],
            memory_limit=local_cfg["memory_limit"],
            dashboard_address=local_cfg["dashboard_address"],
            silence_logs=50,
        )
        client = Client(cluster)
        parallelism = (
            f"{local_cfg['n_workers']} workers x {local_cfg['threads_per_worker']} threads "
            f"(processes={local_cfg['processes']})"
        )
        return client, cluster, parallelism

    scheduler_address = dist_cfg["remote_cluster"]["scheduler_address"]
    client = Client(scheduler_address)
    worker_count = len(dist_cfg["remote_cluster"]["worker_nodes"])
    return client, None, f"remote workers={worker_count}, scheduler={scheduler_address}"


def run_distributed(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> ExperimentResult:
    client, cluster, parallelism = build_dask_client()
    monitor = ResourceMonitor()

    try:
        chunk_rows = 4000
        X_train_da = da.from_array(X_train, chunks=(chunk_rows, X_train.shape[1]))
        y_train_da = da.from_array(y_train, chunks=(chunk_rows,))
        X_test_da = da.from_array(X_test, chunks=(chunk_rows, X_test.shape[1]))

        scaler = DaskStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_da)
        X_test_scaled = scaler.transform(X_test_da)

        model = DaskLogisticRegression(max_iter=300, tol=1e-4)

        monitor.start()
        t0 = time.perf_counter()
        model.fit(X_train_scaled, y_train_da)
        train_time = time.perf_counter() - t0
        peak_rss_mb = monitor.stop_peak_mb()

        t1 = time.perf_counter()
        y_pred = to_numpy(model.predict(X_test_scaled))
        y_prob = to_numpy(model.predict_proba(X_test_scaled)[:, 1])
        infer_time = time.perf_counter() - t1

        acc, f1, auc = metrics_from_outputs(y_test, y_pred, y_prob)

        summary = {
            "coef": to_numpy(model.coef_).tolist(),
            "intercept": to_numpy(model.intercept_).tolist(),
            "scaler_mean": to_numpy(scaler.mean_).tolist(),
            "scaler_scale": to_numpy(scaler.scale_).tolist(),
        }
        out_path = MODEL_DIR / "distributed_dask_logreg.pkl"
        with out_path.open("wb") as f:
            pickle.dump(summary, f)

        notes = "Dask local simulation"
        if INSTALLATION_CONFIG["distributed_system"]["cluster_mode"] == "remote_cluster":
            notes = "Dask remote cluster"

        return ExperimentResult(
            system="distributed_dask",
            algorithm="Logistic Regression",
            accuracy=acc,
            f1=f1,
            roc_auc=auc,
            train_time_sec=train_time,
            infer_time_sec=infer_time,
            peak_rss_mb=peak_rss_mb,
            model_size_mb=file_size_mb(out_path),
            parallelism=parallelism,
            notes=notes,
        )
    finally:
        client.close()
        if cluster is not None:
            cluster.close()


def run_deep_learning(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> ExperimentResult:
    dl_cfg = INSTALLATION_CONFIG["deep_learning_system"]["model"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_scaled),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    train_loader = DataLoader(train_dataset, batch_size=dl_cfg["batch_size"], shuffle=True)

    model = TorchMLP(in_dim=X_train.shape[1], hidden_dim=dl_cfg["hidden_dim"])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=dl_cfg["learning_rate"])

    monitor = ResourceMonitor()
    monitor.start()
    t0 = time.perf_counter()
    model.train()
    for _ in range(dl_cfg["epochs"]):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
    train_time = time.perf_counter() - t0
    peak_rss_mb = monitor.stop_peak_mb()

    t1 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test_scaled))
        probs = torch.sigmoid(logits).cpu().numpy()
    y_prob = probs.astype(np.float64)
    y_pred = (y_prob >= 0.5).astype(np.int32)
    infer_time = time.perf_counter() - t1

    acc, f1, auc = metrics_from_outputs(y_test, y_pred, y_prob)

    out_path = MODEL_DIR / "deep_pytorch_mlp.pt"
    torch.save(
        {"state_dict": model.state_dict(), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_},
        out_path,
    )

    return ExperimentResult(
        system="deep_learning_pytorch",
        algorithm="Neural Logistic (MLP)",
        accuracy=acc,
        f1=f1,
        roc_auc=auc,
        train_time_sec=train_time,
        infer_time_sec=infer_time,
        peak_rss_mb=peak_rss_mb,
        model_size_mb=file_size_mb(out_path),
        parallelism=f"torch_threads={torch.get_num_threads()} on CPU",
        notes="2-layer MLP for same binary classification target",
    )


def save_outputs(results: list[ExperimentResult]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    rows = [asdict(r) for r in results]
    df = pd.DataFrame(rows)
    metric_cols = [
        "accuracy",
        "f1",
        "roc_auc",
        "train_time_sec",
        "infer_time_sec",
        "peak_rss_mb",
        "model_size_mb",
    ]
    df[metric_cols] = df[metric_cols].round(4)

    csv_path = OUTPUT_DIR / "comparison_results.csv"
    md_path = OUTPUT_DIR / "comparison_results.md"
    json_path = OUTPUT_DIR / "comparison_results.json"
    install_path = OUTPUT_DIR / "installation_config.json"

    df.to_csv(csv_path, index=False)
    try:
        markdown_text = df.to_markdown(index=False)
    except Exception:
        # Fallback when optional dependency `tabulate` is unavailable.
        markdown_text = df.to_string(index=False)
    md_path.write_text(markdown_text, encoding="utf-8")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    with install_path.open("w", encoding="utf-8") as f:
        json.dump(INSTALLATION_CONFIG, f, ensure_ascii=False, indent=2)

    print("\n=== Installation Config ===")
    print(json.dumps(INSTALLATION_CONFIG, ensure_ascii=False, indent=2))
    print("\n=== Result Table ===")
    print(markdown_text)
    print(f"\nSaved files:\n- {csv_path}\n- {md_path}\n- {json_path}\n- {install_path}")


def main() -> None:
    set_random_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = build_dataset()
    results: list[ExperimentResult] = []

    results.append(run_single_machine(X_train, X_test, y_train, y_test))
    results.append(run_distributed(X_train, X_test, y_train, y_test))
    results.append(run_deep_learning(X_train, X_test, y_train, y_test))

    save_outputs(results)


if __name__ == "__main__":
    main()
