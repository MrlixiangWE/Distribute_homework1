#!/usr/bin/env python3
"""
Visual task multi-system comparison.

Task: CIFAR-10 multi-class image classification (10 classes).

Systems:
1) single machine: sklearn SGD softmax classifier
2) distributed: dask distributed incremental SGD softmax (local simulation or remote cluster)
3) deep learning: PyTorch softmax linear classifier + ResNet18 baseline
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
import torch.nn.functional as F
from dask.distributed import Client, LocalCluster
from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
from dask_ml.wrappers import Incremental
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18


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
        "packages": ["dask", "distributed", "dask-ml", "scikit-learn", "numpy", "pandas", "psutil"],
        "cluster_mode": "local_simulation",  # local_simulation or remote_cluster
        "local_simulation": {
            "n_workers": 3,
            "threads_per_worker": 2,
            "processes": True,
            "memory_limit": "2GB",
            "dashboard_address": None,
        },
        "remote_cluster": {
            # Replace these with your real machines when you have resources.
            "scheduler_address": "tcp://192.168.10.10:8786",
            "worker_nodes": [
                {"name": "worker-1", "ip": "192.168.10.11", "port": 8786},
                {"name": "worker-2", "ip": "192.168.10.12", "port": 8786},
                {"name": "worker-3", "ip": "192.168.10.13", "port": 8786},
            ],
            "how_to_expand": "Append more worker_nodes entries with target machine IP/port.",
        },
    },
    "deep_learning_system": {
        "name": "PyTorch CPU",
        "python": "3.10+",
        "packages": ["torch", "torchvision", "numpy", "pandas", "scikit-learn", "psutil"],
        "device": "cpu",
        "softmax_linear": {
            "epochs": 14,
            "batch_size": 512,
            "learning_rate": 1e-2,
            "weight_decay": 1e-4,
        },
        "resnet18_transfer": {
            "epochs": 2,
            "batch_size": 96,
            "eval_batch_size": 128,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "use_pretrained": True,
            "freeze_backbone": True,
        },
    },
}


# ============================================================
# Experiment config (fixed in code, no CLI args)
# ============================================================
SEED = 42
NUM_CLASSES = 10
OUTPUT_DIR = Path("outputs_vision")
MODEL_DIR = OUTPUT_DIR / "models"
DATA_DIR = Path("data")

DATASET_CONFIG = {
    "name": "cifar10_visual_scene_object_classification",
    "image_size": (32, 32, 3),
    "train_subset": 20_000,
    "test_subset": 5_000,
    "resnet_train_subset": 6_000,
    "resnet_test_subset": 1_500,
}


@dataclass
class ExperimentResult:
    system: str
    algorithm_case: str
    accuracy: float
    macro_f1: float
    macro_auc_ovr: float
    top3_acc: float
    train_time_sec: float
    infer_time_sec: float
    peak_memory_mb: float
    model_size_mb: float
    parallelism: str
    notes: str


class ResourceMonitor:
    def __init__(self, interval_sec: float = 0.05) -> None:
        self.interval_sec = interval_sec
        self.process = psutil.Process(os.getpid())
        self._peak_rss = self.process.memory_info().rss
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

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


class DaskWorkerMemoryMonitor:
    """
    Approximate worker-memory peak by polling Dask scheduler worker metrics.
    """

    def __init__(self, client: Client, interval_sec: float = 0.2) -> None:
        self.client = client
        self.interval_sec = interval_sec
        self._peak_bytes = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                info = self.client.scheduler_info()
                total_worker_bytes = 0
                for worker in info.get("workers", {}).values():
                    total_worker_bytes += int(worker.get("metrics", {}).get("memory", 0))
                if total_worker_bytes > self._peak_bytes:
                    self._peak_bytes = total_worker_bytes
            except Exception:
                pass
            time.sleep(self.interval_sec)

    def start(self) -> None:
        self._peak_bytes = 0
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_peak_mb(self) -> float:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        return self._peak_bytes / (1024 ** 2)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def stratified_equal_subset(
    X: np.ndarray, y: np.ndarray, total_samples: int, seed: int, num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    if total_samples > len(y):
        raise ValueError("Requested subset larger than dataset.")
    rng = np.random.default_rng(seed)
    per_class = total_samples // num_classes
    remainder = total_samples % num_classes
    selected: list[np.ndarray] = []
    for cls in range(num_classes):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        take = per_class + (1 if cls < remainder else 0)
        selected.append(cls_indices[:take])
    all_idx = np.concatenate(selected)
    rng.shuffle(all_idx)
    return X[all_idx], y[all_idx]


def load_cifar_subsets() -> dict[str, np.ndarray]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_ds = CIFAR10(root=str(DATA_DIR), train=True, download=True)
    test_ds = CIFAR10(root=str(DATA_DIR), train=False, download=True)

    X_train_full = np.asarray(train_ds.data)
    y_train_full = np.asarray(train_ds.targets)
    X_test_full = np.asarray(test_ds.data)
    y_test_full = np.asarray(test_ds.targets)

    X_train, y_train = stratified_equal_subset(
        X_train_full, y_train_full, DATASET_CONFIG["train_subset"], seed=SEED, num_classes=NUM_CLASSES
    )
    X_test, y_test = stratified_equal_subset(
        X_test_full, y_test_full, DATASET_CONFIG["test_subset"], seed=SEED + 1, num_classes=NUM_CLASSES
    )
    X_train_resnet, y_train_resnet = stratified_equal_subset(
        X_train_full, y_train_full, DATASET_CONFIG["resnet_train_subset"], seed=SEED + 2, num_classes=NUM_CLASSES
    )
    X_test_resnet, y_test_resnet = stratified_equal_subset(
        X_test_full, y_test_full, DATASET_CONFIG["resnet_test_subset"], seed=SEED + 3, num_classes=NUM_CLASSES
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_resnet": X_train_resnet,
        "y_train_resnet": y_train_resnet,
        "X_test_resnet": X_test_resnet,
        "y_test_resnet": y_test_resnet,
    }


def extract_downsample_features(images: np.ndarray) -> np.ndarray:
    """
    Convert 32x32x3 image to low-res pooled visual feature vector.
    2x2 average pooling -> 16x16x3, then append image-level channel stats.
    """
    x = images.astype(np.float32) / 255.0
    pooled = x.reshape(x.shape[0], 16, 2, 16, 2, 3).mean(axis=(2, 4))
    global_mean = x.mean(axis=(1, 2))
    global_std = x.std(axis=(1, 2))
    features = np.concatenate([pooled.reshape(x.shape[0], -1), global_mean, global_std], axis=1)
    return features.astype(np.float32)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def evaluate_multiclass(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float, float]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    top3 = float(top_k_accuracy_score(y_true, y_prob, k=3, labels=labels))
    return acc, macro_f1, auc, top3


def safe_softmax_from_scores(scores: np.ndarray) -> np.ndarray:
    """
    Convert decision scores to numerically stable class probabilities.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim == 1:
        s = np.stack([-s, s], axis=1)
    s = s - np.max(s, axis=1, keepdims=True)
    np.exp(s, out=s)
    denom = s.sum(axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return s / denom


def build_dask_client() -> tuple[Client, LocalCluster | None, str]:
    dist_cfg = INSTALLATION_CONFIG["distributed_system"]
    mode = dist_cfg["cluster_mode"]
    if mode == "local_simulation":
        cfg = dist_cfg["local_simulation"]
        cluster = LocalCluster(
            n_workers=cfg["n_workers"],
            threads_per_worker=cfg["threads_per_worker"],
            processes=cfg["processes"],
            memory_limit=cfg["memory_limit"],
            dashboard_address=cfg["dashboard_address"],
            silence_logs=50,
        )
        client = Client(cluster)
        par = f"{cfg['n_workers']} workers x {cfg['threads_per_worker']} threads (processes={cfg['processes']})"
        return client, cluster, par

    scheduler = dist_cfg["remote_cluster"]["scheduler_address"]
    workers = len(dist_cfg["remote_cluster"]["worker_nodes"])
    client = Client(scheduler)
    return client, None, f"remote workers={workers}, scheduler={scheduler}"


def run_single_sklearn_softmax(
    X_train_feat: np.ndarray, y_train: np.ndarray, X_test_feat: np.ndarray, y_test: np.ndarray
) -> ExperimentResult:
    labels = np.arange(NUM_CLASSES)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_feat)
    X_test = scaler.transform(X_test_feat)

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=1200,
        tol=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=SEED,
    )

    monitor = ResourceMonitor()
    monitor.start()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    peak_mb = monitor.stop_peak_mb()

    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)
        if not np.isfinite(y_prob).all():
            raise ValueError("Non-finite probs from predict_proba.")
    except Exception:
        y_prob = safe_softmax_from_scores(model.decision_function(X_test))
    infer_time = time.perf_counter() - t1

    acc, macro_f1, auc, top3 = evaluate_multiclass(y_test, y_pred, y_prob, labels)

    out_path = MODEL_DIR / "single_sklearn_softmax.joblib"
    joblib.dump({"scaler": scaler, "model": model}, out_path)

    return ExperimentResult(
        system="single_machine_sklearn",
        algorithm_case="Softmax Regression (Image Features)",
        accuracy=acc,
        macro_f1=macro_f1,
        macro_auc_ovr=auc,
        top3_acc=top3,
        train_time_sec=train_time,
        infer_time_sec=infer_time,
        peak_memory_mb=peak_mb,
        model_size_mb=file_size_mb(out_path),
        parallelism=f"CPU cores available={os.cpu_count()}",
        notes="Single node SGD softmax on pooled visual features",
    )


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "compute"):
        value = value.compute()
    return np.asarray(value)


def run_distributed_dask_softmax(
    X_train_feat: np.ndarray, y_train: np.ndarray, X_test_feat: np.ndarray, y_test: np.ndarray
) -> ExperimentResult:
    labels = np.arange(NUM_CLASSES)
    client, cluster, parallelism = build_dask_client()
    host_monitor = ResourceMonitor()
    worker_monitor = DaskWorkerMemoryMonitor(client)

    try:
        chunks = 2_000
        X_train_da = da.from_array(X_train_feat, chunks=(chunks, X_train_feat.shape[1]))
        y_train_da = da.from_array(y_train, chunks=(chunks,))
        X_test_da = da.from_array(X_test_feat, chunks=(chunks, X_test_feat.shape[1]))

        scaler = DaskStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_da)
        X_test_scaled = scaler.transform(X_test_da)

        base = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-3,
            max_iter=1,
            tol=None,
            random_state=SEED,
            learning_rate="constant",
            eta0=1e-2,
        )
        model = Incremental(base, scoring="accuracy")

        host_monitor.start()
        worker_monitor.start()
        t0 = time.perf_counter()
        for _ in range(6):
            model.fit(X_train_scaled, y_train_da, classes=labels)
        train_time = time.perf_counter() - t0
        host_peak = host_monitor.stop_peak_mb()
        worker_peak = worker_monitor.stop_peak_mb()

        t1 = time.perf_counter()
        estimator = model.estimator_
        X_test_np = to_numpy(X_test_scaled)
        y_pred = estimator.predict(X_test_np)
        try:
            y_prob = estimator.predict_proba(X_test_np)
            if not np.isfinite(y_prob).all():
                raise ValueError("Non-finite probs from predict_proba.")
        except Exception:
            y_prob = safe_softmax_from_scores(estimator.decision_function(X_test_np))
        infer_time = time.perf_counter() - t1

        acc, macro_f1, auc, top3 = evaluate_multiclass(y_test, y_pred, y_prob, labels)

        out_path = MODEL_DIR / "distributed_dask_softmax.pkl"
        summary = {
            "coef": to_numpy(estimator.coef_).tolist(),
            "intercept": to_numpy(estimator.intercept_).tolist(),
            "classes": to_numpy(estimator.classes_).tolist(),
            "scaler_mean": to_numpy(scaler.mean_).tolist(),
            "scaler_scale": to_numpy(scaler.scale_).tolist(),
        }
        with out_path.open("wb") as f:
            pickle.dump(summary, f)

        return ExperimentResult(
            system="distributed_dask",
            algorithm_case="Softmax Regression (Image Features)",
            accuracy=acc,
            macro_f1=macro_f1,
            macro_auc_ovr=auc,
            top3_acc=top3,
            train_time_sec=train_time,
            infer_time_sec=infer_time,
            peak_memory_mb=host_peak + worker_peak,
            model_size_mb=file_size_mb(out_path),
            parallelism=parallelism,
            notes="Dask distributed incremental SGD softmax (local simulation by default)",
        )
    finally:
        client.close()
        if cluster is not None:
            cluster.close()


def run_pytorch_softmax(
    X_train_feat: np.ndarray, y_train: np.ndarray, X_test_feat: np.ndarray, y_test: np.ndarray
) -> ExperimentResult:
    labels = np.arange(NUM_CLASSES)
    cfg = INSTALLATION_CONFIG["deep_learning_system"]["softmax_linear"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_feat).astype(np.float32)
    X_test = scaler.transform(X_test_feat).astype(np.float32)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    X_test_t = torch.from_numpy(X_test)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=cfg["batch_size"], shuffle=True)

    model = nn.Linear(X_train.shape[1], NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    monitor = ResourceMonitor()
    monitor.start()
    t0 = time.perf_counter()
    model.train()
    for _ in range(cfg["epochs"]):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    train_time = time.perf_counter() - t0
    peak_mb = monitor.stop_peak_mb()

    t1 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    y_prob = probs.astype(np.float64)
    y_pred = np.argmax(y_prob, axis=1)
    infer_time = time.perf_counter() - t1

    acc, macro_f1, auc, top3 = evaluate_multiclass(y_test, y_pred, y_prob, labels)

    out_path = MODEL_DIR / "deep_pytorch_softmax.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        out_path,
    )

    return ExperimentResult(
        system="deep_learning_pytorch",
        algorithm_case="Softmax Regression (Image Features)",
        accuracy=acc,
        macro_f1=macro_f1,
        macro_auc_ovr=auc,
        top3_acc=top3,
        train_time_sec=train_time,
        infer_time_sec=infer_time,
        peak_memory_mb=peak_mb,
        model_size_mb=file_size_mb(out_path),
        parallelism=f"torch_threads={torch.get_num_threads()} on CPU",
        notes="PyTorch implementation of multiclass softmax regression",
    )


def preprocess_for_resnet18(x: torch.Tensor) -> torch.Tensor:
    """
    Resize CIFAR tensors to ImageNet input size and normalize.
    """
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def run_pytorch_resnet18(
    X_train_resnet: np.ndarray,
    y_train_resnet: np.ndarray,
    X_test_resnet: np.ndarray,
    y_test_resnet: np.ndarray,
) -> ExperimentResult:
    labels = np.arange(NUM_CLASSES)
    cfg = INSTALLATION_CONFIG["deep_learning_system"]["resnet18_transfer"]

    X_train = torch.from_numpy((X_train_resnet.astype(np.float32) / 255.0).transpose(0, 3, 1, 2))
    X_test = torch.from_numpy((X_test_resnet.astype(np.float32) / 255.0).transpose(0, 3, 1, 2))
    y_train = torch.from_numpy(y_train_resnet.astype(np.int64))

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, torch.from_numpy(y_test_resnet.astype(np.int64))), batch_size=cfg["eval_batch_size"], shuffle=False)

    weights = ResNet18_Weights.IMAGENET1K_V1 if cfg["use_pretrained"] else None
    try:
        model = resnet18(weights=weights)
    except Exception:
        model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    if cfg["freeze_backbone"]:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    monitor = ResourceMonitor()
    monitor.start()
    t0 = time.perf_counter()
    model.train()
    for _ in range(cfg["epochs"]):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(preprocess_for_resnet18(xb))
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    train_time = time.perf_counter() - t0
    peak_mb = monitor.stop_peak_mb()

    t1 = time.perf_counter()
    model.eval()
    all_probs: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in test_loader:
            logits = model(preprocess_for_resnet18(xb))
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)
            all_probs.append(probs)
            all_pred.append(np.argmax(probs, axis=1))
    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    infer_time = time.perf_counter() - t1

    acc, macro_f1, auc, top3 = evaluate_multiclass(y_test_resnet, y_pred, y_prob, labels)

    out_path = MODEL_DIR / "deep_pytorch_resnet18.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, out_path)

    return ExperimentResult(
        system="deep_learning_pytorch",
        algorithm_case="ResNet18 Transfer (Raw Images)",
        accuracy=acc,
        macro_f1=macro_f1,
        macro_auc_ovr=auc,
        top3_acc=top3,
        train_time_sec=train_time,
        infer_time_sec=infer_time,
        peak_memory_mb=peak_mb,
        model_size_mb=file_size_mb(out_path),
        parallelism=f"torch_threads={torch.get_num_threads()} on CPU",
        notes="Pretrained ResNet18 transfer learning baseline",
    )


def save_outputs(results: list[ExperimentResult], runtime_sec: float) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]
    df = pd.DataFrame(rows)
    metric_cols = [
        "accuracy",
        "macro_f1",
        "macro_auc_ovr",
        "top3_acc",
        "train_time_sec",
        "infer_time_sec",
        "peak_memory_mb",
        "model_size_mb",
    ]
    df[metric_cols] = df[metric_cols].round(4)

    csv_path = OUTPUT_DIR / "comparison_results.csv"
    md_path = OUTPUT_DIR / "comparison_results.md"
    json_path = OUTPUT_DIR / "comparison_results.json"
    install_path = OUTPUT_DIR / "installation_config.json"
    exp_cfg_path = OUTPUT_DIR / "experiment_config.json"

    df.to_csv(csv_path, index=False)
    try:
        markdown_text = df.to_markdown(index=False)
    except Exception:
        markdown_text = df.to_string(index=False)
    md_path.write_text(markdown_text, encoding="utf-8")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    with install_path.open("w", encoding="utf-8") as f:
        json.dump(INSTALLATION_CONFIG, f, ensure_ascii=False, indent=2)
    with exp_cfg_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"seed": SEED, "num_classes": NUM_CLASSES, "dataset_config": DATASET_CONFIG, "total_runtime_sec": runtime_sec},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== Installation Config ===")
    print(json.dumps(INSTALLATION_CONFIG, ensure_ascii=False, indent=2))
    print("\n=== Result Table ===")
    print(markdown_text)
    print(f"\nTotal runtime: {runtime_sec:.2f} sec")
    print(f"\nSaved files:\n- {csv_path}\n- {md_path}\n- {json_path}\n- {install_path}\n- {exp_cfg_path}")


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    data = load_cifar_subsets()

    X_train_feat = extract_downsample_features(data["X_train"])
    X_test_feat = extract_downsample_features(data["X_test"])

    results: list[ExperimentResult] = []
    results.append(run_single_sklearn_softmax(X_train_feat, data["y_train"], X_test_feat, data["y_test"]))
    results.append(run_distributed_dask_softmax(X_train_feat, data["y_train"], X_test_feat, data["y_test"]))
    results.append(run_pytorch_softmax(X_train_feat, data["y_train"], X_test_feat, data["y_test"]))
    results.append(
        run_pytorch_resnet18(
            data["X_train_resnet"],
            data["y_train_resnet"],
            data["X_test_resnet"],
            data["y_test_resnet"],
        )
    )

    total_runtime = time.perf_counter() - start
    save_outputs(results, runtime_sec=total_runtime)


if __name__ == "__main__":
    main()
