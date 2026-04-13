#!/usr/bin/env python3
"""
ResNet18 comparison across three system styles:
1) single machine system
2) distributed system (local simulation with DDP; can switch to remote multi-node)
3) deep learning system (full fine-tune)

Task: CIFAR-10 visual classification (10 classes).
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, top_k_accuracy_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18


# ============================================================
# Installation configs (fixed in code, no CLI params)
# ============================================================
INSTALLATION_CONFIG: dict[str, Any] = {
    "single_machine_system": {
        "name": "PyTorch single process on one server",
        "python": "3.10+",
        "packages": ["torch", "torchvision", "numpy", "pandas", "scikit-learn", "psutil"],
        "model": {
            "pretrained": True,
            "freeze_backbone": True,
            "epochs": 2,
            "batch_size": 96,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "input_size": 112,
        },
    },
    "distributed_system": {
        "name": "PyTorch DDP (DistributedDataParallel)",
        "python": "3.10+",
        "packages": ["torch", "torchvision", "numpy", "pandas", "scikit-learn", "psutil"],
        "cluster_mode": "local_simulation",  # local_simulation or remote_cluster
        "local_simulation": {
            "world_size": 2,
            "master_addr": "127.0.0.1",
            "master_port": 29531,
            "epochs": 2,
            "batch_size_per_process": 64,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "pretrained": True,
            "freeze_backbone": True,
            "input_size": 112,
        },
        "remote_cluster": {
            # Fill these for real multi-machine deployment:
            # Example: 3 machines, each runs this script with different node_rank.
            "master_addr": "192.168.10.10",
            "master_port": 29531,
            "nnodes": 3,
            "node_rank": 0,
            "nproc_per_node": 1,
            "world_size": 3,
            "worker_nodes": [
                {"name": "node-0", "ip": "192.168.10.10", "port": 29531},
                {"name": "node-1", "ip": "192.168.10.11", "port": 29531},
                {"name": "node-2", "ip": "192.168.10.12", "port": 29531},
            ],
            "how_to_expand": "Append worker_nodes and update world_size/nnodes accordingly.",
        },
    },
    "deep_learning_system": {
        "name": "PyTorch advanced fine-tuning pipeline",
        "python": "3.10+",
        "packages": ["torch", "torchvision", "numpy", "pandas", "scikit-learn", "psutil"],
        "model": {
            "pretrained": True,
            "freeze_backbone": False,
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 5e-5,
            "weight_decay": 1e-4,
            "input_size": 112,
        },
    },
}


# ============================================================
# Experiment config
# ============================================================
SEED = 42
NUM_CLASSES = 10
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs_resnet")
MODEL_DIR = OUTPUT_DIR / "models"

DATASET_CONFIG = {
    "name": "cifar10_visual_classification_resnet",
    "train_subset": 6_000,
    "test_subset": 1_500,
}


@dataclass
class ExperimentResult:
    system: str
    algorithm: str
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


class ProcessTreeMemoryMonitor:
    """
    Monitor peak RSS of current process + all child processes.
    """

    def __init__(self, interval_sec: float = 0.05) -> None:
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss_bytes = 0

    def _sample_tree_rss(self) -> int:
        try:
            proc = psutil.Process(os.getpid())
            total = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except Exception:
                    pass
            return total
        except Exception:
            return 0

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._sample_tree_rss()
            if rss > self._peak_rss_bytes:
                self._peak_rss_bytes = rss
            time.sleep(self.interval_sec)

    def start(self) -> None:
        self._peak_rss_bytes = self._sample_tree_rss()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_peak_mb(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return self._peak_rss_bytes / (1024 ** 2)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def stratified_equal_subset(
    X: np.ndarray, y: np.ndarray, total_samples: int, seed: int, num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    if total_samples > len(y):
        raise ValueError("Subset larger than dataset.")
    rng = np.random.default_rng(seed)
    per_class = total_samples // num_classes
    remainder = total_samples % num_classes
    indices: list[np.ndarray] = []
    for cls in range(num_classes):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        take = per_class + (1 if cls < remainder else 0)
        indices.append(cls_idx[:take])
    selected = np.concatenate(indices)
    rng.shuffle(selected)
    return X[selected], y[selected]


def load_cifar_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return X_train, y_train, X_test, y_test


def preprocess_for_resnet(x: torch.Tensor, input_size: int) -> torch.Tensor:
    x = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def build_resnet18(pretrained: bool, freeze_backbone: bool) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        model = resnet18(weights=weights)
    except Exception:
        model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
    return model


def evaluate_model(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: np.ndarray,
    batch_size: int,
    input_size: int,
) -> tuple[float, float, float, float, float]:
    labels = np.arange(NUM_CLASSES)
    loader = DataLoader(
        TensorDataset(x_test, torch.from_numpy(y_test.astype(np.int64))), batch_size=batch_size, shuffle=False
    )
    model.eval()
    probs_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(preprocess_for_resnet(xb, input_size=input_size))
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)
            probs_all.append(probs)
            preds_all.append(np.argmax(probs, axis=1))
    infer_time = time.perf_counter() - t0
    y_prob = np.concatenate(probs_all, axis=0)
    y_pred = np.concatenate(preds_all, axis=0)
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"))
    top3 = float(top_k_accuracy_score(y_test, y_prob, labels=labels, k=3))
    return acc, macro_f1, auc, top3, infer_time


def run_single_machine_resnet(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> ExperimentResult:
    cfg = INSTALLATION_CONFIG["single_machine_system"]["model"]
    x_train = torch.from_numpy((X_train.astype(np.float32) / 255.0).transpose(0, 3, 1, 2))
    x_test = torch.from_numpy((X_test.astype(np.float32) / 255.0).transpose(0, 3, 1, 2))
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    loader = DataLoader(TensorDataset(x_train, y_train_t), batch_size=cfg["batch_size"], shuffle=True)

    model = build_resnet18(pretrained=cfg["pretrained"], freeze_backbone=cfg["freeze_backbone"])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    monitor = ProcessTreeMemoryMonitor()
    monitor.start()
    t0 = time.perf_counter()
    model.train()
    for _ in range(cfg["epochs"]):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(preprocess_for_resnet(xb, input_size=cfg["input_size"]))
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    train_time = time.perf_counter() - t0
    peak_mb = monitor.stop_peak_mb()

    acc, macro_f1, auc, top3, infer_time = evaluate_model(
        model, x_test, y_test, batch_size=128, input_size=cfg["input_size"]
    )
    out_path = MODEL_DIR / "single_machine_resnet18.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, out_path)
    return ExperimentResult(
        system="single_machine_system",
        algorithm="ResNet18",
        accuracy=acc,
        macro_f1=macro_f1,
        macro_auc_ovr=auc,
        top3_acc=top3,
        train_time_sec=train_time,
        infer_time_sec=infer_time,
        peak_memory_mb=peak_mb,
        model_size_mb=file_size_mb(out_path),
        parallelism=f"single process, cpu_threads={torch.get_num_threads()}",
        notes="Single-process transfer learning (fc-layer training)",
    )


def _find_free_port(default_port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", default_port)) != 0:
            return default_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _ddp_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict[str, Any],
    return_dict: dict[str, Any],
    model_path: str,
) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.set_num_threads(1)

    try:
        x_train = torch.from_numpy((X_train.astype(np.float32) / 255.0).transpose(0, 3, 1, 2))
        y_train_t = torch.from_numpy(y_train.astype(np.int64))
        train_ds = TensorDataset(x_train, y_train_t)
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(train_ds, batch_size=cfg["batch_size_per_process"], sampler=sampler)

        model = build_resnet18(pretrained=cfg["pretrained"], freeze_backbone=cfg["freeze_backbone"])
        ddp_model = DDP(model)
        params = [p for p in ddp_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        criterion = nn.CrossEntropyLoss()

        ddp_model.train()
        for epoch in range(cfg["epochs"]):
            sampler.set_epoch(epoch)
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = ddp_model(preprocess_for_resnet(xb, input_size=cfg["input_size"]))
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        dist.barrier()
        if rank == 0:
            x_test = torch.from_numpy((X_test.astype(np.float32) / 255.0).transpose(0, 3, 1, 2))
            acc, macro_f1, auc, top3, infer_time = evaluate_model(
                ddp_model.module,
                x_test=x_test,
                y_test=y_test,
                batch_size=128,
                input_size=cfg["input_size"],
            )
            torch.save({"state_dict": ddp_model.module.state_dict(), "cfg": cfg}, model_path)
            return_dict["accuracy"] = acc
            return_dict["macro_f1"] = macro_f1
            return_dict["macro_auc_ovr"] = auc
            return_dict["top3_acc"] = top3
            return_dict["infer_time_sec"] = infer_time
    finally:
        dist.destroy_process_group()


def run_distributed_resnet(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> ExperimentResult:
    dist_cfg = INSTALLATION_CONFIG["distributed_system"]
    if dist_cfg["cluster_mode"] == "local_simulation":
        cfg = dist_cfg["local_simulation"]
        world_size = int(cfg["world_size"])
        master_addr = cfg["master_addr"]
        master_port = _find_free_port(int(cfg["master_port"]))
        parallelism = f"DDP local simulation, world_size={world_size}"
    else:
        cfg = dist_cfg["remote_cluster"]
        world_size = int(cfg["world_size"])
        master_addr = cfg["master_addr"]
        master_port = int(cfg["master_port"])
        parallelism = f"DDP remote cluster, world_size={world_size}, master={master_addr}:{master_port}"

    manager = mp.Manager()
    shared_dict = manager.dict()
    out_path = MODEL_DIR / "distributed_resnet18_ddp.pt"

    monitor = ProcessTreeMemoryMonitor(interval_sec=0.1)
    monitor.start()
    t0 = time.perf_counter()
    mp.spawn(
        _ddp_worker,
        args=(
            world_size,
            master_addr,
            master_port,
            X_train,
            y_train,
            X_test,
            y_test,
            cfg,
            shared_dict,
            str(out_path),
        ),
        nprocs=world_size,
        join=True,
    )
    train_time = time.perf_counter() - t0
    peak_mb = monitor.stop_peak_mb()

    return ExperimentResult(
        system="distributed_system",
        algorithm="ResNet18",
        accuracy=float(shared_dict["accuracy"]),
        macro_f1=float(shared_dict["macro_f1"]),
        macro_auc_ovr=float(shared_dict["macro_auc_ovr"]),
        top3_acc=float(shared_dict["top3_acc"]),
        train_time_sec=train_time,
        infer_time_sec=float(shared_dict["infer_time_sec"]),
        peak_memory_mb=peak_mb,
        model_size_mb=file_size_mb(out_path),
        parallelism=parallelism,
        notes="DDP synchronized training (local multi-process simulation by default)",
    )


def run_deep_learning_resnet(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> ExperimentResult:
    cfg = INSTALLATION_CONFIG["deep_learning_system"]["model"]
    x_train = torch.from_numpy((X_train.astype(np.float32) / 255.0).transpose(0, 3, 1, 2))
    x_test = torch.from_numpy((X_test.astype(np.float32) / 255.0).transpose(0, 3, 1, 2))
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    train_loader = DataLoader(TensorDataset(x_train, y_train_t), batch_size=cfg["batch_size"], shuffle=True)

    model = build_resnet18(pretrained=cfg["pretrained"], freeze_backbone=cfg["freeze_backbone"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    monitor = ProcessTreeMemoryMonitor()
    monitor.start()
    t0 = time.perf_counter()
    model.train()
    for _ in range(cfg["epochs"]):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(preprocess_for_resnet(xb, input_size=cfg["input_size"]))
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    train_time = time.perf_counter() - t0
    peak_mb = monitor.stop_peak_mb()

    acc, macro_f1, auc, top3, infer_time = evaluate_model(
        model, x_test, y_test, batch_size=128, input_size=cfg["input_size"]
    )
    out_path = MODEL_DIR / "deep_learning_resnet18.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, out_path)
    return ExperimentResult(
        system="deep_learning_system",
        algorithm="ResNet18",
        accuracy=acc,
        macro_f1=macro_f1,
        macro_auc_ovr=auc,
        top3_acc=top3,
        train_time_sec=train_time,
        infer_time_sec=infer_time,
        peak_memory_mb=peak_mb,
        model_size_mb=file_size_mb(out_path),
        parallelism=f"single process full fine-tune, cpu_threads={torch.get_num_threads()}",
        notes="Full-parameter fine-tuning pipeline",
    )


def save_outputs(results: list[ExperimentResult], runtime_sec: float) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in results])
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
    exp_path = OUTPUT_DIR / "experiment_config.json"

    df.to_csv(csv_path, index=False)
    try:
        markdown_text = df.to_markdown(index=False)
    except Exception:
        markdown_text = df.to_string(index=False)
    md_path.write_text(markdown_text, encoding="utf-8")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    install_path.write_text(json.dumps(INSTALLATION_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
    exp_path.write_text(
        json.dumps(
            {
                "seed": SEED,
                "num_classes": NUM_CLASSES,
                "dataset_config": DATASET_CONFIG,
                "total_runtime_sec": runtime_sec,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Installation Config ===")
    print(json.dumps(INSTALLATION_CONFIG, ensure_ascii=False, indent=2))
    print("\n=== Result Table ===")
    print(markdown_text)
    print(f"\nTotal runtime: {runtime_sec:.2f} sec")
    print(
        f"\nSaved files:\n- {csv_path}\n- {md_path}\n- {json_path}\n- {install_path}\n- {exp_path}"
    )


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_cifar_data()
    t0 = time.perf_counter()
    results: list[ExperimentResult] = []
    results.append(run_single_machine_resnet(X_train, y_train, X_test, y_test))
    results.append(run_distributed_resnet(X_train, y_train, X_test, y_test))
    results.append(run_deep_learning_resnet(X_train, y_train, X_test, y_test))
    total_runtime = time.perf_counter() - t0
    save_outputs(results, runtime_sec=total_runtime)


if __name__ == "__main__":
    main()
