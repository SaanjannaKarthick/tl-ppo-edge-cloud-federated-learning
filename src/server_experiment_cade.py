#!/usr/bin/env python3
import argparse
import copy
import json
import os
import random
import time
from collections import deque
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# =========================================================
# Models
# =========================================================
class TelemetryClassifier1DCNN(nn.Module):
    """
    Global FL model: telemetry classification
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


class TelemetryStateEncoder1DCNN(nn.Module):
    """
    Server-side telemetry encoder for RL state abstraction
    """
    def __init__(self, input_len=15, embedding_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.proj(z)


def state_dict_to_numpy(sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    return {k: v.detach().cpu().numpy().tolist() for k, v in sd.items()}


def numpy_to_state_dict(d: Dict[str, Any], device="cpu") -> Dict[str, torch.Tensor]:
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in d.items()}


# =========================================================
# Data helpers
# =========================================================
def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def scale_features(df: pd.DataFrame, feature_cols: List[str], scaler: Dict[str, Any]) -> np.ndarray:
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["scale"], dtype=np.float32)
    X = (X - mean) / std
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def make_loader(X, y, batch_size=128, shuffle=False):
    tx = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    ty = torch.tensor(y, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(tx, ty)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate_model(model, X, y, device="cpu"):
    model.eval()
    loader = make_loader(X, y, batch_size=256, shuffle=False)
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
        "acc": correct / max(total, 1),
        "loss": loss_sum / max(total, 1),
        "n": total,
    }


def build_global_validation(df: pd.DataFrame, label_col: str, feature_cols: List[str], class_map: Dict[str, int], scaler):
    parts_x, parts_y = [], []
    client_ids = sorted(df["client_id"].dropna().unique().tolist())
    for cid in client_ids:
        dfi = df[df["client_id"] == cid].copy()
        dfi["label"] = dfi[label_col].map(class_map).astype(int)
        split_idx = int(len(dfi) * 0.8)
        df_val = dfi.iloc[split_idx:].copy()
        Xi = scale_features(df_val, feature_cols, scaler)
        yi = df_val["label"].to_numpy(np.int64)
        parts_x.append(Xi)
        parts_y.append(yi)
    Xg = np.concatenate(parts_x, axis=0)
    yg = np.concatenate(parts_y, axis=0)
    return Xg, yg


# =========================================================
# FedAvg
# =========================================================
def fedavg(weights_list: List[Dict[str, Any]], num_examples: List[int]) -> Dict[str, Any]:
    total = float(sum(num_examples))
    out = {}
    for k in weights_list[0].keys():
        acc = None
        for w, n in zip(weights_list, num_examples):
            arr = np.array(w[k], dtype=np.float32)
            weighted = (n / total) * arr
            acc = weighted if acc is None else acc + weighted
        out[k] = acc.tolist()
    return out


# =========================================================
# Remote client
# =========================================================
class RemoteClient:
    def __init__(self, idx: int, base_url: str, timeout=600):
        self.idx = idx
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self):
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def telemetry_latest(self):
        r = requests.get(f"{self.base_url}/telemetry/latest", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def train_local(self, payload):
        r = requests.post(f"{self.base_url}/train_local", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def export_train_shard(self, max_rows=None):
        payload = {"max_rows": max_rows}
        r = requests.post(f"{self.base_url}/export_train_shard", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()


# =========================================================
# RL State Builder
# =========================================================
class ServerTelemetryStateBuilder:
    def __init__(self, feature_cols: List[str], scaler: Dict[str, Any], embedding_dim: int = 16):
        self.feature_cols = feature_cols
        self.scaler = scaler
        self.encoder = TelemetryStateEncoder1DCNN(input_len=len(feature_cols), embedding_dim=embedding_dim)
        self.embedding_dim = embedding_dim

    def preprocess_payload(self, payload: Dict[str, Any]):
        row = payload["latest"]
        raw = np.array([0.0 if row.get(c) is None else float(row.get(c, 0.0)) for c in self.feature_cols], dtype=np.float32)
        mean = np.array(self.scaler["mean"], dtype=np.float32)
        std = np.array(self.scaler["scale"], dtype=np.float32)
        scaled = (raw - mean) / std
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return scaled, raw

    def build_state(self, telemetry_payloads, prev_acc, prev_latency, prev_offload_ratio):
        scaled_rows, raw_rows = [], []
        for p in telemetry_payloads:
            s, r = self.preprocess_payload(p)
            scaled_rows.append(s)
            raw_rows.append(r)
        X = np.stack(scaled_rows, axis=0)
        raw_rows = np.stack(raw_rows, axis=0)

        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            z = self.encoder(tx).cpu().numpy()

        flat = z.reshape(-1)
        extra = np.array([prev_acc, prev_latency, prev_offload_ratio], dtype=np.float32)
        state = np.concatenate([flat, extra], axis=0)
        return state.astype(np.float32), raw_rows


# =========================================================
# Strategies
# =========================================================
def strategy_fedavg(num_clients):
    return np.zeros(num_clients, dtype=np.int64)


def strategy_random(num_clients, p=0.5):
    return np.random.binomial(1, p, size=num_clients).astype(np.int64)


def strategy_greedy(raw_rows):
    out = []
    for row in raw_rows:
        cpu = float(row[0])
        load = float(row[3])
        ping = float(row[12])
        jitter = float(row[13])
        loss = float(row[14])

        if ping >= 900 or loss >= 90:
            out.append(0)
            continue

        offload = 1 if ((cpu > 70.0 or load > 1.5) and (ping < 200.0 and jitter < 20.0 and loss < 10.0)) else 0
        out.append(offload)
    return np.array(out, dtype=np.int64)


# =========================================================
# Replay + DQN
# =========================================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(args)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buf), size=min(batch_size, len(self.buf)), replace=False)
        return [self.buf[i] for i in idx]

    def __len__(self):
        return len(self.buf)


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class JointDQNAgent:
    def __init__(self, state_dim, num_clients, lr=1e-3, gamma_rl=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.995, target_update=10, ddqn=False):
        self.num_clients = num_clients
        self.action_dim = 2 ** num_clients
        self.gamma_rl = gamma_rl
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.ddqn = ddqn

        self.q = QNet(state_dim, self.action_dim)
        self.qt = QNet(state_dim, self.action_dim)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=10000)
        self.step_count = 0

    def action_index_to_vec(self, idx):
        bits = [(idx >> i) & 1 for i in range(self.num_clients)]
        return np.array(bits, dtype=np.int64)

    def action_vec_to_index(self, vec):
        idx = 0
        for i, bit in enumerate(vec.astype(int).tolist()):
            idx |= (int(bit) << i)
        return idx

    def act(self, state):
        if np.random.rand() < self.eps:
            idx = np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                idx = int(torch.argmax(self.q(s), dim=1).item())
        return self.action_index_to_vec(idx)

    def remember(self, s, a_vec, r, s2, done):
        self.buffer.push(s, self.action_vec_to_index(a_vec), r, s2, done)

    def train_step(self, batch_size=64):
        if len(self.buffer) < 32:
            return None

        batch = self.buffer.sample(batch_size)
        s = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        a = torch.tensor([b[1] for b in batch], dtype=torch.long).unsqueeze(1)
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1)
        s2 = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
        d = torch.tensor([b[4] for b in batch], dtype=torch.float32).unsqueeze(1)

        qsa = self.q(s).gather(1, a)

        with torch.no_grad():
            if self.ddqn:
                next_a = torch.argmax(self.q(s2), dim=1, keepdim=True)
                next_q = self.qt(s2).gather(1, next_a)
            else:
                next_q = torch.max(self.qt(s2), dim=1, keepdim=True)[0]
            target = r + self.gamma_rl * (1.0 - d) * next_q

        loss = nn.functional.mse_loss(qsa, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.qt.load_state_dict(self.q.state_dict())

        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        return float(loss.item())


# =========================================================
# PPO
# =========================================================
class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, num_clients):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.actor = nn.Linear(256, num_clients)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)


class TLPPOAgent:
    def __init__(self, state_dim, num_clients, lr=3e-4, gamma_rl=0.99, clip_eps=0.2, update_epochs=4):
        self.gamma_rl = gamma_rl
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.net = PPOActorCritic(state_dim, num_clients)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.rollout = []

    def act(self, state):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, value = self.net(s)
        probs = torch.sigmoid(logits)
        dist = torch.distributions.Bernoulli(probs=probs)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=1)
        return action.squeeze(0).numpy().astype(np.int64), float(logprob.item()), float(value.item())

    def remember(self, state, action, logprob, value, reward, done):
        self.rollout.append((state, action, logprob, value, reward, done))

    def train_step(self):
        if len(self.rollout) < 16:
            return None

        states = torch.tensor(np.stack([x[0] for x in self.rollout]), dtype=torch.float32)
        actions = torch.tensor(np.stack([x[1] for x in self.rollout]), dtype=torch.float32)
        old_logprobs = torch.tensor([x[2] for x in self.rollout], dtype=torch.float32)
        values = torch.tensor([x[3] for x in self.rollout], dtype=torch.float32)
        rewards = [x[4] for x in self.rollout]
        dones = [x[5] for x in self.rollout]

        returns = []
        g = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                g = 0.0
            g = r + self.gamma_rl * g
            returns.append(g)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)

        adv = returns - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.update_epochs):
            logits, v = self.net(states)
            probs = torch.sigmoid(logits)
            dist = torch.distributions.Bernoulli(probs=probs)
            logprobs = dist.log_prob(actions).sum(dim=1)
            ratio = torch.exp(logprobs - old_logprobs)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(v.squeeze(1), returns)
            entropy = dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        self.rollout.clear()
        return True


# =========================================================
# Cloud-side training
# =========================================================
def cloud_train_for_client(global_weights, X, y, lr, epochs, batch_size, seed, input_len, num_classes):
    set_seed(seed)
    device = "cpu"

    model = TelemetryClassifier1DCNN(input_len=input_len, num_classes=num_classes).to(device)
    model.load_state_dict(numpy_to_state_dict(global_weights, device=device))

    loader = make_loader(np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    train_time_s = time.time() - t0

    return state_dict_to_numpy(model.state_dict()), train_time_s


# =========================================================
# Reward
# =========================================================
def compute_reward(acc, latency_s, offload_ratio, alpha, beta, gamma):
    latency_norm = min(latency_s / 5.0, 2.0)
    return float(alpha * acc - beta * latency_norm - gamma * offload_ratio)


# =========================================================
# One strategy run
# =========================================================
def run_one_strategy(
    strategy_name,
    seed,
    args,
    clients,
    feature_cols,
    class_map,
    scaler,
    Xg,
    yg
):
    set_seed(seed)
    outdir = args.outdir
    ensure_dir(outdir)

    device = "cpu"
    global_model = TelemetryClassifier1DCNN(input_len=len(feature_cols), num_classes=len(class_map)).to(device)
    global_weights = state_dict_to_numpy(global_model.state_dict())

    state_builder = ServerTelemetryStateBuilder(feature_cols, scaler, embedding_dim=16)
    num_clients = len(clients)
    state_dim = num_clients * 16 + 3

    rl_agent = None
    if strategy_name == "dqn":
        rl_agent = JointDQNAgent(state_dim=state_dim, num_clients=num_clients, ddqn=False)
    elif strategy_name == "ddqn":
        rl_agent = JointDQNAgent(state_dim=state_dim, num_clients=num_clients, ddqn=True)
    elif strategy_name == "tl_ppo":
        rl_agent = TLPPOAgent(state_dim=state_dim, num_clients=num_clients)

    history = []
    prev_acc, prev_latency, prev_offload_ratio = 0.0, 0.0, 0.0

    for rnd in range(1, args.rounds + 1):
        telemetry_payloads = [c.telemetry_latest() for c in clients]
        state, raw_rows = state_builder.build_state(telemetry_payloads, prev_acc, prev_latency, prev_offload_ratio)

        if strategy_name == "fedavg":
            action = strategy_fedavg(num_clients)
        elif strategy_name == "random":
            action = strategy_random(num_clients)
        elif strategy_name in ("greedy", "heuristic"):
            action = strategy_greedy(raw_rows)
        elif strategy_name in ("dqn", "ddqn"):
            action = rl_agent.act(state)
        elif strategy_name == "tl_ppo":
            action, logprob, value = rl_agent.act(state)
        else:
            raise ValueError(f"Unsupported IC2E strategy: {strategy_name}")

        round_t0 = time.time()
        local_weights = []
        local_sizes = []
        client_train_times = []
        client_val_accs = []
        exec_modes = []

        for i, c in enumerate(clients):
            per_seed = seed * 1000 + rnd * 100 + i

            if int(action[i]) == 0:
                # edge local training
                res = c.train_local({
                    "global_weights": global_weights,
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "seed": per_seed,
                })
                local_weights.append(res["weights"])
                local_sizes.append(int(res["n_train"]))
                client_train_times.append(float(res["train_time_s"]))
                client_val_accs.append(float(res["metrics"]["val_acc"]))
                exec_modes.append("edge_local")

            else:
                # real cloud execution path
                shard = c.export_train_shard(max_rows=None)
                w_new, cloud_time = cloud_train_for_client(
                    global_weights=global_weights,
                    X=shard["X"],
                    y=shard["y"],
                    lr=args.lr,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    seed=per_seed,
                    input_len=len(feature_cols),
                    num_classes=len(class_map),
                )
                local_weights.append(w_new)
                local_sizes.append(int(shard["n_train"]))
                client_train_times.append(float(cloud_time))
                client_val_accs.append(np.nan)
                exec_modes.append("cloud_offload")

        round_latency = time.time() - round_t0

        global_weights = fedavg(local_weights, local_sizes)
        global_model.load_state_dict(numpy_to_state_dict(global_weights, device=device))

        evalm = evaluate_model(global_model, Xg, yg, device=device)
        acc = float(evalm["acc"])
        offload_ratio = float(np.mean(action))
        reward = compute_reward(
            acc=acc,
            latency_s=round_latency,
            offload_ratio=offload_ratio,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
        )

        next_payloads = [c.telemetry_latest() for c in clients]
        next_state, _ = state_builder.build_state(next_payloads, acc, round_latency, offload_ratio)
        done = (rnd == args.rounds)

        rl_loss = None
        if strategy_name in ("dqn", "ddqn"):
            rl_agent.remember(state, action, reward, next_state, done)
            rl_loss = rl_agent.train_step(batch_size=64)
        elif strategy_name == "tl_ppo":
            rl_agent.remember(state, action, logprob, value, reward, done)
            rl_loss = rl_agent.train_step()

        prev_acc, prev_latency, prev_offload_ratio = acc, round_latency, offload_ratio

        history.append({
            "round": rnd,
            "strategy": strategy_name,
            "seed": seed,
            "epochs": args.epochs,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "delta_off": args.delta_off,
            "delta_sel": args.delta_sel,
            "lambda_stab": args.lambda_stab,
            "global_acc": acc,
            "reward": reward,
            "latency_s": round_latency,
            "offload_ratio": offload_ratio,
            "action": action.tolist(),
            "exec_modes": exec_modes,
            "client_train_time_mean": float(np.mean(client_train_times)),
            "client_val_acc_mean": float(np.nanmean(client_val_accs)) if np.isfinite(np.nanmean(client_val_accs)) else None,
            "rl_loss": rl_loss,
        })

        print(
            f"[{strategy_name}] seed={seed} round={rnd}/{args.rounds} "
            f"acc={acc:.4f} reward={reward:.4f} latency={round_latency:.4f}s "
            f"offload={offload_ratio:.2f}"
        )

    out = {
        "strategy": strategy_name,
        "seed": seed,
        "rounds": args.rounds,
        "epochs": args.epochs,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "delta_off": args.delta_off,
        "delta_sel": args.delta_sel,
        "lambda_stab": args.lambda_stab,
        "history": history,
    }

    out_path = os.path.join(outdir, f"{strategy_name}_seed{seed}_r{args.rounds}.json")
    save_json(out, out_path)
    print(f"[OK] wrote {out_path}")


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label_col", default="scenario")
    ap.add_argument("--clients", required=True, help="Comma-separated list of client base URLs")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--strategies", default="fedavg,random,greedy,dqn,ddqn,tl_ppo")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.3)

    # accepted for compatibility with your older command
    ap.add_argument("--delta_off", type=float, default=0.01)
    ap.add_argument("--delta_sel", type=float, default=0.03)
    ap.add_argument("--lambda_stab", type=float, default=0.10)

    args = ap.parse_args()

    ensure_dir(args.outdir)

    feature_cols = load_json("/home/ubuntu/afrl_ic2e/feature_cols.json")
    class_map = load_json("/home/ubuntu/afrl_ic2e/class_map.json")
    scaler = load_json("/home/ubuntu/afrl_ic2e/scaler.json")

    df = pd.read_csv(args.csv)
    Xg, yg = build_global_validation(df, args.label_col, feature_cols, class_map, scaler)

    client_urls = [x.strip() for x in args.clients.split(",") if x.strip()]
    clients = [RemoteClient(i, u) for i, u in enumerate(client_urls)]

    for c in clients:
        print("[HEALTH]", c.health())

    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    # IC2E paper does not include TR-DP-AFRL
    if "tr_dp_afrl" in strategies:
        raise SystemExit(
            "[ERROR] tr_dp_afrl is not part of the current IC2E paper draft. "
            "Remove it from --strategies unless you revise the paper."
        )

    for seed in seeds:
        for strat in strategies:
            run_one_strategy(
                strategy_name=strat,
                seed=seed,
                args=args,
                clients=clients,
                feature_cols=feature_cols,
                class_map=class_map,
                scaler=scaler,
                Xg=Xg,
                yg=yg,
            )


if __name__ == "__main__":
    main()
