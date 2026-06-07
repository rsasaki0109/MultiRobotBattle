#!/usr/bin/env python3
"""Distill CountAwarePolicy into the mini battle transformer.

Training uses numpy (offline only); the shipped JSON weights run in pure Python.

    python3 scripts/distill_battle_policy.py
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(_REPO, "mrn_coord"))

from mrn_coord.battle import BattleConfig, battle_step, make_armies, make_free_for_all  # noqa: E402
from mrn_coord.battle_policy.count_aware import CountAwarePolicy  # noqa: E402
from mrn_coord.battle_policy.transformer import (  # noqa: E402
    OUT_DIM,
    SELF_DIM,
    TOKEN_DIM,
    TransformerWeights,
    build_observation,
    decode_output,
    encode_features,
    forward,
)

_OUT = os.path.join(_REPO, "mrn_coord", "mrn_coord", "battle_policy", "weights", "default.json")


def _collect_samples(n_samples, seed):
    rng = random.Random(seed)
    teacher = CountAwarePolicy("auto")
    rows = []
    setups = [
        lambda: make_armies(rng.randint(6, 18), BattleConfig(), seed=rng.randint(0, 9999)),
        lambda: make_free_for_all(rng.randint(6, 12), BattleConfig(),
                                  seed=rng.randint(0, 9999), num_teams=3),
    ]
    while len(rows) < n_samples:
        bots = setups[rng.randint(0, 1)]()
        cfg = BattleConfig()
        for _ in range(rng.randint(1, 6)):
            battle_step(bots, cfg)
        live = [b for b in bots if b.alive]
        if len(live) < 2:
            continue
        for index in range(len(live)):
            obs = build_observation(live, index, perception=cfg.perception)
            if not obs.enemy_tokens:
                continue
            decision = teacher.decide(live, index, cfg)
            if decision is None:
                continue
            self_feat, allies, enemies, enemy_indices = encode_features(
                obs, perception=cfg.perception)
            target = np.zeros(OUT_DIM, dtype=np.float64)
            target[0] = math.log(max(decision.pursue_scale - 0.35, 1e-6) /
                                 max(1.6 - decision.pursue_scale, 1e-6))
            target[1] = math.log(max(decision.retreat_scale - 0.35, 1e-6) /
                                 max(1.6 - decision.retreat_scale, 1e-6))
            target[2] = math.log(max(decision.flock_scale - 0.55, 1e-6) /
                                 max(1.7 - decision.flock_scale, 1e-6))
            target[3] = 5.0 if decision.kite else -5.0
            for slot, idx in enumerate(enemy_indices):
                if idx == decision.target_index:
                    target[4 + slot] = 5.0
                elif idx >= 0:
                    target[4 + slot] = -2.0
            rows.append((np.array(self_feat), np.array(allies), np.array(enemies),
                         enemy_indices, target, live, index, cfg))
            if len(rows) >= n_samples:
                break
    return rows


class BattleTransformerNP:
    """Numpy twin of the pure-Python forward pass (for distillation)."""

    def __init__(self, d_model=24, seed=0):
        rng = np.random.default_rng(seed)
        s = lambda r, c: math.sqrt(6.0 / (r + c))
        self.d_model = d_model
        self.w_self = rng.uniform(-s(d_model, SELF_DIM), s(d_model, SELF_DIM),
                                  (d_model, SELF_DIM))
        self.b_self = np.zeros(d_model)
        self.w_tok = rng.uniform(-s(d_model, TOKEN_DIM), s(d_model, TOKEN_DIM),
                                 (d_model, TOKEN_DIM))
        self.b_tok = np.zeros(d_model)
        self.w_q = rng.uniform(-s(d_model, d_model), s(d_model, d_model),
                               (d_model, d_model))
        self.w_k = rng.uniform(-s(d_model, d_model), s(d_model, d_model),
                               (d_model, d_model))
        self.w_v = rng.uniform(-s(d_model, d_model), s(d_model, d_model),
                               (d_model, d_model))
        self.w_ff1 = rng.uniform(-s(d_model * 2, d_model), s(d_model * 2, d_model),
                                 (d_model * 2, d_model))
        self.b_ff1 = np.zeros(d_model * 2)
        self.w_ff2 = rng.uniform(-s(d_model, d_model * 2), s(d_model, d_model * 2),
                                 (d_model, d_model * 2))
        self.b_ff2 = np.zeros(d_model)
        self.w_out = rng.uniform(-s(OUT_DIM, d_model), s(OUT_DIM, d_model),
                                 (OUT_DIM, d_model))
        self.b_out = np.zeros(OUT_DIM)

    def forward(self, self_feat, allies, enemies):
        toks = np.vstack([allies, enemies])
        agent = self.w_self @ self_feat + self.b_self
        emb = toks @ self.w_tok.T + self.b_tok
        keys = emb @ self.w_k.T
        values = emb @ self.w_v.T
        query = self.w_q @ agent
        scores = keys @ query / math.sqrt(self.d_model)
        attn = np.exp(scores - scores.max())
        attn /= attn.sum()
        ctx = values.T @ attn
        hidden = agent + ctx
        ff1 = self.w_ff1 @ hidden + self.b_ff1
        ff1_relu = np.maximum(ff1, 0.0)
        ff2 = self.w_ff2 @ ff1_relu + self.b_ff2
        out = self.w_out @ ff2 + self.b_out
        cache = dict(self_feat=self_feat, toks=toks, agent=agent, emb=emb,
                     keys=keys, values=values, query=query, scores=scores,
                     attn=attn, ctx=ctx, hidden=hidden, ff1=ff1,
                     ff1_relu=ff1_relu, ff2=ff2)
        return out, cache

    def backward(self, cache, grad_out):
        d = self.d_model
        grads = {}

        grads["w_out"] = np.outer(grad_out, cache["ff2"])
        grads["b_out"] = grad_out

        grad_ff2 = self.w_out.T @ grad_out
        grads["w_ff2"] = np.outer(grad_ff2, cache["ff1_relu"])
        grads["b_ff2"] = grad_ff2

        grad_ff1_relu = self.w_ff2.T @ grad_ff2
        grad_ff1 = grad_ff1_relu * (cache["ff1"] > 0)
        grads["w_ff1"] = np.outer(grad_ff1, cache["hidden"])
        grads["b_ff1"] = grad_ff1

        grad_hidden = self.w_ff1.T @ grad_ff1
        grad_agent = grad_hidden.copy()
        grad_ctx = grad_hidden.copy()

        attn = cache["attn"]
        values = cache["values"]
        grad_attn = values @ grad_ctx
        grad_values = np.outer(attn, grad_ctx)

        scores = cache["scores"]
        grad_scores = attn * (grad_attn - np.dot(grad_attn, attn))
        grad_scores /= math.sqrt(d)

        query = cache["query"]
        keys = cache["keys"]
        grad_query = keys.T @ grad_scores
        grad_keys = np.outer(grad_scores, query)

        grad_agent += self.w_q.T @ grad_query
        grads["w_q"] = np.outer(grad_query, cache["agent"])

        emb = cache["emb"]
        grad_emb = grad_keys @ self.w_k + grad_values @ self.w_v
        grads["w_k"] = grad_keys.T @ emb
        grads["w_v"] = grad_values.T @ emb

        grads["w_tok"] = grad_emb.T @ cache["toks"]
        grads["b_tok"] = grad_emb.sum(axis=0)

        grads["w_self"] = np.outer(grad_agent, cache["self_feat"])
        grads["b_self"] = grad_agent

        return grads

    def apply_grads(self, grads, lr):
        for name, g in grads.items():
            setattr(self, name, getattr(self, name) - lr * g)

    def to_weights(self):
        def arr(x):
            return x.tolist()
        return TransformerWeights(
            d_model=self.d_model,
            w_self=arr(self.w_self), b_self=arr(self.b_self),
            w_tok=arr(self.w_tok), b_tok=arr(self.b_tok),
            w_q=arr(self.w_q), w_k=arr(self.w_k), w_v=arr(self.w_v),
            w_ff1=arr(self.w_ff1), b_ff1=arr(self.b_ff1),
            w_ff2=arr(self.w_ff2), b_ff2=arr(self.b_ff2),
            w_out=arr(self.w_out), b_out=arr(self.b_out),
        )


def train(rows, *, epochs=80, lr=0.01, seed=0):
    net = BattleTransformerNP(seed=seed)
    rng = random.Random(seed)
    n = len(rows)
    for epoch in range(epochs):
        rng.shuffle(rows)
        loss_sum = 0.0
        acc = None
        for self_feat, allies, enemies, _, target, *_rest in rows:
            pred, cache = net.forward(self_feat, allies, enemies)
            diff = pred - target
            loss_sum += float(diff @ diff)
            grad_out = 2.0 * diff / n
            g = net.backward(cache, grad_out)
            if acc is None:
                acc = g
            else:
                for k in g:
                    acc[k] += g[k]
        net.apply_grads(acc, lr)
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}: mse={loss_sum / n:.4f}")
    return net


def evaluate(rows, net):
    ok_target = ok_scale = n = 0
    teacher = CountAwarePolicy("auto")
    weights = net.to_weights()
    for self_feat, allies, enemies, enemy_indices, _target, live, index, cfg in rows:
        raw = forward(weights, self_feat.tolist(), allies.tolist(), enemies.tolist())
        decision = decode_output(raw, enemy_indices, live, index, cfg)
        ref = teacher.decide(live, index, cfg)
        if decision and ref:
            n += 1
            if decision.target_index == ref.target_index:
                ok_target += 1
            if (abs(decision.pursue_scale - ref.pursue_scale) < 0.3 and
                    abs(decision.flock_scale - ref.flock_scale) < 0.3):
                ok_scale += 1
    return {"n": n, "target_match": ok_target / max(n, 1),
            "scale_match": ok_scale / max(n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=_OUT)
    args = ap.parse_args()
    rows = _collect_samples(args.samples, args.seed)
    print(f"collected {len(rows)} teacher samples")
    net = train(rows, epochs=args.epochs, seed=args.seed)
    metrics = evaluate(rows, net)
    print(f"distill metrics: {metrics}")
    net.to_weights().save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
