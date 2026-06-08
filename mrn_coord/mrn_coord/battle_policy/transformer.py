"""Mini cross-attention transformer for decentralized battle tactics.

A TeamHOI-style stack: the observing agent attends to teammate and enemy tokens,
then predicts steering scales, a kite flag, and a target distribution over visible
enemies. Inference is pure Python (stdlib only); weights ship as JSON and are
re-fit with ``scripts/distill_battle_policy.py`` (numpy used offline only).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

from .count_aware import TacticalDecision
from .tokens import build_observation

_KIND_INDEX = {"": 0, "scout": 1, "soldier": 2, "tank": 3, "sniper": 4}
_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "default.json")

SELF_DIM = 8
TOKEN_DIM = 6
MAX_ALLY = 8
MAX_ENEMY = 8
OUT_DIM = 12   # pursue, retreat, flock, kite + 8 enemy logits


def _sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _softmax(xs):
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex)
    return [e / s for e in ex]


def _matvec(mat, vec):
    return [sum(w * v for w, v in zip(row, vec)) for row in mat]


def _vec_add(a, b):
    return [x + y for x, y in zip(a, b)]


def _relu(vec):
    return [max(0.0, x) for x in vec]


def _kind_index(kind):
    return _KIND_INDEX.get(kind or "", 0) / 4.0


def encode_features(obs, *, perception=6.0):
    """Fixed-size self vector + padded ally/enemy token rows."""
    self_feat = [
        obs.hp_frac,
        math.sin(obs.heading),
        math.cos(obs.heading),
        obs.n_allies / 32.0,
        obs.n_enemies / 32.0,
        obs.n_local_allies / 8.0,
        obs.n_local_enemies / 8.0,
        _kind_index(obs.kind),
    ]
    allies = []
    for tok in obs.teammate_tokens[:MAX_ALLY]:
        allies.append([
            tok.rel_x / perception,
            tok.rel_y / perception,
            tok.rel_vx / 3.0,
            tok.rel_vy / 3.0,
            tok.hp_frac,
            tok.dist / perception,
        ])
    while len(allies) < MAX_ALLY:
        allies.append([0.0] * TOKEN_DIM)

    enemies = []
    enemy_indices = []
    for tok in obs.enemy_tokens[:MAX_ENEMY]:
        enemies.append([
            tok.rel_x / perception,
            tok.rel_y / perception,
            tok.rel_vx / 3.0,
            tok.rel_vy / 3.0,
            tok.hp_frac,
            tok.dist / perception,
        ])
        enemy_indices.append(tok.live_index)
    while len(enemies) < MAX_ENEMY:
        enemies.append([0.0] * TOKEN_DIM)
        enemy_indices.append(-1)

    return self_feat, allies, enemies, enemy_indices


@dataclass
class TransformerWeights:
    d_model: int
    w_self: list
    b_self: list
    w_tok: list
    b_tok: list
    w_q: list
    w_k: list
    w_v: list
    w_ff1: list
    b_ff1: list
    w_ff2: list
    b_ff2: list
    w_out: list
    b_out: list

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(**data)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.__dict__, fh)


def forward(weights: TransformerWeights, self_feat, allies, enemies):
    """Cross-attention over ally+enemy tokens; return raw output logits."""
    d = weights.d_model
    agent = _vec_add(_matvec(weights.w_self, self_feat), weights.b_self)

    keys, values = [], []
    for row in allies + enemies:
        emb = _vec_add(_matvec(weights.w_tok, row), weights.b_tok)
        keys.append(_matvec(weights.w_k, emb))
        values.append(_matvec(weights.w_v, emb))

    query = _matvec(weights.w_q, agent)
    scale = math.sqrt(d)
    scores = [sum(q * k for q, k in zip(query, key)) / scale for key in keys]
    attn = _softmax(scores)
    ctx = [0.0] * d
    for a, val in zip(attn, values):
        ctx = [c + a * v for c, v in zip(ctx, val)]
    hidden = _vec_add(agent, ctx)
    ff = _vec_add(_matvec(weights.w_ff1, hidden), weights.b_ff1)
    ff = _relu(ff)
    ff = _vec_add(_matvec(weights.w_ff2, ff), weights.b_ff2)
    return _vec_add(_matvec(weights.w_out, ff), weights.b_out)


def decode_output(raw, enemy_indices, live, index, cfg):
    """Map network logits to a :class:`TacticalDecision`."""
    pursue = 0.35 + 1.25 * _sigmoid(raw[0])
    retreat = 0.35 + 1.25 * _sigmoid(raw[1])
    flock = 0.55 + 1.15 * _sigmoid(raw[2])
    kite = _sigmoid(raw[3]) > 0.55

    best_slot, best_logit = None, float("-inf")
    for slot, logit in enumerate(raw[4:4 + MAX_ENEMY]):
        if enemy_indices[slot] < 0:
            continue
        if logit > best_logit:
            best_logit, best_slot = logit, slot
    if best_slot is None:
        return None
    target = enemy_indices[best_slot]
    from ..battle_teams import teams_are_enemies
    if (target < 0 or target >= len(live)
            or not teams_are_enemies(cfg.alliances, live[index].team,
                                     live[target].team)):
        return None
    return TacticalDecision(target_index=target,
                            pursue_scale=pursue,
                            retreat_scale=retreat,
                            flock_scale=flock,
                            kite=kite)


class TransformerPolicy:
    """Learned decentralized policy — cross-attention over agent tokens."""

    def __init__(self, weights: TransformerWeights):
        self.weights = weights

    @classmethod
    def default(cls, path=_DEFAULT_WEIGHTS):
        return cls(TransformerWeights.load(path))

    def decide(self, live, index, cfg, *, spatial=None):
        obs = build_observation(live, index, perception=cfg.perception,
                                spatial=spatial, alliances=cfg.alliances)
        if not obs.enemy_tokens:
            return None
        self_feat, allies, enemies, enemy_indices = encode_features(
            obs, perception=cfg.perception)
        raw = forward(self.weights, self_feat, allies, enemies)
        return decode_output(raw, enemy_indices, live, index, cfg)
