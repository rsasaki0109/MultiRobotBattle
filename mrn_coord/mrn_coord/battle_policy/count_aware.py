"""Count-aware decentralized battle policies (TeamHOI-lite).

:class:`CountAwarePolicy` scales pursue / flock / retreat from global and local
force ratios, picks focus-fire targets among nearby enemies, and backs snipers
out of melee. No learned weights — pure utility rules on the token observation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .tokens import BattleObservation, build_observation


@dataclass(frozen=True)
class TacticalDecision:
    """Per-bot steering multipliers and combat target for one tick."""

    target_index: int
    pursue_scale: float = 1.0
    retreat_scale: float = 1.0
    flock_scale: float = 1.0
    kite: bool = False


class NearestPolicy:
    """Original behaviour: nearest enemy, fixed config weights."""

    def decide(self, live, index, cfg, *, spatial=None):
        me = live[index]
        positions = [(b.x, b.y) for b in live]
        best, bd = -1, float("inf")
        scan = (spatial.query_disk(me.x, me.y,
                                  math.hypot(cfg.width, cfg.height), positions)
                if spatial is not None else range(len(live)))
        for j in scan:
            if j == index:
                continue
            other = live[j]
            if other.team == me.team:
                continue
            d = math.hypot(me.x - other.x, me.y - other.y)
            if d < bd:
                bd, best = d, j
        if best < 0:
            return None
        return TacticalDecision(target_index=best)


class CountAwarePolicy:
    """Adapt tactics to ally/enemy counts and unit role (TeamHOI-lite).

    ``stance`` selects a baseline posture; ``"auto"`` blends global and local
    force ratios so the same policy works from 2v2 duels to quality-vs-quantity.
    """

    STANCES = ("auto", "aggressive", "defensive", "balanced")

    def __init__(self, stance="auto"):
        if stance not in self.STANCES:
            raise ValueError(f"unknown stance {stance!r}; choose from {self.STANCES}")
        self.stance = stance

    def decide(self, live, index, cfg, *, spatial=None):
        obs = build_observation(live, index, perception=cfg.perception,
                                spatial=spatial)
        if not obs.enemy_tokens:
            return None
        target = self._pick_target(live, index, obs, cfg)
        if target is None:
            return None
        pursue, retreat, flock = self._steering_scales(obs)
        kite = self._should_kite(live[index], live[target], cfg)
        return TacticalDecision(target_index=target,
                                pursue_scale=pursue,
                                retreat_scale=retreat,
                                flock_scale=flock,
                                kite=kite)

    def _steering_scales(self, obs: BattleObservation):
        if self.stance == "aggressive":
            return 1.45, 0.35, 0.85
        if self.stance == "defensive":
            return 0.55, 1.35, 1.45
        if self.stance == "balanced":
            return 1.0, 1.0, 1.0

        # auto — global force ratio + local contact ratio
        global_ratio = obs.n_allies / max(obs.n_enemies, 1)
        local_ratio = (obs.n_local_allies + 1) / max(obs.n_local_enemies, 1)

        if global_ratio >= 1.35:
            pursue, retreat, flock = 1.35, 0.45, 0.95
        elif global_ratio <= 0.75:
            pursue, retreat, flock = 0.6, 1.15, 1.3
        else:
            pursue, retreat, flock = 1.0, 1.0, 1.0

        local_blend = max(0.45, min(1.35, 0.55 + 0.45 * local_ratio))
        pursue *= local_blend
        flock *= max(0.85, 2.0 - local_blend)
        return pursue, retreat, flock

    def _bot_range(self, bot, cfg):
        return bot.attack_range if bot.attack_range is not None else cfg.attack_range

    def _pick_target(self, live, index, obs: BattleObservation, cfg):
        me = live[index]
        b_range = self._bot_range(me, cfg)
        best_j, best_score = None, float("-inf")

        for tok in obs.enemy_tokens:
            enemy = live[tok.live_index]
            dist = tok.dist
            score = 0.0
            if dist <= b_range:
                score += 2.0
            score += 1.5 * (1.0 - tok.hp_frac)
            score -= 0.25 * (dist / max(b_range, 1e-6))

            if me.kind == "sniper":
                ideal = 0.85 * b_range
                score += 1.0 - abs(dist - ideal) / max(b_range, 1e-6)
                if dist < 0.35 * b_range:
                    score -= 1.5
            elif me.kind == "scout" and dist > b_range:
                score -= 0.4

            if score > best_score:
                best_score, best_j = score, tok.live_index

        if best_j is not None:
            return best_j

        # fall back to nearest enemy
        best, bd = None, float("inf")
        for j, other in enumerate(live):
            if other.team == me.team:
                continue
            d = math.hypot(me.x - other.x, me.y - other.y)
            if d < bd:
                bd, best = d, j
        return best

    def _should_kite(self, me, enemy, cfg):
        if me.kind != "sniper":
            return False
        b_range = self._bot_range(me, cfg)
        dist = math.hypot(me.x - enemy.x, me.y - enemy.y)
        return dist < 0.45 * b_range


def policy_for_name(name):
    """Resolve a ``BattleConfig.tactics`` string to a policy instance."""
    if name in (None, "", "nearest"):
        return NearestPolicy()
    if name == "count_aware":
        return CountAwarePolicy("auto")
    if name.startswith("count_aware:"):
        stance = name.split(":", 1)[1]
        return CountAwarePolicy(stance)
    if name == "transformer":
        from .transformer import TransformerPolicy
        return TransformerPolicy.default()
    raise ValueError(f"unknown tactics {name!r}")
