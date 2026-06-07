"""Decentralized battle tactics — TeamHOI-style token observations + policies.

Each robot builds a local observation from teammate and enemy *tokens* (relative
position / velocity / health in the observer's frame). A :class:`BattlePolicy`
maps that view to steering weights and a target pick. The default
:class:`NearestPolicy` preserves the original nearest-enemy behaviour; the
:class:`CountAwarePolicy` adapts pursue / flock / retreat scales to team size
and local force ratios, focuses wounded enemies, and kites snipers.
"""

from .count_aware import CountAwarePolicy, NearestPolicy, TacticalDecision, policy_for_name
from .tokens import AgentToken, BattleObservation, build_observation
from .transformer import TransformerPolicy

__all__ = [
    "AgentToken",
    "BattleObservation",
    "CountAwarePolicy",
    "NearestPolicy",
    "TacticalDecision",
    "TransformerPolicy",
    "build_observation",
    "policy_for_name",
]
