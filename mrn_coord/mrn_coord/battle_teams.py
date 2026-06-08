"""Team / alliance helpers for multi-army swarm battles."""

from __future__ import annotations

RED, BLUE, GREEN, YELLOW = 0, 1, 2, 3


def alliance_of(alliances, team):
    """Alliance id for ``team``; without ``alliances`` each team is its own bloc."""
    if not alliances:
        return team
    return alliances.get(team, team)


def teams_are_enemies(alliances, team_a, team_b):
    """True when two teams should fight (different alliance, or FFA default)."""
    if team_a == team_b:
        return False
    if not alliances:
        return True
    return alliance_of(alliances, team_a) != alliance_of(alliances, team_b)
