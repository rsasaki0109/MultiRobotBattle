"""``mrn_coverage_demo``: frontier detection + multi-robot allocation.

Builds a small partially-explored map, finds and clusters frontiers, assigns
each frontier to a robot by travel cost, and prints the map with robots and
their assigned frontier targets.
"""

from __future__ import annotations

import argparse

from .allocation import allocate_frontiers, bfs_free_distances
from .frontier import cluster_frontiers
from .occupancy import OccupancyGrid

# '.' free, '#' occupied, '?' unknown. Unknown pockets on the left and right
# edges give two frontier clusters; a central wall splits the free interior.
_MAP = [
    "?.......?",
    "?...#...?",
    "?...#...?",
    "?...#...?",
    "?.......?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["hungarian", "greedy"], default="hungarian")
    args = parser.parse_args()

    grid = OccupancyGrid.from_rows(_MAP)
    clusters = cluster_frontiers(grid)
    targets = [c.representative for c in clusters]
    robots = {"1": (3, 0), "2": (5, 4)}

    assignment = allocate_frontiers(grid, robots, targets, method=args.method)

    print(f"method={args.method}")
    print(f"{len(clusters)} frontier cluster(s):")
    for c in clusters:
        print(f"  rep={c.representative} size={c.size}")
    print("assignment (robot -> frontier):")
    for r, f in sorted(assignment.items()):
        dist = bfs_free_distances(grid, robots[r]).get(f)
        print(f"  robot {r} {robots[r]} -> {f}  (cost {dist})")

    # annotate the map: R = robot, F = assigned target, * = other frontier rep
    annotate = {}
    for r, cell in robots.items():
        annotate[cell] = "R"
    for f in targets:
        annotate.setdefault(f, "*")
    for f in assignment.values():
        annotate[f] = "F"

    print("map (R robot, F assigned frontier, * frontier, . free, # wall, ? unknown):")
    for y in range(grid.height - 1, -1, -1):
        row = []
        for x in range(grid.width):
            row.append(annotate.get((x, y), grid.char((x, y))))
        print("  " + "".join(row))


if __name__ == "__main__":
    main()
