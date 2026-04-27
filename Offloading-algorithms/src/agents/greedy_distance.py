"""
GreedyDistanceAgent — Proximity-Based (Nearest-Node) offloading baseline.

Picks the physically closest valid node. Candidates are pre-sorted by
distance in both the dummy and Redis environments.

Standard in vehicular/IoV scenarios; cited as the "Near" baseline in:
  Xu et al., "TransEdge: Task Offloading With GNN and DRL in
  Edge-Computing-Enabled Transportation System",
  IEEE IoTJ, 2024.

This baseline models the common real-world practice of connecting to the
nearest available edge node.

Supports both action-space layouts:
  Redis:  [RSU_0 .. RSU_{N-1}, SV_0 .. SV_{K-1}]  → rsus is a list
  Dummy:  [SV_0 .. SV_{K-1}, RSU, Local]           → rsus is a single object
"""

import numpy as np
from src.config import Config


def _first_valid(mask):
    """Return the first valid action index, or 0 as a last-resort fallback."""
    valid = np.where(mask == 1)[0]
    return int(valid[0]) if len(valid) > 0 else 0


class GreedyDistanceAgent:
    """Picks the physically closest valid node (candidates pre-sorted by distance)."""

    def select_action(self, candidates, rsus, mask):
        if isinstance(rsus, list):
            # Redis layout: try SVs first (already sorted by distance), then RSUs
            for j in range(len(candidates)):
                idx = len(rsus) + j
                if idx < len(mask) and mask[idx] == 1:
                    return idx
            for i in range(len(rsus)):
                if mask[i] == 1:
                    return i
        else:
            # Dummy layout
            for i in range(len(candidates)):
                if i < len(mask) and mask[i] == 1:
                    return i
            rsu_idx = Config.MAX_NEIGHBORS
            if rsu_idx < len(mask) and mask[rsu_idx] == 1:
                return rsu_idx

        return _first_valid(mask)
