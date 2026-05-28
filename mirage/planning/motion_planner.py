from __future__ import annotations

import gymnasium as gym
import networkx as nx
import numpy as np
import torch

from mirage.envs.env_config import ENV_CONFIG


class CellGraphPlanner:
    def __init__(self, env_id: str, device: str | torch.device, subgoal_radius: float = 1.5):
        self.device = torch.device(device)
        self.subgoal_radius = float(subgoal_radius)

        ref = gym.make(
            env_id,
            reward_type=ENV_CONFIG["reward_type"],
            continuing_task=ENV_CONFIG["continuing_task"],
            reset_target=ENV_CONFIG["reset_target"],
        )
        maze = ref.unwrapped.maze
        maze_map = np.asarray(maze.maze_map)
        scale = float(maze.maze_size_scaling)
        x_center = float(maze.x_map_center)
        y_center = float(maze.y_map_center)
        ref.close()

        rows, cols = maze_map.shape
        rc_to_idx: dict[tuple[int, int], int] = {}
        cells_rc: list[tuple[int, int]] = []
        for i in range(rows):
            for j in range(cols):
                if int(maze_map[i, j]) != 1:
                    rc_to_idx[(i, j)] = len(cells_rc)
                    cells_rc.append((i, j))

        n = len(cells_rc)
        cells_xy = np.zeros((n, 2), dtype=np.float32)
        for idx, (i, j) in enumerate(cells_rc):
            cells_xy[idx, 0] = (j + 0.5) * scale - x_center
            cells_xy[idx, 1] = y_center - (i + 0.5) * scale

        g = nx.Graph()
        g.add_nodes_from(range(n))
        for (i, j), idx in rc_to_idx.items():
            for di, dj in ((1, 0), (0, 1)):
                nb = rc_to_idx.get((i + di, j + dj))
                if nb is not None:
                    g.add_edge(idx, nb)

        next_hop = np.zeros((n, n), dtype=np.int64)
        for src in range(n):
            for goal in range(n):
                if src == goal:
                    next_hop[src, goal] = goal
                    continue
                path = nx.shortest_path(g, src, goal)
                next_hop[src, goal] = path[1] if len(path) >= 2 else goal

        self.n_cells = n
        self.cells_xy = torch.from_numpy(cells_xy).to(self.device)
        self.next_hop = torch.from_numpy(next_hop).to(self.device)

    @torch.no_grad()
    def snap_to_cell(self, xy: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(xy.unsqueeze(0), self.cells_xy.unsqueeze(0)).squeeze(0)
        return d.argmin(dim=1)

    @torch.no_grad()
    def compute_subgoals(self, agent_xy: torch.Tensor, goal_xy: torch.Tensor) -> torch.Tensor:
        curr = self.snap_to_cell(agent_xy)
        goal = self.snap_to_cell(goal_xy)
        nxt = self.next_hop[curr, goal]
        return self.cells_xy[nxt]

    @torch.no_grad()
    def should_advance(self, agent_xy: torch.Tensor, subgoal_xy: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(agent_xy - subgoal_xy, dim=1) < self.subgoal_radius
