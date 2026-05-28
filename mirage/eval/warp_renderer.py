from __future__ import annotations

import os
from sys import platform

import gymnasium as gym
import numpy as np

from mirage.envs.env_config import ENV_CONFIG  # noqa: F401


class WarpRenderer:
    def __init__(self, env_id, env_cfg, width=640, height=480,
                 azimuth=90.0, elevation=-90.0, distance=15.0):
        os.environ.setdefault("MUJOCO_GL", "glfw" if platform == "darwin" else "egl")
        import mujoco
        self._mujoco = mujoco
        ref = gym.make(
            env_id,
            render_mode=None,
            reward_type=env_cfg["reward_type"],
            continuing_task=env_cfg["continuing_task"],
            reset_target=env_cfg["reset_target"],
        )
        self.mjm = ref.unwrapped.model
        ref.close()
        self.mjm.vis.global_.offwidth = max(int(self.mjm.vis.global_.offwidth), int(width))
        self.mjm.vis.global_.offheight = max(int(self.mjm.vis.global_.offheight), int(height))
        self.mjd = mujoco.MjData(self.mjm)
        self.renderer = mujoco.Renderer(self.mjm, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation
        self.cam.distance = distance
        self.cam.lookat[:] = [0.0, 0.0, 0.0]
        self.width = width
        self.height = height

    def render(self, qpos, goal_xy=None, subgoal_xy=None):
        self.mjd.qpos[:] = qpos
        self._mujoco.mj_forward(self.mjm, self.mjd)
        self.renderer.update_scene(self.mjd, camera=self.cam)
        scene = self.renderer.scene
        if goal_xy is not None and scene.ngeom < scene.maxgeom:
            g = scene.geoms[scene.ngeom]
            self._mujoco.mjv_initGeom(
                g,
                type=self._mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.3, 0.0, 0.0], dtype=np.float64),
                pos=np.array([float(goal_xy[0]), float(goal_xy[1]), 0.3], dtype=np.float64),
                mat=np.eye(3, dtype=np.float64).flatten(),
                rgba=np.array([1.0, 0.15, 0.15, 1.0], dtype=np.float32),
            )
            g.emission = 1.0
            scene.ngeom += 1
        if subgoal_xy is not None and scene.ngeom < scene.maxgeom:
            g = scene.geoms[scene.ngeom]
            self._mujoco.mjv_initGeom(
                g,
                type=self._mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.25, 0.0, 0.0], dtype=np.float64),
                pos=np.array([float(subgoal_xy[0]), float(subgoal_xy[1]), 0.25], dtype=np.float64),
                mat=np.eye(3, dtype=np.float64).flatten(),
                rgba=np.array([0.15, 0.5, 1.0, 1.0], dtype=np.float32),
            )
            g.emission = 0.7
            scene.ngeom += 1
        return self.renderer.render()

    def close(self):
        self.renderer.close()
