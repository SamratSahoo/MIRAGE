from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from mirage.encoder.load import is_masked, load_encoder
from mirage.paths import project_path, resolve_path


def _infer(cfg: dict, key: str, default):
    v = cfg.get(key, default)
    return default if v is None else v


class LatentGraphPlanner:

    def __init__(self, cfg: dict, device, num_envs: int, state_mode: str):
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.state_mode = str(state_mode)

        self.imagination = str(_infer(cfg, "imagination", "inverse"))
        if self.imagination not in ("inverse", "forward", "none"):
            raise ValueError(f"unknown graph_planning imagination {self.imagination!r}")
        self.subgoal_threshold = float(_infer(cfg, "subgoal_threshold", 0.25))
        self.forward_goal_threshold = float(_infer(cfg, "forward_goal_threshold", 0.20))
        self.forward_horizon = int(_infer(cfg, "forward_horizon", 50))
        self.forward_stride = max(1, int(_infer(cfg, "forward_stride", 2)))
        self.inverse_k_sweep = list(_infer(cfg, "inverse_k_sweep", [3, 5, 8]))
        self.lookahead = max(1, int(_infer(cfg, "lookahead", 1)))
        self.max_subgoals = int(_infer(cfg, "max_subgoals", 16))

        self._forward_active = self.imagination == "forward" and self.state_mode == "latent"
        if self.imagination == "forward" and not self._forward_active:
            print(f"[graph_planning] WARNING: imagination='forward' with state_mode="
                  f"{self.state_mode!r}: forward imagination requires latent states; "
                  f"disconnected pairs will fall back to the goal latent.")

        enc_path = _infer(cfg, "encoder_ckpt",
                          project_path("runs_encoder", "dual_input_masked", "encoder_best.pt"))
        self.encoder, enc_cfg, _ = load_encoder(enc_path, device=self.device, eval_mode=True)
        if not is_masked(enc_cfg):
            raise ValueError(
                f"graph_planning requires a dual-input masked encoder; '{enc_path}' is not masked")
        self.latent_dim = int(self.encoder.latent_dim)

        graph_npz = resolve_path(_infer(cfg, "graph_npz",
                                         project_path("checkpoints", "graph", "graph_K500.npz")))
        G = np.load(graph_npz)
        centroids = np.asarray(G["centroids"], dtype=np.float32)
        self.K = int(centroids.shape[0])
        cfg_K = int(_infer(cfg, "K", self.K))
        if cfg_K != self.K:
            raise ValueError(f"graph_planning K={cfg_K} != centroids in '{graph_npz}' (K={self.K})")
        cent = torch.from_numpy(centroids).to(self.device)
        self.centroids = cent
        self.centroids_unit = F.normalize(cent, dim=-1, eps=1e-8)

        self.disjoint_components = cfg.get("disjoint_components", 5)
        edges = np.asarray(G["edges"], dtype=np.int64)
        if self.disjoint_components is not None and int(self.disjoint_components) > 1:
            edges = self._enforce_disjoint_components(edges, int(self.disjoint_components))
        self._build_predecessors(edges)

        self.fwd = None
        self.iwm = None
        self.iwm_k_max = None
        if self._forward_active:
            self.fwd = self._load_forward_wm(_infer(cfg, "forward_wm_ckpt", None))
        elif self.imagination == "inverse":
            self.iwm, self.iwm_k_max = self._load_inverse_wm(_infer(cfg, "inverse_wm_ckpt", None))

        N = self.num_envs
        D = self.latent_dim
        self.seq = torch.zeros((N, self.max_subgoals, D), device=self.device)
        self.seq_len = torch.zeros(N, dtype=torch.long, device=self.device)
        self.ptr = torch.zeros(N, dtype=torch.long, device=self.device)
        self.plan_goal_xy = torch.full((N, 2), float("nan"), device=self.device)
        self._arange = torch.arange(N, device=self.device)

    def _build_predecessors(self, edges: np.ndarray) -> None:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        keep = edges[:, 0] != edges[:, 1]
        e = edges[keep]
        data = np.ones(e.shape[0], dtype=np.float32)
        A = csr_matrix((data, (e[:, 0], e[:, 1])), shape=(self.K, self.K))
        dist, pred = dijkstra(A, directed=True, indices=np.arange(self.K),
                              return_predecessors=True, unweighted=True)
        self._pred = pred.astype(np.int64)
        self._reach = np.isfinite(dist)

    def _enforce_disjoint_components(self, edges: np.ndarray, k: int) -> np.ndarray:
        from sklearn.cluster import KMeans
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        cent = self.centroids.detach().cpu().numpy()
        grp = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(cent)
        keep = grp[edges[:, 0]] == grp[edges[:, 1]]
        pruned = edges[keep]
        self._node_group = grp
        e_ns = pruned[pruned[:, 0] != pruned[:, 1]]
        A = csr_matrix((np.ones(e_ns.shape[0]), (e_ns[:, 0], e_ns[:, 1])), shape=(self.K, self.K))
        n_cc, _ = connected_components(A, directed=True, connection="weak")
        print(f"[graph_planning] enforce_disjoint_components(k={k}): kept "
              f"{pruned.shape[0]}/{edges.shape[0]} edges -> {n_cc} weakly-connected components "
              f"(>= {k} by construction)")
        return pruned

    def _load_forward_wm(self, ckpt):
        from mirage.world_model.models import DynamicsEnsemble
        path = resolve_path(ckpt or project_path(
            "checkpoints", "forward_world_model", "dynamics_dualinput_best.pt"))
        ck = torch.load(path, map_location=self.device, weights_only=False)
        c = ck["config"]
        sd = ck["ensemble"]
        act_dim = int(sd["members.0.net.0.weight"].shape[1]) - self.latent_dim
        fwd = DynamicsEnsemble(n_members=int(c["n_members"]), latent_dim=self.latent_dim,
                               act_dim=act_dim, hidden_dim=int(c["hidden_dim"]),
                               n_hidden=int(c["n_hidden"]))
        fwd.load_state_dict(sd)
        fwd.eval().to(self.device)
        for p in fwd.parameters():
            p.requires_grad_(False)
        return fwd

    def _load_inverse_wm(self, ckpt):
        from mirage.inverse_world_model.models import InverseWorldModel
        path = resolve_path(ckpt or project_path(
            "checkpoints", "inverse_world_model", "inverse_best.pt"))
        ck = torch.load(path, map_location=self.device, weights_only=False)
        c = ck["config"]
        sd = ck["model"]
        k_max = int(c["k_max"])
        act_dim = int(sd["act_head.weight"].shape[0]) // k_max
        m = InverseWorldModel(latent_dim=self.latent_dim, act_dim=act_dim, k_max=k_max,
                              hidden_dim=int(c["hidden_dim"]), n_hidden=int(c["n_hidden"]),
                              predict_k=bool(c.get("predict_k", False)))
        m.load_state_dict(sd)
        m.eval().to(self.device)
        for p in m.parameters():
            p.requires_grad_(False)
        return m, k_max

    @torch.no_grad()
    def encode(self, obs_107: torch.Tensor, achieved_xy: torch.Tensor):
        proprio = obs_107[:, :27]
        z_state = self.encoder.encode_full(torch.cat([proprio, achieved_xy], dim=-1))
        goal_xy = obs_107[:, 105:107]
        z_goal = self.encoder.encode_goal(torch.cat([proprio, goal_xy], dim=-1))
        return z_state, z_goal

    @torch.no_grad()
    def assign_node(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cdist(z, self.centroids).argmin(dim=1)

    @torch.no_grad()
    def _fwd_predict(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        mu_all, _ = self.fwd(z, a)
        return F.normalize(mu_all.mean(0), dim=-1, eps=1e-8)

    @torch.no_grad()
    def update(self, obs_107: torch.Tensor, achieved_xy: torch.Tensor,
               done_mask: torch.Tensor | None = None, act_fn=None) -> dict:
        z_state, z_goal = self.encode(obs_107, achieved_xy)
        goal_xy = obs_107[:, 105:107]

        d_goal = (self.plan_goal_xy - goal_xy).norm(dim=1)
        replan = (~torch.isfinite(d_goal)) | (d_goal > 1e-6)
        if done_mask is not None:
            replan = replan | done_mask.to(torch.bool)
        if bool(replan.any()):
            idx = torch.nonzero(replan, as_tuple=False).squeeze(1)
            self._replan(idx, z_state, z_goal, goal_xy, act_fn)

        self._advance(z_state)

        has = self.ptr < self.seq_len
        tgt = (self.ptr + (self.lookahead - 1)).clamp(min=0, max=self.max_subgoals - 1)
        tgt = torch.minimum(tgt, (self.seq_len - 1).clamp(min=0))
        sub = self.seq[self._arange, tgt]
        z_subgoal = torch.where(has.unsqueeze(1), sub, z_goal)
        return {"z_state": z_state, "z_goal": z_goal, "z_subgoal": z_subgoal}

    def _advance(self, z_state: torch.Tensor) -> None:
        for _ in range(self.max_subgoals):
            active = self.ptr < self.seq_len
            if not bool(active.any()):
                break
            cur = self.seq[self._arange, self.ptr.clamp(max=self.max_subgoals - 1)]
            d = (z_state - cur).norm(dim=1)
            adv = active & (d < self.subgoal_threshold)
            if not bool(adv.any()):
                break
            self.ptr = self.ptr + adv.long()

    @torch.no_grad()
    def _replan(self, idx, z_state, z_goal, goal_xy, act_fn) -> None:
        zs = z_state[idx]
        zg = z_goal[idx]
        M = int(idx.shape[0])
        c_curr = self.assign_node(zs).cpu().numpy()
        c_goal = self.assign_node(zg).cpu().numpy()

        new_seq = torch.zeros((M, self.max_subgoals, self.latent_dim), device=self.device)
        new_len = torch.ones(M, dtype=torch.long, device=self.device)
        new_seq[:, 0, :] = zg

        same = c_curr == c_goal
        reachable = self._reach[c_curr, c_goal] & (~same)

        graph_local = np.nonzero(reachable)[0]
        if graph_local.size:
            pairs: dict = {}
            for li in graph_local:
                pairs.setdefault((int(c_curr[li]), int(c_goal[li])), []).append(int(li))
            for (s, g), members in pairs.items():
                path = self._reconstruct_path(s, g)
                if path is None or len(path) < 2:
                    continue
                inter_nodes = path[1:][:self.max_subgoals]
                cseq = self.centroids_unit[torch.tensor(inter_nodes, device=self.device)]
                L = int(cseq.shape[0])
                mt = torch.tensor(members, device=self.device, dtype=torch.long)
                new_seq[mt, :L, :] = cseq.unsqueeze(0)
                new_seq[mt, L - 1, :] = zg[mt]
                new_len[mt] = L

        imag_local = np.nonzero((~reachable) & (~same))[0]
        if imag_local.size and self.imagination != "none":
            il = torch.tensor(imag_local, device=self.device, dtype=torch.long)
            z0, zk = zs[il], zg[il]
            seqs = lens = None
            if self.imagination == "inverse" and self.iwm is not None:
                seqs, lens = self._imagine_inverse(z0, zk)
            elif self._forward_active and self.fwd is not None and act_fn is not None:
                seqs, lens = self._imagine_forward(z0, zk, goal_xy[idx][il], act_fn)
            if seqs is not None:
                new_seq[il] = seqs
                new_len[il] = lens

        self.seq[idx] = new_seq
        self.seq_len[idx] = new_len
        self.ptr[idx] = 0
        self.plan_goal_xy[idx] = goal_xy[idx]

    def _reconstruct_path(self, s: int, g: int):
        if s == g:
            return [s]
        if not bool(self._reach[s, g]):
            return None
        pred_row = self._pred[s]
        path = [g]
        j = g
        for _ in range(self.K + 1):
            p = int(pred_row[j])
            if p < 0:
                return None
            path.append(p)
            if p == s:
                path.reverse()
                return path
            j = p
        return None

    @torch.no_grad()
    def _pad_sequences(self, inter_list, zk):
        m = zk.shape[0]
        seqs = torch.zeros((m, self.max_subgoals, self.latent_dim), device=self.device)
        lens = torch.ones(m, dtype=torch.long, device=self.device)
        for i in range(m):
            inter = inter_list[i]
            if inter is not None and inter.shape[0] > 0:
                dd = (inter - zk[i]).norm(dim=1)
                close = torch.nonzero(dd < self.forward_goal_threshold, as_tuple=False)
                if close.numel() > 0:
                    inter = inter[:int(close[0])]
                seq_i = torch.cat([inter, zk[i:i + 1]], dim=0)
            else:
                seq_i = zk[i:i + 1]
            seq_i = seq_i[:self.max_subgoals]
            L = int(seq_i.shape[0])
            seqs[i, :L] = seq_i
            lens[i] = L
        return seqs, lens

    @torch.no_grad()
    def _imagine_inverse(self, z0, zk):
        m = z0.shape[0]
        K = self.iwm_k_max
        if self.iwm.predict_k:
            out = self.iwm(z0, zk)
            lat = F.normalize(self.iwm.reconstruct_latents(z0, out["deltas"]), dim=-1)
            k_pred = out.get("k_pred")
            inter_list = []
            for i in range(m):
                ki = K if k_pred is None else max(1, min(int(round(float(k_pred[i]) * K)), K))
                inter_list.append(lat[i, :max(ki - 1, 0)])
            return self._pad_sequences(inter_list, zk)

        cand = []
        for k in self.inverse_k_sweep:
            k = int(min(max(1, k), K))
            kn = torch.full((m, 1), k / float(K), device=self.device)
            out = self.iwm(z0, zk, kn)
            lat = F.normalize(self.iwm.reconstruct_latents(z0, out["deltas"]), dim=-1)
            endpoint = lat[:, k - 1, :]
            cand.append((k, lat, (endpoint - zk).norm(dim=1)))
        dists = torch.stack([c[2] for c in cand], dim=1)
        best_j = dists.argmin(dim=1)
        inter_list = []
        for i in range(m):
            k, lat, _ = cand[int(best_j[i])]
            inter_list.append(lat[i, :max(k - 1, 0)])
        return self._pad_sequences(inter_list, zk)

    @torch.no_grad()
    def _imagine_forward(self, z0, zk, goal_xy, act_fn):
        m = z0.shape[0]
        traj = [z0]
        z = z0
        stopped = torch.zeros(m, dtype=torch.bool, device=self.device)
        stop_step = torch.full((m,), self.forward_horizon, dtype=torch.long, device=self.device)
        for h in range(self.forward_horizon):
            a = act_fn(z, zk, goal_xy)
            z = self._fwd_predict(z, a)
            traj.append(z)
            newly = (~stopped) & ((z - zk).norm(dim=1) < self.forward_goal_threshold)
            stop_step = torch.where(newly, torch.full_like(stop_step, h + 1), stop_step)
            stopped = stopped | newly
            if bool(stopped.all()):
                break
        traj = torch.stack(traj, dim=1)
        inter_list = [traj[i, 1:int(stop_step[i]) + 1:self.forward_stride] for i in range(m)]
        return self._pad_sequences(inter_list, zk)

    def reset_state(self) -> None:
        self.seq_len.zero_()
        self.ptr.zero_()
        self.plan_goal_xy.fill_(float("nan"))
