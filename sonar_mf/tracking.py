
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment

def polar_to_cart(r: float, az_deg: float) -> np.ndarray:
    th = np.deg2rad(az_deg)
    return np.array([r*np.cos(th), r*np.sin(th)], dtype=np.float64)

@dataclass
class Track:
    id: int
    x: np.ndarray
    P: np.ndarray
    hits: int = 0
    misses: int = 0
    confirmed: bool = False
    history: list = field(default_factory=list)

def ncv_F_Q(dt: float, accel_std: float):
    F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
    q = accel_std**2
    G = np.array([[0.5*dt*dt,0],[0,0.5*dt*dt],[dt,0],[0,dt]], dtype=np.float64)
    Q = G @ (q*np.eye(2)) @ G.T
    return F, Q

def kf_predict(x, P, F, Q):
    return F@x, F@P@F.T + Q

def kf_update(xp, Pp, z, R):
    H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
    y = z - H@xp
    S = H@Pp@H.T + R
    Sinv = np.linalg.inv(S)
    K = Pp@H.T@Sinv
    x = xp + K@y
    P = (np.eye(4) - K@H)@Pp
    d2 = float(y.T@Sinv@y)
    return x, P, d2

class MultiTargetTracker:
    def __init__(self, dt, accel_std, meas_std_m, gate_chi2, assignment_cost_max, confirm_hits, delete_misses):
        self.dt = float(dt)
        self.F, self.Q = ncv_F_Q(self.dt, float(accel_std))
        self.R = (float(meas_std_m)**2) * np.eye(2)
        self.gate_chi2 = float(gate_chi2)
        self.assignment_cost_max = float(assignment_cost_max)
        self.confirm_hits = int(confirm_hits)
        self.delete_misses = int(delete_misses)
        self.tracks: list[Track] = []
        self.archive: list[Track] = []
        self.next_id = 1

    def step(self, dets_polar: list[dict], t_s: float):
        for tr in self.tracks:
            tr.x, tr.P = kf_predict(tr.x, tr.P, self.F, self.Q)

        Z = [polar_to_cart(d["range_m"], d["azimuth_deg"]) for d in dets_polar]

        if not self.tracks:
            for z in Z: self._spawn(z, t_s)
            self._manage()
            return

        if not Z:
            for tr in self.tracks:
                tr.misses += 1
                tr.history.append({"t_s": t_s, "x": tr.x.tolist(), "meas": None, "d2": None})
            self._manage()
            return

        Tn, Mn = len(self.tracks), len(Z)
        C = np.full((Tn, Mn), 1e6, dtype=np.float64)
        H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)

        for i, tr in enumerate(self.tracks):
            S = H@tr.P@H.T + self.R
            Sinv = np.linalg.inv(S)
            for j, z in enumerate(Z):
                y = z - H@tr.x
                d2 = float(y.T@Sinv@y)
                if d2 <= self.gate_chi2:
                    C[i,j] = min(np.sqrt(d2), self.assignment_cost_max)

        rows, cols = linear_sum_assignment(C)
        assigned_t = set(); assigned_m = set()

        for i, j in zip(rows, cols):
            if C[i,j] >= 1e5:
                continue
            tr = self.tracks[i]
            x, P, d2 = kf_update(tr.x, tr.P, Z[j], self.R)
            tr.x, tr.P = x, P
            tr.hits += 1
            tr.misses = 0
            tr.history.append({"t_s": t_s, "x": tr.x.tolist(), "meas": Z[j].tolist(), "d2": d2})
            assigned_t.add(i); assigned_m.add(j)

        for i, tr in enumerate(self.tracks):
            if i not in assigned_t:
                tr.misses += 1
                tr.history.append({"t_s": t_s, "x": tr.x.tolist(), "meas": None, "d2": None})

        for j, z in enumerate(Z):
            if j not in assigned_m:
                self._spawn(z, t_s)

        self._manage()

    def _spawn(self, z: np.ndarray, t_s: float):
        x = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float64)
        P = np.diag([self.R[0,0], self.R[1,1], 10.0, 10.0]).astype(np.float64)
        tr = Track(id=self.next_id, x=x, P=P, hits=1, misses=0, confirmed=False)
        tr.history.append({"t_s": t_s, "x": tr.x.tolist(), "meas": z.tolist(), "d2": 0.0})
        self.next_id += 1
        self.tracks.append(tr)

    def _manage(self):
        for tr in self.tracks:
            if (not tr.confirmed) and tr.hits >= self.confirm_hits:
                tr.confirmed = True
        keep = []
        for tr in self.tracks:
            if tr.misses < self.delete_misses:
                keep.append(tr)
            else:
                self.archive.append(tr)
        self.tracks = keep

    def all_tracks(self) -> list[Track]:
        return self.tracks + self.archive
