
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml

@dataclass(frozen=True)
class WaveformCfg:
    fs_hz: float
    f0_hz: float
    f1_hz: float
    duration_s: float
    taper: str = "hann"

@dataclass(frozen=True)
class SceneTargetCfg:
    id: int
    x0_m: float
    y0_m: float
    vx_mps: float
    vy_mps: float
    amplitude: float

@dataclass(frozen=True)
class SceneCfg:
    seed: int
    max_range_m: float
    pri_s: float
    num_pings: int
    noise_snr_db: float
    targets: List[SceneTargetCfg]

@dataclass(frozen=True)
class BeamformingCfg:
    angles_start: float
    angles_stop: float
    angles_step: float
    linear_shift_pad: int = 1024

@dataclass(frozen=True)
class CfarCfg:
    pfa: float
    guard_a: int
    guard_r: int
    ref_a: int
    ref_r: int

@dataclass(frozen=True)
class BlobCfg:
    connectivity: int
    min_area: int

@dataclass(frozen=True)
class MergeCfg:
    psi_r_m: float
    psi_theta_deg: float

@dataclass(frozen=True)
class TrackingCfg:
    accel_std: float
    meas_std_m: float
    gate_chi2: float
    assignment_cost_max: float
    confirm_hits: int
    delete_misses: int
    max_meas_per_ping: int = 0

@dataclass(frozen=True)
class VizCfg:
    dyn_range_db: float = 50.0
    ping_to_plot: int = 0

@dataclass(frozen=True)
class Config:
    c_mps: float
    waveform: WaveformCfg
    hydrophones_xy_m: List[Tuple[float, float]]
    scene: SceneCfg
    beamforming: BeamformingCfg
    cfar: CfarCfg
    blob: BlobCfg
    merge: MergeCfg
    tracking: TrackingCfg | None
    viz: VizCfg

def load_config(path: str | Path) -> Config:
    path = Path(path)
    data: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

    c_mps = float(data["acoustics"]["c_mps"])

    wf = data["waveform"]
    waveform = WaveformCfg(
        fs_hz=float(wf["fs_hz"]),
        f0_hz=float(wf["f0_hz"]),
        f1_hz=float(wf["f1_hz"]),
        duration_s=float(wf["duration_s"]),
        taper=str(wf.get("taper", "hann")),
    )

    arr = data["array"]["hydrophones_m"]
    hydrophones_xy_m = [(float(x), float(y)) for x, y in arr]

    sc = data["scene"]
    targets = []
    for t in sc.get("targets", []):
        targets.append(SceneTargetCfg(
            id=int(t["id"]),
            x0_m=float(t["x0_m"]),
            y0_m=float(t["y0_m"]),
            vx_mps=float(t["vx_mps"]),
            vy_mps=float(t["vy_mps"]),
            amplitude=float(t.get("amplitude", 1.0)),
        ))
    scene = SceneCfg(
        seed=int(sc.get("seed", 0)),
        max_range_m=float(sc["max_range_m"]),
        pri_s=float(sc["pri_s"]),
        num_pings=int(sc["num_pings"]),
        noise_snr_db=float(sc.get("noise_snr_db", 10.0)),
        targets=targets,
    )

    bf = data["beamforming"]
    ang = bf["angles_deg"]
    beamforming = BeamformingCfg(
        angles_start=float(ang["start"]),
        angles_stop=float(ang["stop"]),
        angles_step=float(ang["step"]),
        linear_shift_pad=int(bf.get("linear_shift_pad", 1024)),
    )

    cf = data["cfar"]
    cfar = CfarCfg(
        pfa=float(cf["pfa"]),
        guard_a=int(cf["guard_a"]),
        guard_r=int(cf["guard_r"]),
        ref_a=int(cf["ref_a"]),
        ref_r=int(cf["ref_r"]),
    )

    bl = data.get("blob", {"connectivity": 1, "min_area": 1})
    blob = BlobCfg(connectivity=int(bl.get("connectivity", 1)),
                   min_area=int(bl.get("min_area", 1)))

    mg = data.get("merge", {"psi_r_m": 10.0, "psi_theta_deg": 6.0})
    merge = MergeCfg(psi_r_m=float(mg["psi_r_m"]),
                     psi_theta_deg=float(mg["psi_theta_deg"]))

    tr = data.get("tracking", None)
    tracking = None
    if tr is not None:
        tracking = TrackingCfg(
            accel_std=float(tr.get("accel_std", 0.25)),
            meas_std_m=float(tr.get("meas_std_m", 4.0)),
            gate_chi2=float(tr.get("gate_chi2", 9.21)),
            assignment_cost_max=float(tr.get("assignment_cost_max", 50.0)),
            confirm_hits=int(tr.get("confirm_hits", 5)),
            delete_misses=int(tr.get("delete_misses", 7)),
            max_meas_per_ping=int(tr.get("max_meas_per_ping", 0)),
        )

    vz = data.get("viz", {})
    viz = VizCfg(dyn_range_db=float(vz.get("dyn_range_db", 50.0)),
                 ping_to_plot=int(vz.get("ping_to_plot", 0)))

    return Config(
        c_mps=c_mps,
        waveform=waveform,
        hydrophones_xy_m=hydrophones_xy_m,
        scene=scene,
        beamforming=beamforming,
        cfar=cfar,
        blob=blob,
        merge=merge,
        tracking=tracking,
        viz=viz,
    )
