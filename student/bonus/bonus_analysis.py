"""Bonus evidence for Day 23: visualisation, CVAT export and calibration drift.

The script never edits platform/ or student/artifacts/. It runs in two stages:

1. ``cache``: decode the segment once, run the provided FPN-ResNet detector and
   save detections, labels, calibration and FRONT images to ``data/bonus_cache``
   (git-ignored, the dataset must not be committed).
2. ``replay``: re-run the exact tracking loop of ``fusion-run-lab`` from the cache,
   optionally with a perturbed camera extrinsic, and write evidence to
   ``student/bonus/``.

Replay with no perturbation must reproduce ``student/artifacts/metrics_fused.json``;
``check`` verifies that, so every bonus number comes from the same pipeline as the
graded run.

Usage (from repo root):
    python student/bonus/bonus_analysis.py cache --config student/config/paths.yaml
    python student/bonus/bonus_analysis.py check
    python student/bonus/bonus_analysis.py calibration
    python student/bonus/bonus_analysis.py visualise
    python student/bonus/bonus_analysis.py cvat
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BONUS = REPO / "student" / "bonus"
CACHE = REPO / "data" / "bonus_cache" / "segment_cache.pickle"
SEED = 0


# --------------------------------------------------------------------------- cache


def _label_record(label):
    box = label.box
    return {
        "id": label.id, "center_x": box.center_x, "center_y": box.center_y,
        "center_z": box.center_z, "length": box.length, "width": box.width,
        "height": box.height, "heading": box.heading,
    }


def _as_label(record):
    """Rebuild the attribute shape that evaluation.tracking_counts reads."""
    return SimpleNamespace(id=record["id"], box=SimpleNamespace(**{
        key: value for key, value in record.items() if key != "id"
    }))


def build_cache(config_path: Path) -> None:
    import torch

    from fusion_lab.evaluation import detection_counts, valid_ground_truth
    from fusion_lab.lidar_pcl import pcl_from_range_image
    from fusion_lab.scripts.run_lab import (
        _load_paths_config, _resolve_weights, _setup_import_paths,
    )
    from fusion_lab.workspace_loader import load_workspace

    _setup_import_paths()
    from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
    from simple_waymo_open_dataset_reader import utils as waymo_utils

    cfg = _load_paths_config(config_path)
    ws = load_workspace()
    det_pipe, bev, det_metrics = ws["detection_pipeline"], ws["bev_mapping"], ws["detection_metrics"]
    weights = _resolve_weights(cfg)
    det_cfg = det_pipe.load_fpn_resnet_config(str(weights))
    model = det_pipe.create_fpn_model(det_cfg, str(weights))
    tfrecord = Path(cfg["waymo_dir"]) / cfg["segment"]
    frame_start, frame_end = int(cfg["frame_start"]), int(cfg["frame_end"])
    vehicle = label_pb2.Label.Type.TYPE_VEHICLE

    frames, camera_calib = [], None
    for cnt, frame in enumerate(WaymoDataFileReader(str(tfrecord))):
        if cnt < frame_start:
            continue
        if cnt > frame_end:
            break
        if camera_calib is None:
            calib = waymo_utils.get(frame.context.camera_calibrations, dataset_pb2.CameraName.FRONT)
            camera_calib = {
                "intrinsic": list(calib.intrinsic), "width": calib.width,
                "height": calib.height, "transform": list(calib.extrinsic.transform),
            }
        points = pcl_from_range_image(frame, dataset_pb2.LaserName.TOP)
        tensor = torch.from_numpy(bev.bev_maps_from_pcl(points, det_cfg)).unsqueeze(0).float()
        detections = det_pipe.detect_objects_from_bev(tensor, model, det_cfg)
        labels = valid_ground_truth(frame.laser_labels, det_cfg, vehicle)
        group = next((g for g in frame.camera_labels if g.name == dataset_pb2.CameraName.FRONT), None)
        front = None if group is None else [
            {"id": l.id, "cx": l.box.center_x, "cy": l.box.center_y,
             "w": l.box.length, "h": l.box.width}
            for l in group.labels if l.type == vehicle
        ]
        image = next((im.image for im in frame.images if im.name == dataset_pb2.CameraName.FRONT), None)
        in_window = points[(points[:, 0] >= det_cfg.lim_x[0]) & (points[:, 0] <= det_cfg.lim_x[1])
                           & (points[:, 1] >= det_cfg.lim_y[0]) & (points[:, 1] <= det_cfg.lim_y[1])]
        rng = np.random.default_rng(cnt)
        keep = rng.choice(len(in_window), size=min(25000, len(in_window)), replace=False)
        frames.append({
            "frame": cnt,
            "detections": [list(map(float, d)) for d in detections],
            "labels": [_label_record(l) for l in labels],
            "det_counts": detection_counts(labels, detections, det_metrics),
            "front": front,
            "image": bytes(image) if image is not None else None,
            "bev_points": in_window[keep, :3].astype(np.float32),
        })
        print(f"cached frame {cnt}: {len(detections)} detections, {len(labels)} GT", flush=True)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "segment": tfrecord.name, "frames": frames, "camera_calib": camera_calib,
        "lim_x": list(det_cfg.lim_x), "lim_y": list(det_cfg.lim_y),
    }
    with CACHE.open("wb") as f:
        pickle.dump(payload, f)
    print(f"wrote {CACHE}")


def load_cache():
    with CACHE.open("rb") as f:
        return pickle.load(f)


# -------------------------------------------------------------------------- replay


def _rotation_z(degrees: float) -> np.ndarray:
    a = np.deg2rad(degrees)
    r = np.eye(4)
    r[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return r


def camera_sensor(cache, ws, yaw_deg: float = 0.0, shift_m=(0.0, 0.0, 0.0)):
    """Camera sensor whose extrinsic believes a drifted mounting.

    The real image is unchanged; only the calibration the tracker uses is rotated by
    ``yaw_deg`` about the camera up axis and shifted by ``shift_m`` (camera axes).
    """
    from fusion_lab.tracking.sensors import Sensor

    calib = cache["camera_calib"]
    transform = np.asarray(calib["transform"], dtype=float).reshape(4, 4)
    drift = _rotation_z(yaw_deg)
    drift[:3, 3] = shift_m
    perturbed = transform @ drift
    namespace = SimpleNamespace(
        intrinsic=calib["intrinsic"], width=calib["width"], height=calib["height"],
        extrinsic=SimpleNamespace(transform=perturbed.reshape(-1).tolist()),
    )
    return Sensor("camera", namespace, ws["camera_fusion"])


class _LimitsConfig(SimpleNamespace):
    pass


def replay(cache, mode: str, yaw_deg: float = 0.0, shift_m=(0.0, 0.0, 0.0),
           keep_history: bool = False):
    """Replay the fusion-run-lab tracking loop on cached detections."""
    from fusion_lab import tracking_params
    from fusion_lab.evaluation import aggregate_records, tracking_counts
    from fusion_lab.scripts.run_lab import _lidar_observations
    from fusion_lab.tracking.filter import Filter
    from fusion_lab.tracking.manager import TrackManager
    from fusion_lab.tracking.sensors import Sensor
    from fusion_lab.workspace_loader import load_workspace

    ws = load_workspace()
    kalman, assoc = ws["kalman"], ws["association"]
    from scipy.stats import chi2
    camera_gate = chi2.ppf(tracking_params.gating_threshold, df=2)

    innovations = []

    class RecordingFilter(Filter):
        def update(self, track, meas):
            if meas.sensor.name == "camera":
                H = meas.sensor.get_H(track.x)
                gamma = np.asarray(kalman.innovation(track.x, meas)).reshape(-1)
                S = np.asarray(kalman.innovation_covariance(track.P, meas, H))
                innovations.append({
                    "frame": current_frame, "track": track.id,
                    "gamma_u": float(gamma[0]), "gamma_v": float(gamma[1]),
                    "nis": float(gamma @ np.linalg.solve(S, gamma)),
                })
            super().update(track, meas)

    rng = np.random.default_rng(SEED)
    lidar = Sensor("lidar", None, ws["camera_fusion"])
    camera = camera_sensor(cache, ws, yaw_deg, shift_m) if mode == "fused" else None
    limits = _LimitsConfig(lim_x=cache["lim_x"], lim_y=cache["lim_y"])
    KF = RecordingFilter(kalman)
    manager = TrackManager(ws["track_management"])
    records, history, gating = [], [], []
    current_frame = None
    for item in cache["frames"]:
        cnt = current_frame = item["frame"]
        labels = [_as_label(r) for r in item["labels"]]
        observations = _lidar_observations(cnt, item["detections"], lidar, limits)
        for track in manager.track_list:
            KF.predict(track)
            track.set_t(cnt * tracking_params.dt)
        assoc.associate_and_update(manager, observations, KF, lidar)
        snapshot_lidar = [(t.id, np.asarray(t.x).reshape(-1)[:3].copy(), t.state)
                          for t in manager.track_list]
        camera_obs = None
        if camera is not None and item["front"] is not None:
            camera_obs = []
            for box in item["front"]:
                centre = np.array([box["cx"], box["cy"]])
                camera.generate_measurement(cnt, centre + rng.normal(0, 0.5, 2), camera_obs)
            # Gate diagnostics: for each visible confirmed track, its best camera d^2.
            visible = [t for t in manager.track_list
                       if t.state == "confirmed" and camera.in_fov(t.x)]
            if visible and camera_obs:
                costs = np.array([[assoc.mahalanobis_distance(t, m) for m in camera_obs]
                                  for t in visible])
                best = costs.min(axis=1)
                gating.append({"frame": cnt, "visible": len(visible),
                               "rejected": int((best >= camera_gate).sum())})
            assoc.associate_and_update(manager, camera_obs, KF, camera)
        record = {"mode": mode, "frame": cnt, **item["det_counts"],
                  "valid_gt": len(labels), **tracking_counts(manager.track_list, labels)}
        records.append(record)
        if keep_history:
            history.append({
                "frame": cnt,
                "after_lidar": snapshot_lidar,
                "after_camera": [(t.id, np.asarray(t.x).reshape(-1)[:3].copy(), t.state,
                                  t.length, t.width, t.height, t.yaw, t.score)
                                 for t in manager.track_list],
                "lidar_obs": [np.asarray(o.z).reshape(-1).copy() for o in observations],
                "camera_obs": None if camera_obs is None else
                              [np.asarray(o.z).reshape(-1).copy() for o in camera_obs],
            })
    frames = [cache["frames"][0]["frame"], cache["frames"][-1]["frame"]]
    metrics = aggregate_records(records, mode, frames, SEED, cache["segment"])
    return {"metrics": metrics, "records": records, "innovations": innovations,
            "gating": gating, "history": history, "camera": camera, "gate": camera_gate}


# -------------------------------------------------------------------------- tasks


def check_reproduces_artifacts(cache) -> None:
    artifacts = REPO / "student" / "artifacts"
    for mode in ("lidar", "fused"):
        expected = json.loads((artifacts / f"metrics_{mode}.json").read_text())
        got = replay(cache, mode)["metrics"]
        same = got == expected
        print(f"{mode}: replay {'matches' if same else 'DIFFERS FROM'} metrics_{mode}.json")
        if not same:
            print(json.dumps({"replay": got["tracking"], "artifact": expected["tracking"]}, indent=2))
            sys.exit(1)


def _quality(stats, det_tp):
    precision = stats["matches"] / max(1, stats["matches"] + stats["ghost_track_frames"])
    coverage = stats["matches"] / max(1, det_tp)
    return precision, coverage


def calibration_study(cache) -> None:
    levels = [(-0.5, (0, 0, 0)), (-0.25, (0, 0, 0)), (0.0, (0, 0, 0)), (0.25, (0, 0, 0)), (0.5, (0, 0, 0)), (1.0, (0, 0, 0)),
              (2.0, (0, 0, 0)), (5.0, (0, 0, 0)), (0.0, (0, 0.5, 0))]
    lidar = replay(cache, "lidar")["metrics"]
    rows = []
    for yaw, shift in levels:
        result = replay(cache, "fused", yaw, shift)
        stats = result["metrics"]["tracking"]["fused"]
        inn = result["innovations"]
        gu = np.array([i["gamma_u"] for i in inn]) if inn else np.array([np.nan])
        nis = np.array([i["nis"] for i in inn]) if inn else np.array([np.nan])
        visible = sum(g["visible"] for g in result["gating"])
        rejected = sum(g["rejected"] for g in result["gating"])
        precision, coverage = _quality(stats, result["metrics"]["detection"]["tp"])
        f_i = cache["camera_calib"]["intrinsic"][0]
        rows.append({
            "yaw_deg": yaw, "shift_left_m": shift[1],
            "expected_shift_px": round(float(f_i * np.tan(np.deg2rad(yaw))), 1),
            "camera_updates": len(inn),
            "mean_gamma_u_px": round(float(np.mean(gu)), 2),
            "mean_abs_gamma_u_px": round(float(np.mean(np.abs(gu))), 2),
            "mean_nis": round(float(np.mean(nis)), 2),
            "visible_confirmed_track_frames": visible,
            "gate_rejected_track_frames": rejected,
            "gate_reject_rate": round(rejected / max(1, visible), 3),
            "rmse": stats["rmse"], "matches": stats["matches"],
            "ghost_track_frames": stats["ghost_track_frames"],
            "missed_gt_frames": stats["missed_gt_frames"],
            "precision_track": round(precision, 4), "coverage": round(coverage, 4),
            "rmse_minus_lidar": None if stats["rmse"] is None else
                                round(stats["rmse"] - lidar["tracking"]["lidar"]["rmse"], 4),
        })
        print(rows[-1], flush=True)
    out = {"lidar_reference": lidar["tracking"]["lidar"], "gate_chi2_2dof_0995": result["gate"],
           "rows": rows}
    (BONUS / "calibration_study.json").write_text(json.dumps(out, indent=2))
    _plot_calibration(rows, lidar["tracking"]["lidar"]["rmse"])


def _plot_calibration(rows, lidar_rmse) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yaw_rows = [r for r in rows if r["shift_left_m"] == 0]
    yaw = [r["yaw_deg"] for r in yaw_rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(yaw, [r["mean_gamma_u_px"] for r in yaw_rows], "o-", label="mean γ_u (signed)")
    axes[0].plot(yaw, [r["mean_abs_gamma_u_px"] for r in yaw_rows], "s--", label="mean |γ_u|")
    axes[0].set(xlabel="camera yaw error (deg)", ylabel="pixels",
                title="Camera innovation (accepted updates)")
    axes[0].legend()
    axes[1].plot(yaw, [r["gate_reject_rate"] for r in yaw_rows], "o-", color="tab:red")
    axes[1].set(xlabel="camera yaw error (deg)", ylabel="fraction",
                title="Visible confirmed tracks rejected by χ² gate")
    axes[2].plot(yaw, [r["rmse"] for r in yaw_rows], "o-", label="fused RMSE")
    axes[2].axhline(lidar_rmse, color="gray", ls="--", label="LiDAR-only RMSE")
    axes[2].axhline(lidar_rmse + 0.05, color="tab:red", ls=":", label="LiDAR + 0.05 m")
    axes[2].set(xlabel="camera yaw error (deg)", ylabel="m", title="3D position RMSE")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(BONUS / "calibration_drift.png", dpi=110)
    print("wrote calibration_drift.png")


def _project(camera, xyz):
    try:
        state = np.asmatrix(np.r_[xyz, [0, 0, 0]]).T
        if not camera.in_fov(state):
            return None
        return np.asarray(camera.get_hx(state)).reshape(-1)
    except ValueError:
        return None


def _per_track_errors(cache, fused, lidar):
    """Per fused track: 3D error to nearest GT for the fused track and its LiDAR twin."""
    frames = {item["frame"]: item for item in cache["frames"]}
    lidar_by_frame = {s["frame"]: s for s in lidar["history"]}
    out = {}
    for s in fused["history"]:
        t = s["frame"]
        gts = np.array([[l["center_x"], l["center_y"], l["center_z"]]
                        for l in frames[t]["labels"]]).reshape(-1, 3)
        lid = [r for r in lidar_by_frame[t]["after_camera"] if r[2] == "confirmed"]
        for row in s["after_camera"]:
            if row[2] != "confirmed" or not len(gts) or not lid:
                continue
            g = gts[np.linalg.norm(gts[:, :2] - row[1][:2], axis=1).argmin()]
            twin = min(lid, key=lambda r: np.linalg.norm(r[1][:2] - row[1][:2]))
            if np.linalg.norm(g[:2] - row[1][:2]) > 2.0:
                continue
            out.setdefault(row[0], {"frames": [], "fused": [], "lidar": []})
            out[row[0]]["frames"].append(t)
            out[row[0]]["fused"].append(float(np.linalg.norm(row[1] - g)))
            out[row[0]]["lidar"].append(float(np.linalg.norm(twin[1] - g)))
    return out


def visualise(cache) -> None:
    """BEV and FRONT-image figures showing where the camera update moved a track."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fused = replay(cache, "fused", keep_history=True)
    lidar = replay(cache, "lidar", keep_history=True)
    camera = fused["camera"]
    frames = {item["frame"]: item for item in cache["frames"]}

    errors = _per_track_errors(cache, fused, lidar)
    table = []
    for tid, e in sorted(errors.items()):
        row = {"track": tid, "frames": len(e["frames"]),
               "rmse_fused_m": round(float(np.sqrt(np.mean(np.square(e["fused"])))), 4),
               "rmse_lidar_m": round(float(np.sqrt(np.mean(np.square(e["lidar"])))), 4)}
        row["fused_minus_lidar_m"] = round(row["rmse_fused_m"] - row["rmse_lidar_m"], 4)
        table.append(row)
    long_tracks = [r for r in table if r["frames"] >= 20]
    winner = min(long_tracks, key=lambda r: r["fused_minus_lidar_m"])
    loser = max(long_tracks, key=lambda r: r["fused_minus_lidar_m"])
    focus_track = winner["track"]

    # Frame where the camera step most reduced the focus track's XY error to GT.
    best = None
    for step in fused["history"]:
        before = {tid: xyz for tid, xyz, _ in step["after_lidar"]}
        gts = np.array([[l["center_x"], l["center_y"]]
                        for l in frames[step["frame"]]["labels"]]).reshape(-1, 2)
        for row in step["after_camera"]:
            if row[0] == focus_track and row[2] == "confirmed" and focus_track in before and len(gts):
                g = gts[np.linalg.norm(gts - row[1][:2], axis=1).argmin()]
                gain = np.linalg.norm(before[focus_track][:2] - g) - np.linalg.norm(row[1][:2] - g)
                if best is None or gain > best[2]:
                    move = float(np.linalg.norm(row[1][:2] - before[focus_track][:2]))
                    best = (move, step["frame"], gain)
    move, focus_frame, _ = best
    step = next(s for s in fused["history"] if s["frame"] == focus_frame)
    item = frames[focus_frame]
    before = {tid: xyz for tid, xyz, _ in step["after_lidar"]}
    after = {row[0]: row for row in step["after_camera"]}
    lid = np.array(step["lidar_obs"]).reshape(-1, 3)
    focus_after, focus_before = after[focus_track][1], before[focus_track]
    focus_gt = min(item["labels"], key=lambda l: np.hypot(l["center_x"] - focus_after[0],
                                                          l["center_y"] - focus_after[1]))

    # Figure 1: BEV of the frame, plus a zoom on the focus track.
    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(13, 7.5), gridspec_kw={"width_ratios": [1.1, 1]})
    pts = item["bev_points"]
    for axis in (ax, zoom):
        axis.scatter(pts[:, 1], pts[:, 0], s=0.3, c="0.75")
        for l in item["labels"]:
            axis.add_patch(plt.Rectangle((l["center_y"] - l["width"] / 2, l["center_x"] - l["length"] / 2),
                                         l["width"], l["length"], fill=False, ec="tab:green", lw=1.2))
            axis.scatter(l["center_y"], l["center_x"], marker="+", c="tab:green", s=60)
        axis.scatter(lid[:, 1], lid[:, 0], marker="x", c="tab:blue", s=50)
        for tid, row in after.items():
            if row[2] != "confirmed":
                continue
            a = row[1]
            axis.scatter(a[1], a[0], c="tab:red", s=30, zorder=4)
            if tid in before:
                b = before[tid]
                axis.scatter(b[1], b[0], facecolors="none", edgecolors="tab:purple", s=60, zorder=3)
                axis.annotate("", xy=(a[1], a[0]), xytext=(b[1], b[0]),
                              arrowprops={"arrowstyle": "->", "color": "tab:purple", "lw": 1.5})
            axis.annotate(f"T{tid}", (a[1], a[0]), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set(xlim=cache["lim_y"][::-1], ylim=cache["lim_x"], xlabel="y (m, left +)", ylabel="x (m, forward)",
           title=f"BEV frame {focus_frame}")
    cx, cy = focus_after[0], focus_after[1]
    zoom.set(xlim=(cy + 1.6, cy - 1.6), ylim=(cx - 1.6, cx + 1.6), xlabel="y (m)", ylabel="x (m)",
             title=f"Zoom on T{focus_track}: camera moves it {move:.2f} m")
    zoom.set_aspect("equal")
    handles = [plt.Line2D([], [], marker="x", ls="", c="tab:blue", label="LiDAR detection centre"),
               plt.Line2D([], [], marker="o", ls="", mfc="none", mec="tab:purple", label="track after LiDAR update"),
               plt.Line2D([], [], marker="o", ls="", c="tab:red", label="track after camera update"),
               plt.Line2D([], [], marker="+", ls="", c="tab:green", label="GT vehicle centre / box")]
    zoom.legend(handles=handles, loc="lower left", fontsize=8)
    gt_before = float(np.hypot(focus_before[0] - focus_gt["center_x"], focus_before[1] - focus_gt["center_y"]))
    gt_after = float(np.hypot(focus_after[0] - focus_gt["center_x"], focus_after[1] - focus_gt["center_y"]))
    zoom.text(0.02, 0.98, f"XY error to GT: {gt_before:.2f} m -> {gt_after:.2f} m",
              transform=zoom.transAxes, va="top", fontsize=9,
              bbox={"facecolor": "white", "alpha": 0.8})
    fig.tight_layout()
    fig.savefig(BONUS / "viz_bev_camera_update.png", dpi=110)
    plt.close(fig)

    # Figure 2: FRONT image, full frame and a 2x crop around the focus track.
    image = cv2.imdecode(np.frombuffer(item["image"], np.uint8), cv2.IMREAD_COLOR)
    for z in step["camera_obs"] or []:
        cv2.drawMarker(image, (int(z[0]), int(z[1])), (0, 200, 0), cv2.MARKER_CROSS, 24, 2)
    p_focus = None
    for tid, row in after.items():
        if row[2] != "confirmed":
            continue
        p_before = _project(camera, before[tid]) if tid in before else None
        p_after = _project(camera, row[1])
        if p_before is not None:
            cv2.circle(image, tuple(int(v) for v in p_before), 7, (255, 120, 0), 2)
        if p_after is not None:
            cv2.circle(image, tuple(int(v) for v in p_after), 7, (0, 0, 255), 2)
            cv2.putText(image, f"T{tid}", (int(p_after[0]) + 12, int(p_after[1]) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            if tid == focus_track:
                p_focus = p_after
    full = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    if p_focus is not None:
        u, v = int(p_focus[0]), int(p_focus[1])
        x0, y0 = max(0, u - 160), max(0, v - 110)
        crop = cv2.resize(image[y0:y0 + 220, x0:x0 + 320], None, fx=1.5, fy=1.5,
                          interpolation=cv2.INTER_NEAREST)
        h, w = crop.shape[:2]
        full[-h - 10:-10, -w - 10:-10] = crop
        cv2.rectangle(full, (full.shape[1] - w - 10, full.shape[0] - h - 10),
                      (full.shape[1] - 10, full.shape[0] - 10), (255, 255, 255), 2)
    legend = ["green +: camera measurement (GT 2D centre + seeded noise)",
              "blue o: track projected after LiDAR update",
              "red o: track projected after camera update",
              f"inset: 3x zoom (vs. full view) around T{focus_track}, frame {focus_frame}"]
    panel = full.copy()
    cv2.rectangle(panel, (0, 0), (640, 22 + 26 * len(legend)), (30, 30, 30), -1)
    full = cv2.addWeighted(panel, 0.65, full, 0.35, 0)
    for k, text in enumerate(legend):
        cv2.putText(full, text, (12, 26 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
    cv2.imwrite(str(BONUS / "viz_front_camera_update.jpg"), full, [cv2.IMWRITE_JPEG_QUALITY, 88])

    # Figure 3: per-frame error, the most-helped track and the most-hurt track.
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for axis, row in zip(axes, (winner, loser)):
        e = errors[row["track"]]
        axis.plot(e["frames"], e["lidar"], color="gray", label="LiDAR-only")
        axis.plot(e["frames"], e["fused"], color="tab:red", label="LiDAR + camera")
        title = "T{}: RMSE LiDAR {:.3f} m -> fused {:.3f} m".format(
            row["track"], row["rmse_lidar_m"], row["rmse_fused_m"])
        axis.set(ylabel="3D error (m)", title=title)
        axis.legend(loc="upper right")
    axes[1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(BONUS / "viz_track_error_over_time.png", dpi=110)
    plt.close(fig)

    summary = {"focus_frame": focus_frame, "focus_track": focus_track,
               "camera_correction_m": round(move, 4),
               "focus_xy_error_before_after_m": [round(gt_before, 4), round(gt_after, 4)],
               "per_track": table}
    (BONUS / "viz_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cvat_export(cache) -> None:
    """Export fused confirmed tracks as JSON (platform hook) and CVAT-for-video XML.

    CVAT boxes live in BEV image pixels (608 x 608, forward = up), so the BEV PNGs
    written to data/bonus_cache/bev_frames can be uploaded as the CVAT task images.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from xml.sax.saxutils import escape

    from fusion_lab.evaluation import TRACK_GATE_METERS, _partial_assignment
    from fusion_lab.export_cvat import export_tracks_json

    fused = replay(cache, "fused", keep_history=True)
    frames = {item["frame"]: item for item in cache["frames"]}
    size = 608
    lim_x, lim_y = cache["lim_x"], cache["lim_y"]

    def to_px(x, y):
        col = (lim_y[1] - y) / (lim_y[1] - lim_y[0]) * size
        row = (lim_x[1] - x) / (lim_x[1] - lim_x[0]) * size
        return col, row

    per_frame, events, owner = [], [], {}
    tracks_xml: dict[int, list[str]] = {}
    for idx, step in enumerate(fused["history"]):
        t = step["frame"]
        confirmed = [row for row in step["after_camera"] if row[2] == "confirmed"]
        labels = frames[t]["labels"]
        per_frame.append({"frame": t, "tracks": [
            {"id": row[0], "x": float(row[1][0]), "y": float(row[1][1]), "z": float(row[1][2]),
             "length": row[3], "width": row[4], "height": row[5], "yaw": row[6],
             "score": row[7], "state": row[2]} for row in step["after_camera"]]})
        # Same 2 m one-to-one gate as the grader: ghosts and GT->track ID switches.
        pos = np.array([row[1][:2] for row in confirmed]).reshape(-1, 2)
        gts = np.array([[l["center_x"], l["center_y"]] for l in labels]).reshape(-1, 2)
        dist = np.linalg.norm(pos[:, None, :] - gts[None, :, :], axis=2)
        pairs = _partial_assignment(dist, np.isfinite(dist) & (dist <= TRACK_GATE_METERS))
        matched = {i for i, _ in pairs}
        for i, row in enumerate(confirmed):
            if i not in matched:
                events.append({"frame": t, "type": "ghost", "track": row[0]})
        for i, j in pairs:
            gid, tid = labels[j]["id"], confirmed[i][0]
            if gid in owner and owner[gid] != tid:
                events.append({"frame": t, "type": "id_switch", "gt": gid,
                               "from_track": owner[gid], "to_track": tid})
            owner[gid] = tid
        # Unconfirmed tracks far from every GT vehicle are would-be ghosts that the
        # LiDAR confirmation rule keeps out of the graded (confirmed) set.
        for row in step["after_camera"]:
            if row[2] == "confirmed":
                continue
            gap = (np.linalg.norm(gts - row[1][:2], axis=1).min() if len(gts) else np.inf)
            if gap > TRACK_GATE_METERS:
                events.append({"frame": t, "type": "suppressed_ghost", "track": row[0],
                               "state": row[2], "score": round(float(row[7]), 3),
                               "nearest_gt_m": None if not np.isfinite(gap) else round(float(gap), 2)})
        for row in step["after_camera"]:
            x, y = row[1][0], row[1][1]
            half_l, half_w = row[3] / 2, row[4] / 2
            c0, r0 = to_px(x + half_l, y + half_w)
            c1, r1 = to_px(x - half_l, y - half_w)
            tracks_xml.setdefault(row[0], []).append(
                f'      <box frame="{idx}" keyframe="1" outside="0" occluded="0" '
                f'xtl="{min(c0, c1):.2f}" ytl="{min(r0, r1):.2f}" '
                f'xbr="{max(c0, c1):.2f}" ybr="{max(r0, r1):.2f}" z_order="0">'
                f'<attribute name="state">{row[2]}</attribute></box>')

    export_tracks_json(per_frame, BONUS / "cvat_tracks.json")
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>",
             "  <meta><task><name>day23-fused-tracks-bev</name>"
             f"<size>{len(fused['history'])}</size><mode>interpolation</mode>"
             "<labels><label><name>vehicle</name><attributes><attribute><name>state</name>"
             "<mutable>True</mutable><input_type>select</input_type><default_value>initialized</default_value>"
             "<values>initialized\ntentative\nconfirmed</values></attribute></attributes></label></labels>"
             "</task></meta>"]
    for tid, boxes in sorted(tracks_xml.items()):
        lines.append(f'  <track id="{tid}" label="vehicle" source="manual">')
        lines.extend(boxes)
        # Close the track one frame after its last box so CVAT does not extend it.
        last = int(boxes[-1].split('frame="')[1].split('"')[0])
        if last + 1 < len(fused["history"]):
            lines.append(boxes[-1].replace(f'frame="{last}"', f'frame="{last + 1}"')
                         .replace('outside="0"', 'outside="1"'))
        lines.append("  </track>")
    lines.append("</annotations>")
    (BONUS / "cvat_tracks_bev.xml").write_text("\n".join(lines), encoding="utf-8")
    (BONUS / "cvat_events.json").write_text(json.dumps(events, indent=2))

    # BEV frames for the CVAT task (kept outside git) plus one committed example
    # frame per event type with the ghost / switched track highlighted.
    out_dir = CACHE.parent / "bev_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    highlight = {}
    for kind in ("ghost", "id_switch", "suppressed_ghost"):
        found = next((e for e in events if e["type"] == kind), None)
        if found and found["frame"] not in highlight:
            highlight[found["frame"]] = found
    for idx, step in enumerate(fused["history"]):
        t = step["frame"]
        fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        pts = frames[t]["bev_points"]
        cols, rows = to_px(pts[:, 0], pts[:, 1])
        ax.scatter(cols, rows, s=0.15, c="k")
        ax.set_xlim(0, size); ax.set_ylim(size, 0); ax.axis("off")
        if t in highlight:
            event = highlight[t]
            for l in frames[t]["labels"]:
                c, r = to_px(l["center_x"], l["center_y"])
                ax.add_patch(plt.Circle((c, r), 9, fill=False, ec="tab:green", lw=1.5))
            for row in step["after_camera"]:
                c, r = to_px(row[1][0], row[1][1])
                hot = row[0] in (event.get("track"), event.get("to_track"))
                colour = "tab:red" if hot else ("tab:blue" if row[2] == "confirmed" else "tab:orange")
                ax.add_patch(plt.Rectangle((c - 6, r - 12), 12, 24, fill=False, ec=colour,
                                           lw=2 if hot else 1, ls="-" if row[2] == "confirmed" else "--"))
                ax.text(c + 8, r, f"T{row[0]} {row[2]} s={row[7]:.2f}", color=colour, fontsize=7)
            detail = {k: v for k, v in event.items() if k not in ("frame", "type")}
            ax.text(5, 15, f"frame {t}: {event['type']}  {json.dumps(detail)}", color="tab:red", fontsize=6)
            ax.text(5, 30, "green circle = GT vehicle centre; solid box = confirmed, dashed = unconfirmed",
                    fontsize=6)
            fig.savefig(BONUS / f"cvat_{event['type']}_frame{t:03d}.png", dpi=100)
        fig.savefig(out_dir / f"frame_{idx:06d}.png", dpi=100)
        plt.close(fig)
    summary = {
        "tracks_exported": len(tracks_xml), "frames": len(fused["history"]),
        "ghost_events": sum(e["type"] == "ghost" for e in events),
        "id_switch_events": sum(e["type"] == "id_switch" for e in events),
        "suppressed_ghost_events": sum(e["type"] == "suppressed_ghost" for e in events),
        "suppressed_ghost_tracks": sorted({e["track"] for e in events if e["type"] == "suppressed_ghost"}),
        "highlighted": highlight,
    }
    (BONUS / "cvat_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", choices=["cache", "check", "calibration", "visualise", "cvat", "all"])
    parser.add_argument("--config", type=Path, default=REPO / "student" / "config" / "paths.yaml")
    args = parser.parse_args()
    if args.task == "cache":
        build_cache(args.config)
        return
    cache = load_cache()
    if args.task in ("check", "all"):
        check_reproduces_artifacts(cache)
    if args.task in ("calibration", "all"):
        calibration_study(cache)
    if args.task in ("visualise", "all"):
        visualise(cache)
    if args.task in ("cvat", "all"):
        cvat_export(cache)


if __name__ == "__main__":
    main()
