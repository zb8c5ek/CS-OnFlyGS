"""
Rerun-based visualization for CS-OnFlyGS, styled after MASt3R-SLAM's RerunVisualizer.

Provides:
- 3D Gaussian positions as point cloud (colored by SH DC -> RGB)
- Camera frustums: current frame (green) + keyframes (red)
- Camera trajectory as connected line strip
- Camera positions as colored spheres (always visible fallback)
- Current frame image + latest keyframe image panels
- Scalar metrics: PSNR, #Gaussians, #Keyframes, focal length
- Temporal scrubbing via frame sequence index
- Optional .rrd file recording for offline review

Usage:
    from rerun_viz import RerunVisualizer
    viz = RerunVisualizer(scene_model, save_path="logs/run.rrd")
    viz.update(frame_idx, image, metrics)
"""

import numpy as np
import torch

try:
    import rerun as rr
    import rerun.blueprint as rrb

    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False


class RerunVisualizer:
    """Rerun-based OnFlyGS visualizer, M3R-SLAM style."""

    def __init__(
        self,
        scene_model,
        max_points: int = 200_000,
        app_id: str = "OnFlyGS-SLAM",
        save_path: str | None = None,
    ):
        if not HAS_RERUN:
            raise ImportError(
                "rerun-sdk is required for visualization. "
                "Install with: pip install rerun-sdk"
            )

        self.scene_model = scene_model
        self.max_points = max_points
        self.logged_kf_count = 0
        self._first_update = True

        # Kill stale Rerun viewers
        import subprocess, os, time

        if os.name == "nt":
            subprocess.run(["taskkill", "/f", "/im", "rerun.exe"], capture_output=True)
        else:
            subprocess.run(["pkill", "-f", "rerun"], capture_output=True)
        time.sleep(1)

        # Init Rerun
        rr.init(app_id)
        rr.spawn(connect=True)

        if save_path is not None:
            import pathlib

            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            rec = rr.get_global_data_recording()
            rec.set_sinks(
                rr.GrpcSink(),
                rr.FileSink(save_path),
            )
            print(f"[RerunViz] Recording will be saved to: {save_path}")

        # Coordinate system
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

        # Blueprint: 3D scene + image panels (M3R-SLAM layout)
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="3D Scene",
                    origin="/world",
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(
                        name="Current Frame",
                        origin="/images/current",
                    ),
                    rrb.Spatial2DView(
                        name="Rendered View",
                        origin="/images/rendered",
                    ),
                    rrb.TimeSeriesView(
                        name="Metrics",
                        origin="/metrics",
                    ),
                ),
                column_shares=[3, 1],
            ),
        )
        rr.send_blueprint(blueprint)

    def update(self, frame_idx: int, current_image=None, metrics: dict = None):
        """Update visualization. Call each frame from the main loop.
        Each section is isolated so one failure doesn't block the rest."""
        rr.set_time("frame", sequence=frame_idx)

        sm = self.scene_model
        keyframes = sm.keyframes
        n_kf = len(keyframes)

        # --- Current frame image ---
        if current_image is not None:
            try:
                self._log_image("/images/current", current_image)
            except Exception as e:
                print(f"[RerunViz] current image error: {e}")

        # --- Keyframe cameras (new ones only) ---
        try:
            for kf_idx in range(self.logged_kf_count, n_kf):
                kf = keyframes[kf_idx]
                self._log_keyframe_camera(kf_idx, kf)
            self.logged_kf_count = n_kf
        except Exception as e:
            print(f"[RerunViz] keyframe camera error: {e}")
            import traceback; traceback.print_exc()

        # --- Latest keyframe image ---
        if n_kf > 0:
            try:
                last_kf = keyframes[-1]
                self._log_image("/images/rendered", last_kf.image_pyr[0])
            except Exception as e:
                print(f"[RerunViz] rendered image error: {e}")

        # --- Camera positions as spheres + trajectory ---
        try:
            self._log_camera_positions(keyframes)
        except Exception as e:
            print(f"[RerunViz] trajectory error: {e}")
            import traceback; traceback.print_exc()

        # --- Gaussian point cloud ---
        try:
            self._log_gaussians()
        except Exception as e:
            print(f"[RerunViz] gaussians error: {e}")
            import traceback; traceback.print_exc()

        # --- Scalar metrics ---
        try:
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        rr.log(f"/metrics/{key}", rr.Scalar(value))
            rr.log("/metrics/n_gaussians", rr.Scalar(sm.n_active_gaussians))
            rr.log("/metrics/n_keyframes", rr.Scalar(n_kf))
        except Exception as e:
            print(f"[RerunViz] metrics error: {e}")

        # --- Debug info on first successful update ---
        if self._first_update:
            self._first_update = False
            print(f"[RerunViz] First update: frame={frame_idx}, "
                  f"n_keyframes={n_kf}, n_gaussians={sm.n_active_gaussians}")

    def _log_image(self, entity: str, image_tensor):
        """Log an image tensor (C,H,W float [0,1] or H,W,C uint8)."""
        if isinstance(image_tensor, torch.Tensor):
            img = image_tensor.detach().cpu()
            if img.dim() == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)  # C,H,W -> H,W,C
            img = (img.float().clamp(0, 1) * 255).byte().numpy()
        else:
            img = image_tensor
        rr.log(entity, rr.Image(img))

    def _get_cam_world_from_kf(self, keyframe):
        """Extract camera-in-world transform from keyframe.
        Returns (t_world, R_cam2world) as numpy arrays."""
        Rt = keyframe.get_Rt().detach().cpu().numpy()
        R_w2c = Rt[:3, :3]  # world-to-camera rotation
        t_w2c = Rt[:3, 3]   # world-to-camera translation
        # Invert: camera-to-world
        R_c2w = R_w2c.T
        t_world = -R_w2c.T @ t_w2c
        return t_world, R_c2w

    def _log_keyframe_camera(self, kf_idx: int, keyframe):
        """Log keyframe camera frustum in world space."""
        t_world, R_c2w = self._get_cam_world_from_kf(keyframe)

        entity = f"/world/keyframes/kf_{kf_idx}"
        rr.log(
            entity,
            rr.Transform3D(
                translation=t_world,
                mat3x3=R_c2w,
            ),
        )

        w = keyframe.width
        h = keyframe.height
        focal = keyframe.f.item() if isinstance(keyframe.f, torch.Tensor) else keyframe.f
        rr.log(
            f"{entity}/pinhole",
            rr.Pinhole(
                focal_length=focal,
                width=w,
                height=h,
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.15,
            ),
        )

    def _log_camera_positions(self, keyframes):
        """Log camera centres as points and a connected trajectory line strip."""
        if len(keyframes) == 0:
            return

        centres = []
        for kf in keyframes:
            c = kf.get_centre(approx=True)
            if isinstance(c, torch.Tensor):
                c = c.detach().cpu().numpy().flatten()
            centres.append(c)

        centres_np = np.array(centres, dtype=np.float32)

        # Log keyframe positions as red spheres (always visible in 3D)
        rr.log(
            "/world/kf_positions",
            rr.Points3D(
                positions=centres_np,
                colors=np.full((len(centres_np), 3), [255, 80, 80], dtype=np.uint8),
                radii=0.02,
            ),
        )

        # Current camera = last keyframe, green sphere
        rr.log(
            "/world/current_pos",
            rr.Points3D(
                positions=[centres_np[-1]],
                colors=[[0, 255, 0]],
                radii=0.04,
            ),
        )

        # Trajectory line
        if len(centres_np) >= 2:
            rr.log(
                "/world/trajectory",
                rr.LineStrips3D(
                    [centres_np.tolist()],
                    colors=[0, 255, 0],
                    radii=0.005,
                ),
            )

        # Also log current camera frustum
        last_kf = keyframes[-1]
        t_world, R_c2w = self._get_cam_world_from_kf(last_kf)
        rr.log(
            "/world/current_camera",
            rr.Transform3D(
                translation=t_world,
                mat3x3=R_c2w,
            ),
        )
        w = last_kf.width
        h = last_kf.height
        focal = last_kf.f.item() if isinstance(last_kf.f, torch.Tensor) else last_kf.f
        rr.log(
            "/world/current_camera/pinhole",
            rr.Pinhole(
                focal_length=focal,
                width=w,
                height=h,
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.2,
            ),
        )

    def _log_gaussians(self):
        """Log active Gaussian positions as a point cloud, colored by SH DC."""
        sm = self.scene_model
        n = sm.n_active_gaussians
        if n == 0:
            return

        with torch.no_grad():
            # Snapshot tensors (no lock needed — we tolerate stale reads)
            xyz = sm.xyz.detach().clone().cpu().numpy()
            f_dc = sm.f_dc.detach().clone().cpu().numpy().reshape(-1, 3)
            opacity = sm.opacity.detach().clone().cpu().numpy().reshape(-1)

        # Sanity check dimensions match
        if xyz.shape[0] != f_dc.shape[0] or xyz.shape[0] != opacity.shape[0]:
            print(f"[RerunViz] Shape mismatch: xyz={xyz.shape}, f_dc={f_dc.shape}, opacity={opacity.shape}")
            return

        # SH DC to color: C = SH_C0 * f_dc + 0.5
        SH_C0 = 0.28209479177387814
        colors = (f_dc * SH_C0 + 0.5).clip(0, 1)
        colors = (colors * 255).astype(np.uint8)

        # Filter by opacity (> 0.05 to include most visible Gaussians)
        valid = opacity > 0.05
        n_valid = int(valid.sum())
        if n_valid == 0:
            # Fallback: log all points if none pass opacity filter
            print(f"[RerunViz] No Gaussians above opacity threshold, logging all {n}")
            pass  # skip filtering
        else:
            xyz = xyz[valid]
            colors = colors[valid]
            n = n_valid

        # Subsample if too many
        if n > self.max_points:
            step = n / self.max_points
            idx = (np.arange(self.max_points) * step).astype(np.int64)
            xyz = xyz[idx]
            colors = colors[idx]

        # Check for NaN/Inf
        finite_mask = np.isfinite(xyz).all(axis=1)
        if not finite_mask.all():
            xyz = xyz[finite_mask]
            colors = colors[finite_mask]
            if len(xyz) == 0:
                print("[RerunViz] All Gaussian positions are NaN/Inf")
                return

        rr.log(
            "/world/gaussians",
            rr.Points3D(
                positions=xyz,
                colors=colors,
                radii=0.005,
            ),
        )
