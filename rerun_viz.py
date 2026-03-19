"""
Rerun-based visualization for CS-OnFlyGS, styled after MASt3R-SLAM's RerunVisualizer.

Provides:
- 3D Gaussian positions as point cloud (colored by SH DC → RGB)
- Camera frustums: current frame (green) + keyframes (red)
- Camera trajectory as connected line strip
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
        C_conf_threshold: float = 0.1,
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
        self.C_conf_threshold = C_conf_threshold
        self.logged_kf_count = 0

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
            print(f"Rerun recording will be saved to: {save_path}")

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
        """Update visualization. Call each frame from the main loop."""
        try:
            self._update_impl(frame_idx, current_image, metrics)
        except Exception as e:
            print(f"[RerunViz] update error at frame {frame_idx}: {e}")
            import traceback

            traceback.print_exc()

    def _update_impl(self, frame_idx: int, current_image, metrics):
        rr.set_time("frame", sequence=frame_idx)

        sm = self.scene_model
        keyframes = sm.keyframes

        # --- Current frame image ---
        if current_image is not None:
            self._log_image("/images/current", current_image)

        # --- Keyframe cameras ---
        n_kf = len(keyframes)
        for kf_idx in range(self.logged_kf_count, n_kf):
            kf = keyframes[kf_idx]
            self._log_keyframe_camera(kf_idx, kf)
        self.logged_kf_count = n_kf

        # --- Latest keyframe image ---
        if n_kf > 0:
            last_kf = keyframes[-1]
            self._log_image("/images/rendered", last_kf.image_pyr[0])

        # --- Camera trajectory ---
        self._log_trajectory(keyframes)

        # --- Gaussian point cloud ---
        self._log_gaussians()

        # --- Scalar metrics ---
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    rr.log(f"/metrics/{key}", rr.Scalar(value))

        # Always log counts
        rr.log("/metrics/n_gaussians", rr.Scalar(sm.n_active_gaussians))
        rr.log("/metrics/n_keyframes", rr.Scalar(n_kf))

    def _log_image(self, entity: str, image_tensor):
        """Log an image tensor (C,H,W float [0,1] or H,W,C uint8)."""
        if isinstance(image_tensor, torch.Tensor):
            img = image_tensor.detach().cpu()
            if img.dim() == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)  # C,H,W → H,W,C
            img = (img.float().clamp(0, 1) * 255).byte().numpy()
        else:
            img = image_tensor
        rr.log(entity, rr.Image(img))

    def _log_keyframe_camera(self, kf_idx: int, keyframe):
        """Log keyframe camera frustum in world space."""
        Rt = keyframe.get_Rt().detach().cpu()
        R = Rt[:3, :3].numpy()  # world-to-camera rotation
        t = Rt[:3, 3].numpy()  # world-to-camera translation

        # World-to-camera → camera position in world = -R^T @ t
        # Rerun Transform3D expects parent-from-child (world-from-camera)
        R_inv = R.T
        t_world = -R.T @ t

        entity = f"/world/keyframes/kf_{kf_idx}"
        rr.log(
            entity,
            rr.Transform3D(
                translation=t_world,
                mat3x3=R_inv,
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
                image_plane_distance=0.03,
                color=[255, 80, 80],
            ),
        )

    def _log_current_camera(self, keyframe):
        """Log the most recent camera as 'current' (green frustum)."""
        Rt = keyframe.get_Rt().detach().cpu()
        R = Rt[:3, :3].numpy()
        t = Rt[:3, 3].numpy()
        R_inv = R.T
        t_world = -R.T @ t

        rr.log(
            "/world/current_camera",
            rr.Transform3D(
                translation=t_world,
                mat3x3=R_inv,
            ),
        )

        w = keyframe.width
        h = keyframe.height
        focal = keyframe.f.item() if isinstance(keyframe.f, torch.Tensor) else keyframe.f
        rr.log(
            "/world/current_camera/pinhole",
            rr.Pinhole(
                focal_length=focal,
                width=w,
                height=h,
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.05,
                color=[0, 255, 0],
            ),
        )

    def _log_trajectory(self, keyframes):
        """Log camera centres as a connected trajectory line strip."""
        if len(keyframes) < 2:
            return

        centres = []
        for kf in keyframes:
            c = kf.get_centre(approx=True)
            if isinstance(c, torch.Tensor):
                c = c.detach().cpu().numpy()
            centres.append(c.tolist())

        # Also update current camera to latest keyframe
        self._log_current_camera(keyframes[-1])

        rr.log(
            "/world/trajectory",
            rr.LineStrips3D(
                [centres],
                colors=[0, 255, 0],
                radii=0.002,
            ),
        )

    def _log_gaussians(self):
        """Log active Gaussian positions as a point cloud, colored by SH DC."""
        sm = self.scene_model
        n = sm.n_active_gaussians
        if n == 0:
            return

        with torch.no_grad():
            xyz = sm.xyz.detach().cpu().numpy()
            # Extract RGB from SH DC coefficient (band 0)
            f_dc = sm.f_dc.detach().cpu().numpy().reshape(-1, 3)
            # SH DC to color: C = SH_C0 * f_dc + 0.5
            SH_C0 = 0.28209479177387814
            colors = (f_dc * SH_C0 + 0.5).clip(0, 1)
            colors = (colors * 255).astype(np.uint8)

            # Subsample if too many
            if n > self.max_points:
                step = n / self.max_points
                idx = (np.arange(self.max_points) * step).astype(np.int64)
                xyz = xyz[idx]
                colors = colors[idx]

            # Filter by opacity
            opacity = sm.opacity.detach().cpu().numpy().reshape(-1)
            valid = opacity > self.C_conf_threshold
            if valid.sum() == 0:
                return
            xyz = xyz[valid]
            colors = colors[valid]

        rr.log(
            "/world/gaussians",
            rr.Points3D(
                positions=xyz,
                colors=colors,
                radii=0.003,
            ),
        )
