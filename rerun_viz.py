"""
Rerun-based visualization for CS-OnFlyGS, styled after MASt3R-SLAM's RerunVisualizer.

Provides:
- 3D Gaussian positions as point cloud (colored by SH DC -> RGB)
- Camera frustums drawn as line strips (version-proof, no Pinhole dependency)
- Camera trajectory as connected line strip
- Camera positions as colored spheres
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
        frustum_scale: float = 0.1,
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
        self.frustum_scale = frustum_scale
        self.logged_kf_count = 0
        self._debug_count = 0

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

        # Coordinate system: OpenCV convention (X-right, Y-down, Z-forward)
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

        # --- Latest keyframe image ---
        if n_kf > 0:
            try:
                last_kf = keyframes[-1]
                self._log_image("/images/rendered", last_kf.image_pyr[0])
            except Exception as e:
                print(f"[RerunViz] rendered image error: {e}")

        # --- Camera frustums + trajectory + positions (all in one) ---
        try:
            self._log_cameras(keyframes, n_kf)
        except Exception as e:
            print(f"[RerunViz] camera/trajectory error: {e}")
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

        # Debug: print data ranges on first few updates
        self._debug_count += 1
        if self._debug_count <= 3:
            self._print_debug(keyframes, sm)

    # ------------------------------------------------------------------ #
    #  Image logging                                                      #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Camera frustums, positions, trajectory                             #
    # ------------------------------------------------------------------ #

    def _get_cam_world(self, keyframe):
        """Return (position_world[3], R_c2w[3,3]) from a keyframe."""
        Rt = keyframe.get_Rt().detach().cpu().numpy().astype(np.float64)
        R_w2c = Rt[:3, :3]
        t_w2c = Rt[:3, 3]
        R_c2w = R_w2c.T
        pos = -R_c2w @ t_w2c
        return pos.astype(np.float32), R_c2w.astype(np.float32)

    def _make_frustum_lines(self, pos, R_c2w, w, h, focal, scale=None):
        """Build 8 line segments forming a camera frustum wireframe.
        Returns list of [start, end] pairs in world coordinates."""
        if scale is None:
            scale = self.frustum_scale
        # Half-sizes at unit depth
        hw = (w / 2.0) / focal * scale
        hh = (h / 2.0) / focal * scale
        d = scale  # depth of frustum tip

        # Four corners of the image plane in camera coords (OpenCV: X-right, Y-down, Z-forward)
        corners_cam = np.array([
            [-hw, -hh, d],
            [ hw, -hh, d],
            [ hw,  hh, d],
            [-hw,  hh, d],
        ], dtype=np.float32)

        # Transform to world
        corners_w = (R_c2w @ corners_cam.T).T + pos

        # 8 line segments: 4 edges of the rectangle + 4 from origin to corners
        lines = []
        for i in range(4):
            lines.append([pos.tolist(), corners_w[i].tolist()])             # origin -> corner
            lines.append([corners_w[i].tolist(), corners_w[(i+1)%4].tolist()])  # rectangle edge
        return lines

    def _log_cameras(self, keyframes, n_kf):
        """Log all camera frustums, positions, and trajectory."""
        if n_kf == 0:
            return

        centres = []
        all_frustum_lines = []
        current_frustum_lines = []

        for kf_idx, kf in enumerate(keyframes):
            pos, R_c2w = self._get_cam_world(kf)
            centres.append(pos)

            w = float(kf.width)
            h = float(kf.height)
            f_val = kf.f.item() if isinstance(kf.f, torch.Tensor) else float(kf.f)

            lines = self._make_frustum_lines(pos, R_c2w, w, h, f_val)

            if kf_idx < n_kf - 1:
                all_frustum_lines.extend(lines)
            else:
                current_frustum_lines = lines

        centres_np = np.array(centres, dtype=np.float32)

        # Keyframe frustums (red)
        if all_frustum_lines:
            rr.log(
                "/world/kf_frustums",
                rr.LineStrips3D(
                    all_frustum_lines,
                    colors=[255, 80, 80],
                    radii=0.002,
                ),
            )

        # Current camera frustum (green, thicker)
        if current_frustum_lines:
            rr.log(
                "/world/current_frustum",
                rr.LineStrips3D(
                    current_frustum_lines,
                    colors=[0, 255, 0],
                    radii=0.004,
                ),
            )

        # Keyframe positions as red spheres
        rr.log(
            "/world/kf_positions",
            rr.Points3D(
                positions=centres_np,
                colors=np.full((len(centres_np), 3), [255, 80, 80], dtype=np.uint8),
                radii=0.015,
            ),
        )

        # Current camera as green sphere
        rr.log(
            "/world/current_pos",
            rr.Points3D(
                positions=[centres_np[-1]],
                colors=[[0, 255, 0]],
                radii=0.03,
            ),
        )

        # Trajectory line
        if len(centres_np) >= 2:
            rr.log(
                "/world/trajectory",
                rr.LineStrips3D(
                    [centres_np.tolist()],
                    colors=[0, 200, 255],
                    radii=0.003,
                ),
            )

    # ------------------------------------------------------------------ #
    #  Gaussian point cloud                                               #
    # ------------------------------------------------------------------ #

    def _log_gaussians(self):
        """Log active Gaussian positions as a point cloud, colored by SH DC."""
        sm = self.scene_model
        n = sm.n_active_gaussians
        if n == 0:
            return

        with torch.no_grad():
            xyz = sm.xyz.detach().cpu().numpy()
            f_dc = sm.f_dc.detach().cpu().numpy().reshape(-1, 3)
            # sm.opacity already applies sigmoid, so values are in [0,1]
            opacity = sm.opacity.detach().cpu().numpy().reshape(-1)

        if xyz.shape[0] != f_dc.shape[0] or xyz.shape[0] != opacity.shape[0]:
            print(f"[RerunViz] Shape mismatch: xyz={xyz.shape}, f_dc={f_dc.shape}, opacity={opacity.shape}")
            return

        # SH DC to color: C = SH_C0 * f_dc + 0.5
        SH_C0 = 0.28209479177387814
        colors = (f_dc * SH_C0 + 0.5).clip(0, 1)
        colors = (colors * 255).astype(np.uint8)

        # Filter by opacity
        valid = opacity > 0.05
        n_valid = int(valid.sum())
        if n_valid == 0:
            print(f"[RerunViz] No Gaussians above opacity 0.05 (n={n}), logging all")
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

        # Remove NaN/Inf
        finite_mask = np.isfinite(xyz).all(axis=1)
        if not finite_mask.all():
            xyz = xyz[finite_mask]
            colors = colors[finite_mask]
            if len(xyz) == 0:
                return

        rr.log(
            "/world/gaussians",
            rr.Points3D(
                positions=xyz,
                colors=colors,
                radii=0.005,
            ),
        )

    # ------------------------------------------------------------------ #
    #  Debug                                                              #
    # ------------------------------------------------------------------ #

    def _print_debug(self, keyframes, sm):
        """Print diagnostic info for the first few frames."""
        n_kf = len(keyframes)
        n_gs = sm.n_active_gaussians
        print(f"[RerunViz DEBUG] n_keyframes={n_kf}, n_gaussians={n_gs}")

        if n_kf > 0:
            for i, kf in enumerate(keyframes[:3]):
                pos, _ = self._get_cam_world(kf)
                print(f"  kf[{i}] pos={pos}, f={kf.f}, w={kf.width}, h={kf.height}")

        if n_gs > 0:
            with torch.no_grad():
                xyz = sm.xyz.detach().cpu().numpy()
                print(f"  gaussians xyz range: min={xyz.min(axis=0)}, max={xyz.max(axis=0)}")
                print(f"  gaussians xyz mean: {xyz.mean(axis=0)}")
