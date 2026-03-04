from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless rendering (rgb_array / video frames)
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# ---------- Utility: color helpers ----------

def _rgb(r, g, b) -> np.ndarray:
    return np.array([r, g, b], dtype=np.float32) / 255.0

# Colors requested
COLOR_OBSTACLE   = _rgb(0, 0, 0)         # black
COLOR_BG_FREE    = _rgb(46, 101, 279)      # blue
COLOR_COVERED    = _rgb(255, 255, 255)   # white
COLOR_SIGHT      = _rgb(158, 232, 255)   # light blue
COLOR_DRONE      = _rgb(255, 0, 0)       # red
COLOR_CAR        = _rgb(255, 220, 0)     # yellow


def _alpha_blend(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    """
    base, overlay: float32 in [0,1], shape (...,3)
    """
    return base * (1.0 - alpha) + overlay * alpha


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def _square_slice(cx: int, cy: int, radius: int, H: int, W: int) -> Tuple[slice, slice]:
    """
    Return y-slice and x-slice for a square window centered at (cx,cy) with radius.
    Uses standard array indexing: [y, x]
    """
    x0 = _clamp_int(cx - radius, 0, W - 1)
    x1 = _clamp_int(cx + radius, 0, W - 1)
    y0 = _clamp_int(cy - radius, 0, H - 1)
    y1 = _clamp_int(cy + radius, 0, H - 1)
    return slice(y0, y1 + 1), slice(x0, x1 + 1)



@dataclass
class RenderConfig:
    cell_px: int = 8                 # pixel size per cell in output frame
    sight_alpha: float = 0.35        # blending for sight overlay
    title_fontsize: int = 10
    agent_marker_scale: float = 1.2  # marker size relative to cell_px


class MatplotlibGridRenderer:
    """
    Persistent renderer that produces rgb frames quickly.
    Call render_frame(...) each step to get an np.uint8 RGB image.
    """

    def __init__(self, height: int, width: int, cfg: RenderConfig | None = None):
        self.H = int(height)
        self.W = int(width)
        self.cfg = cfg or RenderConfig()

        dpi = 100
        fig_w_px = max(200, self.W * self.cfg.cell_px)
        fig_h_px = max(200, self.H * self.cfg.cell_px + 50)  # extra for title text
        fig_w_in = fig_w_px / dpi
        fig_h_in = fig_h_px / dpi

        self.fig = Figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)

        # One axes for the grid
        self.ax = self.fig.add_axes([0.02, 0.02, 0.96, 0.90])  # leave top space
        self.ax.set_axis_off()

        # Prepare the image artist (updated each frame)
        # We'll draw RGB array directly via imshow
        empty = np.zeros((self.H, self.W, 3), dtype=np.float32)
        self.im = self.ax.imshow(
            empty,
            origin="lower",           # so (0,0) appears bottom-left
            interpolation="nearest"
        )

        # Scatter artists for agents (updated each frame)
        self.drone_scatter = self.ax.scatter([], [], s=1, c=[COLOR_DRONE], marker="s")
        self.car_scatter   = self.ax.scatter([], [], s=1, c=[COLOR_CAR], marker="s")

    def render_frame(
        self,
        obstacle_mask: np.ndarray,
        coverage: np.ndarray,
        agent_state: Dict[str, Any],
        *,
        step_reward: Optional[float] = None,
        infos: Optional[Dict[str, Any]] = None,
        drone_fov: int = 21,
        car_fov: int = 7,
    ) -> np.ndarray:
        """
        Args:
            obstacle_mask: (H,W) 0 free / 1 blocked
            coverage:      (H,W) 0..1 (treated as covered if >0)
            agent_state:   dict agent_id -> object with fields:
                           .type ('drone'/'car'), .x, .y, .battery, .is_active
            step_reward:   current reward (optional)
            infos:         info dict (optional)
            drone_fov/car_fov: window sizes (odd); 21 means radius 10
        Returns:
            rgb frame (uint8) suitable for ffmpeg
        """
        # Defensive: ensure shapes
        obstacle_mask = np.asarray(obstacle_mask)
        coverage = np.asarray(coverage)
        assert obstacle_mask.shape == (self.H, self.W)
        assert coverage.shape == (self.H, self.W)

        # Base layer: free background blue
        img = np.zeros((self.H, self.W, 3), dtype=np.float32)
        img[:] = COLOR_BG_FREE

        # Obstacles: black (on top of background)
        obs = (obstacle_mask == 1)
        img[obs] = COLOR_OBSTACLE

        # Covered: white (only on non-obstacle cells)
        cov = (coverage > 0.0) & (~obs)
        img[cov] = COLOR_COVERED

        # Sight overlay: light blue alpha-blended on non-obstacle cells
        # (you can change this to overlay even on obstacles if you want)
        sight_alpha = float(self.cfg.sight_alpha)

        for aid, s in agent_state.items():
            if not getattr(s, "is_active", True):
                continue
            x = int(getattr(s, "x"))
            y = int(getattr(s, "y"))
            a_type = getattr(s, "type", "car")

            win = drone_fov if a_type == "drone" else car_fov
            radius = win // 2

            ys, xs = _square_slice(x, y, radius, self.H, self.W)

            # overlay only on free cells (not obstacles)
            mask_free = ~obs[ys, xs]
            if np.any(mask_free):
                region = img[ys, xs]
                region[mask_free] = _alpha_blend(region[mask_free], COLOR_SIGHT, sight_alpha)
                img[ys, xs] = region

        # Update the grid image
        self.im.set_data(img)

        # Agents on top: drones red, cars yellow
        drones_xy: List[Tuple[int, int]] = []
        cars_xy: List[Tuple[int, int]] = []
        for aid, s in agent_state.items():
            if not getattr(s, "is_active", True):
                continue
            x = int(getattr(s, "x"))
            y = int(getattr(s, "y"))
            if getattr(s, "type", "car") == "drone":
                drones_xy.append((x, y))
            else:
                cars_xy.append((x, y))

        if drones_xy:
            dx, dy = zip(*drones_xy)
        else:
            dx, dy = [], []
        if cars_xy:
            cx, cy = zip(*cars_xy)
        else:
            cx, cy = [], []

        # Marker size tuned to cell size
        marker_s = (self.cfg.cell_px * self.cfg.agent_marker_scale) ** 2

        self.drone_scatter.set_offsets(np.column_stack([dx, dy]) if len(dx) else np.zeros((0, 2)))
        self.car_scatter.set_offsets(np.column_stack([cx, cy]) if len(cx) else np.zeros((0, 2)))
        self.drone_scatter.set_sizes([marker_s] * (len(dx) if len(dx) else 1))
        self.car_scatter.set_sizes([marker_s] * (len(cx) if len(cx) else 1))

        # Title line with reward/info/battery
        parts = []
        if step_reward is not None:
            parts.append(f"reward={step_reward:.3f}")

        # Battery line per active agent
        batt_bits = []
        for aid, s in agent_state.items():
            if not getattr(s, "is_active", True):
                continue
            b = float(getattr(s, "battery", 0.0))
            batt_bits.append(f"{aid}:{b:.0f}%")
        if batt_bits:
            parts.append("battery[" + " ".join(batt_bits) + "]")

        # Optional extra info
        if infos:
            # Keep it short; you can customize what you display
            # e.g., coverage %, step, collisions, etc.
            common = infos.get("__common__", infos)
            if isinstance(common, dict):
                if "coverage" in common:
                    parts.append(f"coverage={common['coverage']:.3f}")
                if "t" in common:
                    parts.append(f"t={common['t']}")

        self.fig.suptitle(" | ".join(parts), fontsize=self.cfg.title_fontsize, y=0.98)

        # Render to RGB array
        self.canvas.draw()
        w, h = self.canvas.get_width_height()
        buf = np.frombuffer(self.canvas.tostring_rgb(), dtype=np.uint8)
        frame = buf.reshape((h, w, 3))
        return frame.copy()  # copy so it won't be tied to the canvas buffer