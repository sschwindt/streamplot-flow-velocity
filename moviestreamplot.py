#!/usr/bin/env python3
"""
Flow-velocity Streamplot Generator
--------------------------------
Creates a streamplot from a 1080 p video clip of water flow. Flow velocities are approximative!

Note:
    * The camera should be placed perpendicular to the flow.
    * The script assumes 1080p resolution, that is, 1920x1080 images

"""

import logging
from pathlib import Path
import sys
import cv2  # opencv
import numpy as np
import matplotlib.pyplot as plt
from PIL.DdsImagePlugin import DDS_HEADER_FLAGS_PITCH
from matplotlib.colors import Normalize
from skimage.measure import block_reduce   # pip install scikit-image
import gc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# USER PARAMETERS ------------------------------------------------------------------------------------------------------
FILENAME            = "test-videos/mq-inlet.mov"
BLOCK               = (32, 32)  # down-sampling window (rows, cols): min(frame_dim / BLOCK) = approx. 70 to 250 - the smaller BLOCK is, the more noise
MAG_THRESH          = 0.1       # px / frame, vectors below are discarded - set smaller 0.5 for FPS=30; min. 0.3, max. 0.5
DISTANCE_2_OBJECT   = 0.75       # m from lens to water
LENS_TYPE           = "wide"    # "normal" (Main Camera) or "wide" (Ultrawide Camera) - defines opening angle
PX_WIDTH            = 1920      # horizontal resolution in pixels
DPI                 = 600       # export figure (jpg) dpi
SHOW_FIGURE = False
# ----------------------------------------------------------------------------------------------------------------------

# Logger
LOG_PATH = Path(__file__).with_name("logfile.log")
LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# File handler: everything
fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
fh.setFormatter(logging.Formatter(LOG_FMT, DATE_FMT))
fh.setLevel(logging.DEBUG)
root_logger.addHandler(fh)

# Console handler: INFO and above
sh = logging.StreamHandler(sys.stderr)
sh.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
sh.setLevel(logging.INFO)
root_logger.addHandler(sh)

log = logging.getLogger(__name__)


# CORE FUNCTIONS
def compute_mm_per_px(distance_m: float, lens: str, px_width: int) -> float:
    """Return mm / px from a simple FOV model."""
    diag_angle = 109 if lens.lower() == "wide" else 69  # degrees
    scene_width = 2 * distance_m * np.tan(np.deg2rad(diag_angle / 2))
    return scene_width / px_width * 1000  # to millimetres


def make_streamplot(u, v, speed,
                    block=BLOCK,
                    background=None,
                    alpha=0.5):
    """
    Make a streamplot from u, v, and speed; optionally add a full-resolution
    background image.

    Parameters
    ----------
    u, v : 2D ndarrays
        Down-sampled velocity components (same shape).
    speed : 2D ndarray
        m / s magnitude, same shape as u.
    block : tuple(int, int)
        (rows, cols) used in block_reduce; needed to place streamlines
        in *pixel* coordinates.
    background : ndarray or None
        Raw color frame straight from cv2.VideoCapture.read()
        (shape h * w * 3, BGR).
    alpha : float
        0 = invisible, 1 = opaque background.

    Saves a JPG on top of the video and shows the figure if SHOW_FIGURE is True.
    """

    log.info("Rendering streamplot ...")
    h_ds, w_ds = u.shape
    by, bx     = block

    # Build *pixel-space* coordinate grid (centres of each block)
    x_centres = (np.arange(w_ds) + 0.5) * bx
    y_centres = (np.arange(h_ds) + 0.5) * by
    X, Y      = np.meshgrid(x_centres, y_centres)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Optional full-res background
    if background is not None:
        bg_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        Hpx, Wpx, _ = bg_rgb.shape
        ax.imshow(bg_rgb,
                  extent=[0, Wpx, Hpx, 0],   # left, right, bottom, top
                  origin='upper',
                  alpha=alpha,
                  zorder=0)
        # Keep axis limits fixed to the image
        ax.set_xlim(0, Wpx)
        ax.set_ylim(Hpx, 0)
    else:
        Hpx, Wpx = h_ds*by, w_ds*bx    # fallback for axis scaling

    # Normalize velocity to align with UE and OpenFOAM color maps
    norm = Normalize(vmin=0.0, vmax=2.0)

    # Streamplot (-v to convert OpenCV down-is-positive to matplotlib up)
    strm = ax.streamplot(
        X, Y,
        u,
        v,
        color=speed,
        cmap='turbo',
        norm=norm,
        linewidth=2 * speed / (speed.max() + 1e-9),
        density=1.2,
        arrowsize=1.5,
        zorder=2
    )


    # Decorations
    cb_ax = inset_axes(
        ax,  # parent axes
        width="3%",  # 3 percent of the parent width
        height="100%",  # full parent height
        loc="lower left",
        bbox_to_anchor=(1.02, 0, 1, 1),  # (x0, y0, width, height) in axes-coords
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    fig.colorbar(strm.lines, cax=cb_ax,
                 label=r'm s$^{-1}$',
                 norm=norm,
                 )
    ax.set_axis_off()
    # ax.set_title('Flow velocity streamplot')
    # ax.set_xlabel('Pixel column')
    # ax.set_ylabel('Pixel row')

    plt.tight_layout()
    if SHOW_FIGURE:
        plt.show()

    out_png = Path(FILENAME).with_suffix('.jpg')
    fig.savefig(out_png, bbox_inches='tight', dpi=DPI)
    log.info("Saved figure to %s", out_png)


def main() -> None:
    try:
        # Derived scale factor
        mm_per_px = compute_mm_per_px(DISTANCE_2_OBJECT, LENS_TYPE, PX_WIDTH)
        msg_guide = "Guide values (normal/main camera) for distances of:\n\t - 0.5 m, mm/px should be ~0.44\n\t - 2.8 m, mm/px should be ~1.46 "
        log.info("mm / px = %.3f  (distance %.2f m, lens '%s')",
                 mm_per_px, DISTANCE_2_OBJECT, LENS_TYPE)
        log.info(msg_guide)

        # Open video
        log.info("Opening video %s ...", FILENAME)
        cap = cv2.VideoCapture(FILENAME)
        if not cap.isOpened():
            log.error("Could not open video %s", FILENAME)
            sys.exit(1)

        fps = cap.get(cv2.CAP_PROP_FPS)
        log.info("Video FPS: %.3f", fps)

        ret, prev_bgr = cap.read()
        if not ret:
            log.error("Could not read first frame")
            sys.exit(1)
        prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)

        # Pre-allocate accumulators to avoid huge Python list
        accum_u = np.zeros_like(prev, dtype=np.float32)
        accum_v = np.zeros_like(prev, dtype=np.float32)
        frame_count = 0

        # Optical-flow accumulation (running mean)
        while True:
            ret, curr_bgr = cap.read()
            if not ret:
                break
            curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            accum_u += flow[..., 0]
            accum_v += flow[..., 1]
            frame_count += 1
            prev = curr

            if frame_count % 100 == 0:
                log.debug("Processed %d frames (RAM friendly streaming mode)", frame_count)

            # Free OpenCV buffers promptly
            del flow, curr_bgr
            gc.collect()

        cap.release()
        if frame_count == 0:
            log.error("No frames processed -- exiting.")
            sys.exit(1)
        log.info("Total frames processed: %d", frame_count)

        # 3. Mean flow field -----------------------------------------------------
        u = accum_u / frame_count
        v = accum_v / frame_count
        del accum_u, accum_v  # free memory early
        gc.collect()

        # 4. Speed & mask --------------------------------------------------------
        speed = np.hypot(u, v)
        mask = speed >= MAG_THRESH
        u[~mask] = v[~mask] = speed[~mask] = 0

        # 5. Down-sample ---------------------------------------------------------
        log.debug("Down-sampling with block size %s", BLOCK)
        u_ds     = block_reduce(u, BLOCK, np.mean)
        v_ds     = block_reduce(v, BLOCK, np.mean)
        speed_ds = block_reduce(speed, BLOCK, np.mean)
        del u, v, speed
        gc.collect()

        # Convert speed to m/s
        speed_m_s = speed_ds * mm_per_px * fps / 1000.
        log.info("Max speed: %.1f m/s", speed_m_s.max())

        make_streamplot(u_ds, v_ds, speed_m_s, background=prev_bgr)

        log.info("Finished successfully.")

    except Exception as e:
        log.exception(f"Unhandled exception -- aborting.\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
