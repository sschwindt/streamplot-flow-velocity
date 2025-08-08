# Flow-velocity Streamplot Generator

Generate dense, publication-quality streamplots of surface flow velocity from videos of water runoff using OpenCV, [Farnebäck optical flow](http://link.springer.com/10.1007/3-540-45103-X_50),  and matplotlib. Velocities are **approximative** and intended for exploratory analysis and visualisation.

---

## Table of contents

* [Features](#features)
* [How it works (high level)](#how-it-works-high-level)
* [Repository layout](#repository-layout)
* [Requirements](#requirements)
* [Installation](#installation)
* [Quick start](#quick-start)
* [Configurable parameters](#configurable-parameters)
* [Output](#output)
* [Calibration & assumptions](#calibration--assumptions)
* [Logging](#logging)
* [Performance notes](#performance-notes)
* [Troubleshooting](#troubleshooting)
* [Contributing & code style](#contributing--code-style)
* [License](#license)
* [Cite this tool](#cite-this-tool)

---

## Features

* **Dense optical flow** via Farnebäck for per-pixel motion vectors, averaged across frames for robustness.
* **Grid down-sampling** to stabilise vectors and reduce visual clutter (configurable `BLOCK`).
* **Physical units (m/s)** estimated from a simple camera model (`compute_mm_per_px`) and video FPS.
* **Streamplot rendering** overlaid on the original frame, with a colorbar and Turbo colormap.
* **RAM-friendly streaming**: processes frames in a loop, releasing buffers; progress logged every 100 frames.

---

## How it works (high level)

The script reads a video, computes **Farnebäck optical flow** between consecutive grayscale frames, accumulates the (u, v) fields as a running mean, masks low-magnitude vectors, then downsamples and converts speed to m/s before drawing a matplotlib **streamplot** on top of a full-resolution background frame.

---

## Repository layout

```
.
├── moviestreamplot.py          # main script
├── test-videos/                # place input .mov/.mp4 here
└── test-output/                # suggested location for generated .jpg outputs
```

> By default, the script saves the result **next to the input video** (same name, `.jpg`). See [Output](#output) for how to save into `/test-output`.

---

## Requirements

* Python 3.9+ (recommended)
* Packages:

  * `opencv-python` (OpenCV 4.x)
  * `numpy`
  * `matplotlib`
  * `scikit-image` (for `block_reduce`)
* Optional: `pip-tools`, `venv`/`conda`

The script imports `cv2`, `numpy`, `matplotlib`, and `skimage.measure.block_reduce`.

---

## Installation

### Option 1: Explicit library installation

1. Create & activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2. Install dependencies
```bash
pip install --upgrade pip
pip install opencv-python numpy matplotlib scikit-image
```

### Option 2: Use requirements.txt

1. Create & activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2. Upgrade pip and install from requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quick start

1. Put a **1080p** water-flow video in `./test-videos/` (e.g., `hq100-inlet.mov`). The default script path is `test-videos/hq100-inlet.mov`. To use a resolution other than 1080p, change the value of the `PX_WIDTH` user parameter in the script.
2. (Optional) Open `moviestreamplot.py` and adjust the [parameters](#configurable-parameters).
3. Run:

```bash
python moviestreamplot.py
```

The console logs the progress and writes it to a `logfile.log`. A `.jpg` is saved to the same folder as the source video when the process is complete.

---

## Configurable parameters

At the top of the script, user-tunable constants are located:

| Name                | Purpose                                                                | Default                         |
| ------------------- | ---------------------------------------------------------------------- | ------------------------------- |
| `FILENAME`          | Path to input video                                                    | `"test-videos/hq100-inlet.mov"` |
| `BLOCK`             | Down-sampling window `(rows, cols)`; smaller → more detail (and noise) | `(64, 64)`                      |
| `MAG_THRESH`        | Discards vectors below threshold (px/frame); use \~0.3–0.5 for 30 FPS  | `0.3`                           |
| `DISTANCE_2_OBJECT` | Camera-to-water distance (m) for scale                                 | `0.75`                          |
| `LENS_TYPE`         | `"normal"` or `"wide"`; sets FOV                                       | `"wide"`                        |
| `PX_WIDTH`          | Horizontal resolution (px)                                             | `1920`                          |
| `SHOW_FIGURE`       | Show interactive plot window                                           | `False`                         |

---

## Output

* **Default behaviour:** the figure is saved as a `.jpg` **next to the video** (`Path(FILENAME).with_suffix('.jpg')`). For example, `test-videos/hq100-inlet.mov` >> `test-videos/hq100-inlet.jpg`.
* **Save to `/test-output`:** edit the save line in `make_streamplot`:

```python
# replace:
out_png = Path(FILENAME).with_suffix('.jpg')
# with:
out_name = Path(FILENAME).with_suffix('.jpg').name
out_png = Path("test-output") / out_name
```

> The image includes a colorbar labelled in **m s⁻¹** and uses the Turbo colormap with line width scaled by speed. Axes are hidden for a clean figure.

---

## Calibration & assumptions

* The **mm/px** scale uses a simple FOV model with diagonal angle **109°** (wide) or **69°** (normal); `compute_mm_per_px(distance_m, lens, px_width)` converts to mm/px. Example guide values are logged.
* The camera should be **perpendicular to the flow**
* The script assumes **1080p** imagery (`1920 x 1080`). To use a resolution other than 1080p, change the value of the `PX_WIDTH` user parameter in the script.
* Reported speeds are approximate and depend on scene geometry, lens, and stability.

---

## Logging

A detailed log is written to `logfile.log` (script directory) and INFO-level messages are echoed to the console. Progress messages appear every 100 frames; exceptions are caught and logged before exit.

---

## Performance notes

* Frames are read in a loop and OpenCV buffers are promptly released (`del` + `gc.collect()`), which helps keep memory bounded on long videos.
* The **down-sampling block size** is the main knob for noise vs. detail; try `(32, 32)` for finer flow or `(96, 96)` for more smoothing.
* If your video is not 1080p, update `PX_WIDTH` and consider revising the assumptions accordingly.

---

## Troubleshooting

* **"Could not open video ..."** → check `FILENAME` path, codec support, and permissions.
* **"No frames processed"** → the video could not be read or ended immediately; verify file integrity.
* **All speeds near zero** → lower `MAG_THRESH`, verify camera is steady, and ensure sufficient texture for optical flow.

---

## Contributing & code style

* This project follows **PEP 8** and uses type hints where practical.
* Please run a formatter (**black**) and linter (**ruff**/`flake8`) before submitting PRs.
* Add/adjust docstrings and comments to keep the code self-documenting.
* Tests and small sample clips are welcome (place in `test-videos/` with short filenames).

---

## License

**BSD 3-Clause License.** See `LICENSE` in the repository. You’re free to use, modify, and redistribute with attribution.

---

## Cite this tool

If you use this in research or a report, please cite the repository:

```
@software{schwindt2025streamplot-generator,
  title = {Flow-velocity Streamplot Generator},
  author = {Sebsatian Schwindt},
  year = {2025},
  note = {BSD-3-Clause License},
  url = {https://github.com/sschwindt/streamplot-flow-velocity},
  doi = {zenodo}
}
```

---

