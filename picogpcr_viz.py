"""Train the minimal PicoGPCR model and animate its denoising process as a GIF.

    python picogpcr_viz.py
    python picogpcr_viz.py --data data/md_expanded/dataset.npz --out md.gif
    python picogpcr_viz.py --denoise-steps 100 --no-show
    python picogpcr_viz.py --steps 20 --no-show --out preview.gif

Uses picogpcr.py and its prepared datasets.
Defaults to 200 denoising updates (about 21 seconds at 10 fps), after 2,000 training steps.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

import picogpcr as p


@torch.no_grad()
def generate_frames(model, condition, alpha, embedding, seed, denoise_steps=p.TIMESTEPS):
    """DDIM with a configurable number of updates, retaining every state."""
    if not 2 <= denoise_steps <= p.TIMESTEPS:
        raise ValueError(f"denoise_steps must be between 2 and {p.TIMESTEPS}")
    noisy = torch.randn(1, p.LATENT, generator=torch.Generator().manual_seed(seed))
    frames = [noisy.numpy().reshape(7, 6).copy()]
    times = np.linspace(p.TIMESTEPS - 1, 0, denoise_steps, dtype=int)
    for i, time in enumerate(times):
        clean = model(noisy, condition[None], embedding[time:time + 1]).clamp(-2, 2)
        noise = (noisy - alpha[time].sqrt() * clean) / (1 - alpha[time]).sqrt()
        next_alpha = alpha[times[i + 1]] if i + 1 < len(times) else torch.tensor(1.0)
        noisy = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * noise
        frames.append(noisy.numpy().reshape(7, 6).copy())
    frames = np.asarray(frames)
    if not np.isfinite(frames).all():
        raise FloatingPointError("Nonfinite sampling result")
    return frames


def animate(coordinates, template, example, output, fps, show):
    with plt.style.context("dark_background"), plt.rc_context({"grid.color": "#334155"}):
        background = "#000000"
        figure = plt.figure(figsize=(7, 7), facecolor=background)
        view = figure.add_subplot(111, projection="3d", facecolor=background)
        for axis in (view.xaxis, view.yaxis, view.zaxis):
            axis.set_pane_color(background)
        figure.subplots_adjust(left=0.03, right=0.90, bottom=0.10, top=0.88)
        figure.suptitle("PicoGPCR: Building Scaffold from Noise", fontsize=17, y=0.97)
        title = figure.text(0.5, 0.91, "", ha="center", fontsize=11)

        # Keep the camera and scale fixed across the complete trajectory.
        low, high = coordinates.min(axis=(0, 1)), coordinates.max(axis=(0, 1))
        center, radius = (low + high) / 2, (high - low).max() / 2 + 2
        for setter, midpoint in zip((view.set_xlim, view.set_ylim, view.set_zlim), center):
            setter(midpoint - radius, midpoint + radius)
        view.set_box_aspect((1, 1, 1))
        view.view_init(elev=20, azim=-60)
        view.set(xlabel="X (Å)", ylabel="Y (Å)", zlabel="Z (Å)")

        backbone, = view.plot(*coordinates[0].T, color="#94a3b8", linewidth=1, alpha=0.7)
        numbers = np.array([r["number"] for r in template["residues"]])
        colors = ("#a78bfa", "#60a5fa", "#22d3ee", "#34d399", "#facc15", "#fb923c", "#f472b6")
        helices = []
        for i, (lo, hi) in enumerate(template["helices"]):
            ids = (numbers >= lo) & (numbers <= hi)
            line, = view.plot(*coordinates[0, ids].T, color=colors[i],
                              linewidth=2, marker="o", markersize=2.5)
            helices.append((ids, line))

        def update(frame):
            backbone.set_data_3d(*coordinates[frame].T)
            for ids, line in helices:
                line.set_data_3d(*coordinates[frame, ids].T)
            label = "Initial noise" if frame == 0 else f"Denoising step {frame}/{len(coordinates) - 1}"
            title.set_text(f"{label}")

        # Hold the final scaffold for one second before the GIF loops.
        frames = list(range(len(coordinates))) + [len(coordinates) - 1] * fps
        animation = FuncAnimation(figure, update, frames=frames, interval=1000 / fps, blit=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        animation.save(output, writer=PillowWriter(fps=fps), dpi=100,
                       savefig_kwargs={"facecolor": background, "transparent": False})
        print(f"Saved {output} · {len(coordinates) - 1} denoising steps · {len(frames) / fps:.1f} seconds")
        if show:
            plt.show()
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=p.ROOT / "data/synthetic/synthetic.npz")
    parser.add_argument("--steps", type=int, default=2000, help="Training iterations (default: 2000)")
    parser.add_argument("--model", choices=("mlp", "attention"), default="mlp", help="Denoiser architecture")
    parser.add_argument("--denoise-steps", type=int, default=p.TIMESTEPS,
                        help=f"Denoising updates / animation steps (2–{p.TIMESTEPS}; default: {p.TIMESTEPS})")
    parser.add_argument("--example", type=int, default=0, help="Dataset row supplying molecule features")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=10, help="GIF frames per second (1–50)")
    parser.add_argument("--out", type=Path, default=Path("visuals/picogpcr_viz.gif"))
    parser.add_argument("--no-show", action="store_true", help="Save the GIF without opening a window")
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.fps <= 50:
        parser.error("--steps must be positive and --fps must be between 1 and 50")
    if not 2 <= args.denoise_steps <= p.TIMESTEPS:
        parser.error(f"--denoise-steps must be between 2 and {p.TIMESTEPS}")
    if args.out.suffix.lower() != ".gif":
        parser.error("--out must end in .gif")
    if args.no_show:
        plt.switch_backend("Agg")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    template = json.loads((p.ROOT / "data/template.json").read_text())
    latent, condition, rotation_scale = p.load_data(args.data, template)
    if not 0 <= args.example < len(latent):
        parser.error("--example must be a valid dataset row")
    alpha, embedding = p.schedule()
    model, _ = p.train(latent, condition, alpha, embedding, args.steps, args.model)
    poses = generate_frames(model, condition[args.example], alpha, embedding, args.seed, args.denoise_steps)
    coordinates = np.array([p.decode(frame, template, rotation_scale) for frame in poses])
    animate(coordinates, template, args.example, args.out, args.fps, not args.no_show)


if __name__ == "__main__":
    main()
