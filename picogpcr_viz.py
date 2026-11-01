"""Train the minimal PicoGPCR model and animate its denoising process as a GIF.

    python picogpcr_viz.py
    python picogpcr_viz.py --data data/md_expanded/dataset.npz --out md.gif
    python picogpcr_viz.py --denoise-steps 100 --no-show
    python picogpcr_viz.py --steps 20 --no-show --out preview.gif

Uses picogpcr.py and its prepared datasets.
Defaults to 200 denoising updates (about 21 seconds at 10 fps), after 2,000 training steps.
The camera slowly rotates and reframes; --eta 0 disables fresh sampling noise.
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
def generate_frames(model, condition, alpha, embedding, seed, denoise_steps=p.TIMESTEPS, eta=0.3):
    """Seeded stochastic DDIM; eta=0 restores deterministic updates."""
    if not 2 <= denoise_steps <= p.TIMESTEPS:
        raise ValueError(f"denoise_steps must be between 2 and {p.TIMESTEPS}")
    if not 0 <= eta <= 1:
        raise ValueError("eta must be between 0 and 1")
    generator = torch.Generator().manual_seed(seed)
    noisy = torch.randn(1, p.LATENT, generator=generator)
    frames = [noisy.numpy().reshape(7, 6).copy()]
    times = np.linspace(p.TIMESTEPS - 1, 0, denoise_steps, dtype=int)
    for i, time in enumerate(times):
        clean = model(noisy, condition[None], embedding[time:time + 1]).clamp(-2, 2)
        noise = (noisy - alpha[time].sqrt() * clean) / (1 - alpha[time]).sqrt()
        next_alpha = alpha[times[i + 1]] if i + 1 < len(times) else torch.tensor(1.0)
        # DDIM variance: fresh noise fades to zero at the final clean estimate.
        sigma = eta * ((1 - next_alpha) / (1 - alpha[time]) * (1 - alpha[time] / next_alpha)).clamp(min=0).sqrt()
        noisy = next_alpha.sqrt() * clean + (1 - next_alpha - sigma.square()).clamp(min=0).sqrt() * noise
        if eta:
            noisy += sigma * torch.randn(noisy.shape, generator=generator)
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

        # Smooth framing, anticipating expansion while keeping every point in view.
        centers = (coordinates.min(axis=1) + coordinates.max(axis=1)) / 2
        for i in range(1, len(centers)):
            centers[i] = 0.85 * centers[i - 1] + 0.15 * centers[i]
        radii = np.abs(coordinates - centers[:, None, :]).max(axis=(1, 2)) + 2
        for i in range(len(radii) - 2, -1, -1):
            radii[i] = max(radii[i], 0.85 * radii[i + 1] + 0.15 * radii[i])
        for i in range(1, len(radii)):
            radii[i] = max(radii[i], 0.85 * radii[i - 1] + 0.15 * radii[i])
        view.set_box_aspect((1, 1, 1))
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
            for setter, midpoint in zip((view.set_xlim, view.set_ylim, view.set_zlim), centers[frame]):
                setter(midpoint - radii[frame], midpoint + radii[frame])
            view.view_init(elev=20, azim=-60 + 3 * frame / fps)
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
    parser.add_argument("--eta", type=float, default=0.3, help="Fresh sampling noise (0–1; 0 is deterministic)")
    parser.add_argument("--out", type=Path, default=Path("visuals/picogpcr_viz.gif"))
    parser.add_argument("--no-show", action="store_true", help="Save the GIF without opening a window")
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.fps <= 50:
        parser.error("--steps must be positive and --fps must be between 1 and 50")
    if not 2 <= args.denoise_steps <= p.TIMESTEPS:
        parser.error(f"--denoise-steps must be between 2 and {p.TIMESTEPS}")
    if not 0 <= args.eta <= 1:
        parser.error("--eta must be between 0 and 1")
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
    poses = generate_frames(model, condition[args.example], alpha, embedding, args.seed, args.denoise_steps, args.eta)
    coordinates = np.array([p.decode(frame, template, rotation_scale) for frame in poses])
    animate(coordinates, template, args.example, args.out, args.fps, not args.no_show)


if __name__ == "__main__":
    main()
