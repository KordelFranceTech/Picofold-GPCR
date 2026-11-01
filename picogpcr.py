"""Minimal PicoGPCR: load an NPZ, train, and draw a generated C-alpha backbone.

    python picogpcr.py
    python picogpcr.py --data data/md_expanded/dataset.npz
    python picogpcr.py --steps 20 --no-show --out preview.png

Both the synthetic dataset and prepared MD dataset are bundled - no downloads are needed.
Note that this is a teaching demo, not a benchmarked protein folding model for medical use.
The selected row supplies molecule features, not target coordinates, at sampling.
The plot shows scaffold C-alpha atoms, not an all-atom molecule or a binding pose.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
LATENT, CONDITION, TIME, WIDTH, DEPTH, TIMESTEPS = 42, 265, 16, 192, 4, 200


# Model: seven helix poses + molecule features + time -> clean helix poses.
class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(WIDTH, WIDTH)
        self.b = nn.Linear(WIDTH, WIDTH)

    def forward(self, x):
        return F.relu(x + 0.5 * self.b(F.relu(self.a(x))))


class PicoGPCR(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(LATENT + CONDITION + TIME, WIDTH)
        self.blocks = nn.ModuleList([Residual() for _ in range(DEPTH)])
        self.output = nn.Linear(WIDTH, LATENT)

    def forward(self, noisy, condition, time):
        x = F.relu(self.input(torch.cat((noisy, condition, time), dim=-1)))
        for block in self.blocks:
            x = block(x)
        return self.output(x)


class HelixAttention(nn.Module):
    """Seven helix tokens exchange information through four attention blocks."""
    def __init__(self):
        super().__init__()
        width = 128
        self.pose = nn.Linear(6, width)
        self.context = nn.Linear(CONDITION + TIME, width)
        self.helix = nn.Parameter(torch.randn(1, 7, width) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=2 * width,
                                       dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(4)
        ])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 6))

    def forward(self, noisy, condition, time):
        x = self.pose(noisy.reshape(-1, 7, 6)) + self.helix
        x = x + self.context(torch.cat((condition, time), dim=-1))[:, None]
        for block in self.blocks:
            x = block(x)
        return self.output(x).flatten(1)


def schedule():
    times = np.arange(TIMESTEPS, dtype=np.float64)
    angles = times[:, None] / (TIMESTEPS - 1) * 2.0 ** np.arange(TIME // 2) * math.pi
    embedding = np.concatenate((np.sin(angles), np.cos(angles)), axis=1)
    alpha = np.cos(((times + 1) / TIMESTEPS + 0.008) / 1.008 * math.pi / 2) ** 2
    alpha = np.maximum(alpha / math.cos(0.008 / 1.008 * math.pi / 2) ** 2, 1e-5)
    return torch.tensor(alpha, dtype=torch.float32), torch.tensor(embedding, dtype=torch.float32)


def load_data(path, template):
    with np.load(path, allow_pickle=False) as data:
        latent = torch.tensor(data["latent"], dtype=torch.float32)
        condition = torch.tensor(data["condition"], dtype=torch.float32)
        metadata = json.loads(str(data["metadata"]))
    if latent.ndim != 2 or latent.shape[1] != LATENT or condition.shape != (len(latent), CONDITION):
        raise ValueError("Expected latent [N, 42] and condition [N, 265] arrays")
    if not len(latent) or not torch.isfinite(latent).all() or not torch.isfinite(condition).all():
        raise ValueError("Dataset must be nonempty and finite")
    if metadata.get("feature_schema") != "rdkit-morgan-r2-chiral-256-desc8-presence-v1":
        raise ValueError("Unsupported molecule features")
    if metadata.get("template_id") != template["id"]:
        raise ValueError("Dataset uses a different scaffold")
    schema = metadata.get("model_schema", "picogpcr-v1")
    if schema not in ("picogpcr-v1", "picogpcr-v2"):
        raise ValueError("Unsupported geometry schema")
    scales = {"translation_scale_angstrom": 2.0, "rotation_scale_radian": 0.5 if schema.endswith("v2") else 0.1,
              "latent_limit": 2.0}
    for key, value in scales.items():
        if metadata.get(key, value if schema.endswith("v1") else None) != value:
            raise ValueError(f"Incompatible geometry field: {key}")
    if latent.abs().max() > 2.0001:
        raise ValueError("Helix poses exceed the supported range")
    print(f"Loaded {len(latent):,} rows: {metadata.get('source', 'unspecified')} ({schema})")
    return latent, condition, scales["rotation_scale_radian"]


def train(latent, condition, alpha, embedding, steps, architecture="mlp"):
    model = {"mlp": PicoGPCR, "attention": HelixAttention}[architecture]()
    average = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    losses = []
    for step in range(steps):
        indices = torch.randint(len(latent), (64,))
        clean, molecule = latent[indices], condition[indices].clone()
        molecule[torch.rand(64) < 0.2] = 0  # Occasionally learn without a molecule condition.
        time = torch.randint(TIMESTEPS, (64,))
        a = alpha[time, None]
        noisy = a.sqrt() * clean + (1 - a).sqrt() * torch.randn_like(clean)
        loss = F.mse_loss(model(noisy, molecule, embedding[time]), clean)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        with torch.no_grad():
            for averaged, weight in zip(average.parameters(), model.parameters()):
                averaged.lerp_(weight, 0.01)
        losses.append(loss.item())
        if step == 0 or (step + 1) % 200 == 0 or step + 1 == steps:
            print(f"Step {step + 1:4d} | training MSE {loss.item():.5f}")
    return average, losses


@torch.no_grad()
def generate(model, condition, alpha, embedding, seed):
    noisy = torch.randn(1, LATENT, generator=torch.Generator().manual_seed(seed))
    times = np.linspace(TIMESTEPS - 1, 0, 40, dtype=int)
    for i, time in enumerate(times):
        clean = model(noisy, condition[None], embedding[time:time + 1]).clamp(-2, 2)
        noise = (noisy - alpha[time].sqrt() * clean) / (1 - alpha[time]).sqrt()
        next_alpha = alpha[times[i + 1]] if i + 1 < len(times) else torch.tensor(1.0)
        noisy = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * noise
    return noisy.numpy().reshape(7, 6)


def decode(poses, template, rotation_scale):
    reference = np.array([r["xyz"] for r in template["residues"]], dtype=float)
    numbers = np.array([r["number"] for r in template["residues"]])
    coordinates, covered = reference.copy(), np.zeros(len(reference), dtype=bool)
    for pose, (lo, hi) in zip(poses, template["helices"]):
        ids = (numbers >= lo) & (numbers <= hi)
        center = reference[ids].mean(axis=0)
        rotation = Rotation.from_rotvec(pose[3:] * rotation_scale).as_matrix()
        coordinates[ids] = (reference[ids] - center) @ rotation.T + center + pose[:3] * 2
        covered |= ids
    # Carry the reference loops along with their helix endpoints.
    anchors = np.flatnonzero(covered)
    displacement = coordinates[anchors] - reference[anchors]
    for axis in range(3):
        coordinates[~covered, axis] += np.interp(np.flatnonzero(~covered), anchors, displacement[:, axis])
    return coordinates


def visualize(coordinates, losses, example, output, show):
    with plt.style.context("dark_background"), plt.rc_context({"grid.color": "#334155"}):
        figure = plt.figure(figsize=(11, 5), layout="constrained", facecolor="#000000")
        curve = figure.add_subplot(121)
        curve.plot(np.arange(1, len(losses) + 1), losses, linewidth=0.8)
        curve.set(xlabel="Training step", ylabel="Training MSE", title="Denoising loss (training only)")
        view = figure.add_subplot(122, projection="3d")
        for axis in (view.xaxis, view.yaxis, view.zaxis):
            axis.set_pane_color("#000000")
        view.plot(*coordinates.T, color="gray", alpha=0.5, linewidth=1)
        view.scatter(*coordinates.T, c=np.arange(len(coordinates)), cmap="viridis", s=9)
        center, radius = coordinates.mean(axis=0), np.ptp(coordinates, axis=0).max() / 2 + 1
        for setter, midpoint in zip((view.set_xlim, view.set_ylim, view.set_zlim), center):
            setter(midpoint - radius, midpoint + radius)
        view.set_box_aspect((1, 1, 1))
        view.set(xlabel="X (Å)", ylabel="Y (Å)", zlabel="Z (Å)", title=f"Generated Cα trace")
        figure.suptitle("PicoGPCR: educational scaffold visualization")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160, facecolor="#000000", transparent=False)
        print(f"Saved {output}")
        if show:
            plt.show()
        plt.close(figure)


def main():
    """
    `--steps` controls training iterations
    `--example` selects a zero-based row's molecule features, not its target poses 
    `--seed` controls training and sampling (default `42`)
    Both scripts support `--data`, `--example`, `--seed`, `--out`, and `--no-show`. 
    They save visuals, not trained weights.
    Note that the prior run's artifacts are replaced upon each new generation.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data/synthetic/synthetic.npz")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--model", choices=("mlp", "attention"), default="mlp", help="Denoiser architecture")
    parser.add_argument("--example", type=int, default=0, help="Dataset row supplying molecule features")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("visuals/picogpcr.png"))
    parser.add_argument("--no-show", action="store_true", help="Save the plot without opening a window")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.no_show:
        plt.switch_backend("Agg")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    template = json.loads((ROOT / "data/template.json").read_text())
    latent, condition, rotation_scale = load_data(args.data, template)
    if not 0 <= args.example < len(latent):
        parser.error("--example must be a valid dataset row")
    alpha, embedding = schedule()
    model, losses = train(latent, condition, alpha, embedding, args.steps, args.model)
    poses = generate(model, condition[args.example], alpha, embedding, args.seed)
    coordinates = decode(poses, template, rotation_scale)
    visualize(coordinates, losses, args.example, args.out, not args.no_show)


if __name__ == "__main__":
    main()
