"""Train and export both MD models as standalone Hugging Face model folders.

    python export_model.py
    python export_model.py --model attention --steps 2000 --out model

Exports EMA inference weights, not optimizer state. Nothing is uploaded.
"""

import argparse
import hashlib
import json
import shutil
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

import picogpcr as p


def export_model(model, losses, condition, template, metadata, rotation_scale, args, architecture):
    output = args.out / architecture
    output.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(state, str(output / "model.safetensors"), metadata={"format": "pt"})
    torch.save(state, output / "model.pt")
    config = {
        "model_type": "picogpcr", "architectures": [type(model).__name__], "architecture": architecture,
        "torch_dtype": "float32", "parameter_count": sum(value.numel() for value in model.parameters()),
        "latent_size": p.LATENT, "condition_size": p.CONDITION, "time_size": p.TIME,
        "hidden_size": p.WIDTH if architecture == "mlp" else 128, "depth": p.DEPTH,
        "attention_heads": 4 if architecture == "attention" else None,
        "feature_schema": metadata["feature_schema"],
        "geometry": {"model_schema": metadata.get("model_schema", "picogpcr-v1"),
                     "template_id": template["id"], "translation_scale_angstrom": 2.0,
                     "rotation_scale_radian": rotation_scale, "latent_limit": 2.0},
        "diffusion": {"prediction_type": "clean_latent", "schedule": "cosine", "offset": 0.008,
                      "alpha_floor": 1e-5, "timesteps": p.TIMESTEPS, "sampler": "ddim",
                      "sampling_steps": 40, "eta": args.eta},
        "training": {"dataset_file": args.data.name, "dataset_rows": args.rows,
                     "dataset_sha256": args.dataset_sha256, "steps": args.steps, "seed": args.seed,
                     "batch_size": 64, "optimizer": "AdamW", "learning_rate": 0.001,
                     "weight_decay": 0.0001, "gradient_clip": 1.0, "condition_dropout": 0.2,
                     "weights": "ema", "ema_decay": 0.99, "last_batch_training_mse": losses[-1],
                     "evaluation": "training only; no held-out validation or test"},
        "data_provenance": {key: value for key, value in metadata.items()
                            if isinstance(value, (str, int, float, bool))},
        "template_sha256": hashlib.sha256((p.ROOT / "data/template.json").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256((p.ROOT / "picogpcr.py").read_bytes()).hexdigest(),
        "environment": {name: version(name) for name in ("torch", "numpy", "scipy", "matplotlib", "safetensors")},
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "example_condition.json").write_text(json.dumps(
        {"dataset_row": args.example, "condition": condition.tolist()}, indent=2) + "\n")
    shutil.copyfile(p.ROOT / "data/template.json", output / "template.json")
    for name in ("picogpcr.py", "requirements.txt", "LICENSE"):
        shutil.copyfile(p.ROOT / name, output / name)
    credits = (p.ROOT / "README.md").read_text().split("## Data and credits\n", 1)[1].split("## License", 1)[0].strip()
    (output / "README.md").write_text(f"""---
license: mit
tags:
- pytorch
- safetensors
- gpcr
- diffusion
---

# PicoGPCR: {architecture}

Custom PyTorch model with {config['parameter_count']:,} parameters, trained on
{args.rows:,} `{metadata['source']}` examples for {args.steps:,} steps with seed {args.seed}.
`model.safetensors` contains the EMA inference weights (decay 0.99), not optimizer state.
The input is a noisy 42-value helix pose, 265 precomputed molecule features, and 16 time features.
The output is a clean 42-value pose. The included decoder produces a 302-position C-alpha trace.

## Load locally

Run from this folder after `python -m pip install -r requirements.txt`:

```python
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
import picogpcr as p

root = Path(".")
config = json.loads((root / "config.json").read_text())
model = getattr(p, config["architectures"][0])()
model.load_state_dict(load_file(str(root / "model.safetensors")), strict=True)
model.eval()
torch.set_num_threads(2)
example = json.loads((root / "example_condition.json").read_text())
condition = torch.tensor(example["condition"], dtype=torch.float32)
alpha, embedding = p.schedule()
poses = p.generate(model, condition, alpha, embedding, seed={args.seed}, eta=config["diffusion"]["eta"])
template = json.loads((root / "template.json").read_text())
coordinates = p.decode(poses, template, config["geometry"]["rotation_scale_radian"])
print(coordinates.shape)  # (302, 3)
```

The example comes from dataset row {args.example}; all-zero features mean unconditioned generation.
New inputs must use the feature schema in `config.json`; this package does not convert SMILES.
The config records geometry scales, diffusion defaults, data/source hashes, and dependency versions.
Use the explicit loader above; this is a custom PyTorch package, not a Transformers `AutoModel`.
This folder can be uploaded as the root of its own Hugging Face model repository.

## Scope

Training-only educational geometry: no held-out accuracy is claimed. MD examples are correlated
simulated frames; synthetic examples, when used, are artificial. Helices remain rigid and loops
are interpolated. Outputs are not binding predictions, molecular dynamics, or all-atom structures.
Fresh sampling noise changes generated poses, not the training objective.

## Data and credits

{credits}

## License

Code and weights follow the project's [MIT license](LICENSE). Template and training-data
attributions and source terms are listed above. Training datasets are not included in this export.
""")
    # Check serialization against the trained EMA, including stochastic generation.
    restored = type(model)().eval()
    restored.load_state_dict(load_file(str(output / "model.safetensors")), strict=True)
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in state.items())
    pytorch_state = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    assert all(torch.equal(value, pytorch_state[key]) for key, value in state.items())
    alpha, embedding = p.schedule()
    expected = p.generate(model, condition, alpha, embedding, args.seed, args.eta)
    actual = p.generate(restored, condition, alpha, embedding, args.seed, args.eta)
    assert np.array_equal(expected, actual) and np.isfinite(actual).all()
    print(f"Exported {output} · {config['parameter_count']:,} parameters · exact reload parity")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=p.ROOT / "data/md_expanded/dataset.npz")
    parser.add_argument("--model", choices=("mlp", "attention", "both"), default="both")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--example", type=int, default=0)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--out", type=Path, default=p.ROOT / "model")
    args = parser.parse_args()
    if args.steps < 1 or not 0 <= args.eta <= 1:
        parser.error("--steps must be positive and --eta must be between 0 and 1")
    torch.set_num_threads(2)
    template = json.loads((p.ROOT / "data/template.json").read_text())
    latent, condition, rotation_scale = p.load_data(args.data, template)
    if not 0 <= args.example < len(latent):
        parser.error("--example must be a valid dataset row")
    with np.load(args.data, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
    args.rows = len(latent)
    args.dataset_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    alpha, embedding = p.schedule()
    for architecture in ("mlp", "attention") if args.model == "both" else (args.model,):
        torch.manual_seed(args.seed)
        model, losses = p.train(latent, condition, alpha, embedding, args.steps, architecture)
        export_model(model, losses, condition[args.example], template, metadata, rotation_scale, args, architecture)


if __name__ == "__main__":
    main()
