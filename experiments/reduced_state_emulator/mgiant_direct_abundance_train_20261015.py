"""Matched two- and six-profile M-giant networks with 84 inputs.

The inputs are those of the direct-abundance family: T_eff, log g, the
microturbulence, [Fe/H] and the 80 [X/Fe] of the other elements. Training rows
are the direct-abundance training split of the normal stars plus the M-giant
training rows of the 84-input corpus. Architecture, loss, the half-and-half
batches, the standardization and the checkpoint selection are those of
``coauthor_mgiant_train_20261009``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from coauthor_train_20261005 import load_family
from coauthor_mgiant_train_20261009 import train


def load(corpus_root: Path, split_path: Path, mgiant: Path) -> dict:
    normal = load_family(corpus_root, "direct_abundance")
    normal_labels = normal["features"].copy()
    normal_labels[:, 0] = normal["teff"]
    split = json.loads(split_path.read_text())["indices"]
    with np.load(mgiant, allow_pickle=False) as d:
        if not np.allclose(d["standard_rosseland_optical_depth"], normal["tau"], rtol=1e-12, atol=0):
            raise ValueError("M-giant corpus uses a different optical-depth grid")
        cool_labels, cool_profiles, roles = d["labels"].copy(), d["atmosphere_profiles"].copy(), d["roles"].copy()
    offset = len(normal_labels)
    labels = np.concatenate((normal_labels, cool_labels))
    features = labels.copy()
    features[:, 0] = 5040.0 / labels[:, 0]
    return dict(labels=labels, features=features, teff=labels[:, 0], tau=normal["tau"],
                profiles=np.concatenate((normal["profiles"], cool_profiles)),
                normal_train=np.asarray(split["train"]), normal_validation=np.asarray(split["validation"]),
                cool_train=offset + np.flatnonzero(roles == "train"),
                cool_validation=offset + np.flatnonzero(roles == "validation"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True, help="direct_abundance split.json of the coauthor validation")
    parser.add_argument("--mgiant", type=Path, required=True, help="84-input M-giant corpus.npz")
    parser.add_argument("--reference", type=Path, required=True, help="direct-abundance checkpoint (architecture)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profiles", type=int, choices=(2, 6), required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    data = load(args.corpus_root, args.split, args.mgiant)
    config = dict(torch.load(args.reference, map_location="cpu", weights_only=False)["model"]["config"])
    train(data, args.profiles, config, args)
    path = args.out / f"checkpoint_{args.profiles}profile.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    with np.load(args.mgiant, allow_pickle=False) as d:
        fields = [str(f) for f in d["label_fields"]]
    checkpoint.update(family="mgiant_direct_abundance")
    checkpoint["labels"]["fields"] = fields
    torch.save(checkpoint, path)


if __name__ == "__main__":
    main()
