"""Freeze the safezone v2 warm starts for the inner-loop validation (9 dwarfs).

Protocol: ``notes/m_star_inner_loop_validation_preregistration_20260924.md``.
The three safezone v2 checkpoints predict every opened validation row of the
safezone corpus; the coordinate-wise median in physical (m, T) is the warm
start, exactly as ``evaluate_mstar_candidate_v1`` builds it.  The profile
metrics over all 18 rows are recomputed as a reproduction check against the
safezone v2 record.  Only the dwarf rows are written, with the reference
product each row was taken from (``source_product_paths``, relative to the
emulator-v1 campaign tree).

    python -m experiments.reduced_state_emulator.build_m_star_inner_loop_validation_inputs_20260924
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from reduced_state.emulator import load_physical_checkpoint, predict_physical_state

from .evaluate_mstar_candidate_v1 import profile_metrics
from .train_mstar_physical_v1 import load_cool_corpus

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / 'results/m_star_cool_corpus_safezone_v1/cool_truth_corpus.npz'
CHECKPOINT_DIR = REPO / 'artifacts/m_star_emulator_safezone_v2'
SEEDS = (20260831, 20260901, 20260902)
SAFEZONE_SUMMARY = REPO / 'results/m_star_emulator_safezone_v2/candidate_validation/validation_summary.json'
DEVELOPMENT_OVERLAP = ('g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3400',)
OUT = REPO / 'results/m_star_inner_loop_validation_20260924/inputs'


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    cool = load_cool_corpus(CORPUS)
    with np.load(CORPUS, allow_pickle=False) as raw:
        provenance = {key: raw[key] for key in (
            'track_ids', 'label_fields', 'source_product_paths', 'source_campaigns', 'flux_metric_fields',
            'flux_metrics',
        )}
    validation = np.flatnonzero(cool['roles'] == 'validation')
    labels = cool['labels'][validation]
    masses, temperatures = [], []
    for seed in SEEDS:
        model, standardization, _meta = load_physical_checkpoint(
            CHECKPOINT_DIR / f'checkpoint_mstar_seed{seed}.pt'
        )
        mass, temperature = predict_physical_state(model, standardization, labels)
        masses.append(mass)
        temperatures.append(temperature)
    mass = np.median(np.stack(masses, axis=0), axis=0)
    temperature = np.median(np.stack(temperatures, axis=0), axis=0)

    reproduced = profile_metrics(
        mass, temperature, cool['column_mass'][validation], cool['temperature'][validation],
    )
    recorded = json.loads(SAFEZONE_SUMMARY.read_text())['profile_metrics']
    for field in ('temperature_relative', 'column_mass_dex'):
        for stat in ('median', 'p95', 'max'):
            if not np.isclose(reproduced[field][stat], recorded[field][stat], rtol=1e-12, atol=0.0):
                raise SystemExit(
                    f'warm starts do not reproduce safezone v2: {field} {stat} '
                    f'{reproduced[field][stat]!r} != {recorded[field][stat]!r}'
                )

    dwarf = np.flatnonzero(labels[:, 1] >= 3.5)
    rows = validation[dwarf]
    node_ids = cool['node_ids'][rows]
    flux_fields = [str(name) for name in provenance['flux_metric_fields']]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / 'warm_starts.npz',
        node_ids=node_ids,
        track_ids=provenance['track_ids'][rows],
        labels=cool['labels'][rows],
        label_fields=provenance['label_fields'],
        column_mass=mass[dwarf],
        temperature=temperature[dwarf],
        reference_product_paths=provenance['source_product_paths'][rows],
        reference_campaigns=provenance['source_campaigns'][rows],
        reference_flux_p95_percent=provenance['flux_metrics'][rows, flux_fields.index('p95_absolute_flux_error_percent')],
        held_out=np.array([str(node) not in DEVELOPMENT_OVERLAP for node in node_ids]),
    )
    manifest = {
        'preregistration': 'notes/m_star_inner_loop_validation_preregistration_20260924.md',
        'corpus': str(CORPUS.relative_to(REPO)),
        'corpus_sha256': _sha256(CORPUS),
        'checkpoints': [
            {'seed': seed, 'sha256': _sha256(CHECKPOINT_DIR / f'checkpoint_mstar_seed{seed}.pt')}
            for seed in SEEDS
        ],
        'ensemble_policy': 'coordinate-wise median of three physical-state predictions',
        'reproduced_profile_metrics_all_validation_rows': reproduced,
        'development_overlap': list(DEVELOPMENT_OVERLAP),
        'warm_starts_sha256': _sha256(OUT / 'warm_starts.npz'),
        'nodes': [str(node) for node in node_ids],
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
