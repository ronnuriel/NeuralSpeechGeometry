# Kunz Neural Speech Geometry

Reproducible exploratory analysis of the public Kunz et al. intracortical speech dataset, beginning with **attempted vocalized speech versus passive listening**.

The first milestone asks:

> When both conditions are represented in the same low-dimensional neural space, how do their population trajectories and word-level geometries differ?

## Status

- The repository and analysis code are ready.
- The Dryad `interleavedVerbalBehaviors.zip` archive has been downloaded, checksum-verified, extracted, and audited locally.
- The first notebook runs end to end on deterministic synthetic data and on the real T15 interleaved session.
- The tested `.mat` adapter handles MATLAB indexing, row/column cue lists, channel sets, block centering, and T12's condition-specific listening alignment.
- See [REAL_DATA_REPORT.md](REAL_DATA_REPORT.md) for the first T15/i6v result and its limitations.
- Raw data are never committed to Git.

## Why one shared PCA?

The primary geometry plot has one point per word and condition. Each point is a population vector averaged across a fixed response window and repeated trials. We pool all 14 equally weighted centroids, standardize every neural feature with the **same** scaler, and fit **one** PCA basis:

```text
2 conditions x 7 word centroids x neural features
                         |
                   shared scaler
                         |
                     one PCA
                         |
          attempted and listening words
          in the same coordinate system
```

This makes visual distances meaningful: both conditions use identical axes. Separate PCAs would create independently rotated coordinate systems that cannot be compared directly.

A secondary time-resolved PCA uses pooled `trial × time` observations as rows and neural features as columns. It visualizes condition trajectories in another single shared basis. It is not the inferential sample space: whole trials remain the sampling unit.

PCA is descriptive. Separation in a plot does not, by itself, establish decoding performance, causality, or statistical significance. It can also reflect condition-average amplitude, timing, session drift, or block order.

### A different PCA answers a different question

An electrode-as-point analysis would use `channels x time` and ask which electrodes have similar temporal profiles. That is useful as a secondary channel-organization analysis, but it is not the primary population-dynamics analysis in this repository.

## Critical design limitation and recommended archive

In `isolatedVerbalBehaviors`, behaviors were collected in separate blocks. Intracortical recordings can drift between blocks, so attempted-versus-listening differences are **exploratory and potentially confounded by block/time**. We therefore:

1. analyze each participant separately;
2. preserve session, block, word, and trial labels;
3. resample and permute whole trials rather than treating time bins as independent;
4. stratify comparisons by session and word when the data permit; and
5. use `interleavedVerbalBehaviors`, where cue types were intermixed within blocks, for the primary direct condition contrast.

For this exact question, the recommended first real-data target is therefore `interleavedVerbalBehaviors.zip`. It contains attempted, imagined, and listening trials randomly interleaved within blocks. The attempted condition was vocalized for T12 and T15, mimed for T16, and described as attempted speech for T17; the adapter must preserve that distinction rather than merging all four participants under one label. The default participant is T15.

## Dataset

Primary source: [Kunz et al., *Inner speech in motor cortex and implications for speech neuroprostheses*](https://doi.org/10.1016/j.cell.2025.06.015), Cell (2025).

Public data: [Dryad dataset, DOI 10.5061/dryad.gf1vhhn1j](https://doi.org/10.5061/dryad.gf1vhhn1j). The full release is about 10.8 GB. `interleavedVerbalBehaviors.zip` is about 777 MB and is preferred for the direct contrast; `isolatedVerbalBehaviors.zip` is about 3.53 GB and contains all seven separately blocked behaviors for T12, T15, T16, and T17.

Official analysis code: [nptl-stanford/inner_speech](https://github.com/nptl-stanford/inner_speech). It confirms fields such as `binnedTX`, `spikePow`, `trialCues`, `cueList`, `goTrialEpochs`, and `blockNum`; these names will still be verified against the downloaded files before an adapter is enabled.

The paper's Figure 2 analysis used smoothed threshold-crossing and spike-band-power features, 500 ms behavior-specific windows, and word-level neural vectors. It fitted the display basis on the seven attempted-vocalized word vectors and projected other behaviors into that basis. See the authors' [concise Figure 2 method](https://bio-protocol.org/exchange/minidetail?id=21668723&type=30). This repository's symmetric two-condition view instead fits its display basis to all 14 word centroids; the attempted-fit version is a planned paper-style sensitivity analysis.

## Repository map

```text
configs/default.yaml                         analysis choices
configs/t15_interleaved_binnedtx.yaml        first real-data run
data/README.md                               data placement and contract
data/raw/                                    immutable .mat files (ignored)
notebooks/00_mat_file_audit.ipynb            inspect real files without assumptions
notebooks/01_attempted_vs_passive_shared_pca.ipynb
REAL_DATA_REPORT.md                          audit and first-result summary
src/kunz_speech_geometry/                    reusable loading and analysis code
tests/                                       synthetic unit and smoke tests
results/figures/                             generated figures (ignored)
results/tables/                              generated summaries (ignored)
```

## Quick start

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m ipykernel install --user --name kunz-geometry --display-name "Kunz geometry"
jupyter lab
```

Open `notebooks/01_attempted_vs_passive_shared_pca.ipynb` and run all cells. With the default configuration it uses synthetic data, so collaborators can inspect the full workflow immediately.

To execute the same notebook on the downloaded T15 data:

```bash
KUNZ_CONFIG=configs/t15_interleaved_binnedtx.yaml \
  jupyter nbconvert --to notebook --execute \
  notebooks/01_attempted_vs_passive_shared_pca.ipynb \
  --output t15_interleaved_binnedtx_executed.ipynb
```

Run the checks with:

```bash
pytest
```

## Connecting the `.mat` files

1. For the direct condition comparison, download and extract `interleavedVerbalBehaviors.zip` outside Git. Use `isolatedVerbalBehaviors.zip` for the broader seven-behavior exploratory analysis.
2. Copy or symlink only the files needed for one participant into `data/raw/`.
3. Run `notebooks/00_mat_file_audit.ipynb` (or `kunz-audit-mat data/raw/file.mat`).
4. Use `load_interleaved_mat()` with an explicit feature, channel set, and epoch; the included T15 configuration is the reference example.
5. Keep `configs/default.yaml` in synthetic mode so the repository remains runnable without the large source files.

Do not silently infer axis order. MATLAB arrays often squeeze singleton dimensions, and files saved as MATLAB v7.3 require HDF5-aware loading.

## Canonical in-memory contract

Every real-data adapter must return a `NeuralDataset` with:

| Field | Shape / columns | Meaning |
|---|---|---|
| `rates` | `[trial, time, feature]` | One documented neural feature type and unit |
| `time_ms` | `[time]` | Time relative to a named alignment event |
| `trials` | one row per trial | `participant`, `session`, `block`, `trial_id`, `condition`, `word` |
| `channels` | one row per feature/channel | `channel_id`, array/region metadata when available |

Raw condition labels must be preserved alongside canonical labels. Threshold-crossing rates and spike-band power should be analyzed separately before any justified feature concatenation.

## Notebook 01: analysis plan

1. Validate shapes, finite values, timing, units, labels, and trial balance.
2. Plot trial counts and population activity before dimensionality reduction.
3. Smooth and baseline-correct each trial using the configured windows.
4. Fit one pooled channel scaler; use balanced trials for the secondary trajectory PCA.
5. Inspect explained variance and channel loadings.
6. Fit the primary shared PCA to 14 equally weighted condition-by-word centroids and plot matched words.
7. Quantify time-resolved centroid distance, word-geometry correlation, and principal angles.
8. Use whole-trial, session-and-word-stratified permutations for an initial null test.
9. Repeat across preprocessing choices and PC counts before interpreting the geometry.
10. Save parameters and summaries beside generated results.

## Interpretation checklist

- Is a condition difference present before PCA in population firing-rate summaries?
- Does it persist after within-session preprocessing and condition-balanced PCA fitting?
- Is it driven by a small number of channels, a single session, or one word?
- Does it survive leave-one-session-out and leave-one-word-out sensitivity analyses?
- Do conclusions agree across threshold crossings and spike-band power?
- Is the direct condition effect estimated in the interleaved task rather than attributed to separate isolated blocks?

## Near-term roadmap

- Implement the exact Kunz `.mat` adapter after auditing one representative file.
- Add the authors' attempted-fit PCA as a paper-aligned sensitivity view.
- Add session/block drift diagnostics and leave-one-block/session-out sensitivity checks.
- Add cross-validated Mahalanobis distances and bootstrap confidence intervals.
- Add the isolated seven-behavior descriptive analysis as a separate notebook.

## Reproducibility

- Synthetic data and permutations use a fixed seed.
- Raw data and generated outputs are ignored by Git.
- Analysis parameters live in versioned YAML.
- Reusable logic lives in `src/`, not only in notebooks.
- Statistical resampling uses trials as the independent unit.

The Dryad data are released under CC0. A code license for this repository should be selected by the research group before public release.
