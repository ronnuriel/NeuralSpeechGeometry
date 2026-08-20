# First real-data report

## What was downloaded

Source: [Dryad DOI 10.5061/dryad.gf1vhhn1j](https://doi.org/10.5061/dryad.gf1vhhn1j)

Archive: `interleavedVerbalBehaviors.zip`

- Size: 777,419,058 bytes
- SHA-256: `19f90f09f2ea32f1428b7cc1c7dd8c0606dfbc988c57fcaf64aea03e77b9d409`
- ZIP integrity test: passed
- License: CC0
- Local location: `data/raw/downloads/` (ignored by Git)

The interleaved archive is the appropriate first dataset for a direct attempted-versus-listening comparison because cue types occur within the same recording blocks. This reduces the block/time confound present in the isolated-behavior archive.

## Archive inventory

| Participant | Session | Trials | Neural channels | Bin | Relevant behaviors |
|---|---|---:|---:|---:|---|
| T12 | 2024-04-11 | 440 | 128 | 20 ms | passive listening, attempted vocalized, motoric imagined |
| T15 | 2024-06-14 | 638 | 256 | 10 ms | attempted vocalized, imagined listening, passive listening |
| T16 | 2024-07-17 | 440 | 256 | 20 ms | auditory imagined, attempted **mimed**, listening |
| T17 | 2024-12-09 | 551 | 256 | 10 ms | attempted speech, imagined listening, passive listening |

Every file contains continuous `binnedTX` and `spikePow` features, `blockNum`, go and delay epochs, trial cue IDs and text, channel-set metadata, and the true bin size. The cue-list orientation differs across files, and MATLAB indices are 1-based; the adapter normalizes both explicitly.

## First executed analysis: T15, i6v, threshold crossings

The first run deliberately uses one participant, one neural feature, and one array:

- Participant/session: T15, 2024-06-14
- Conditions: attempted vocalized speech versus passive listening
- Words: `ban`, `choice`, `day`, `feel`, `kite`, `though`, `were`
- Trials: 308 total; 22 trials per word per condition
- Array: `i6v`, 64 channels
- Feature: threshold-crossing counts (`binnedTX`)
- Epoch: -500 to 990 ms relative to behavior onset
- Preprocessing: continuous block-mean subtraction, within-trial 60 ms Gaussian smoothing, trial/channel baseline subtraction from -500 to -100 ms
- Fixed response window: 0 to 500 ms
- Shared trajectory PCA: one pooled scaler and one PCA fitted with 154 trials from each condition
- Word visualization PCA: one symmetric basis fitted to the 14 equally weighted condition-by-word centroids
- Geometry statistics: computed in the full pooled-standardized 64-channel space, not only in the plotted PCs

Quality checks passed: no missing/infinite values, no constant selected channels, all seven words are balanced, and both conditions occur in every analyzed block.

## Initial observations

These are exploratory results, not a final biological conclusion.

1. The condition-average trajectories separate strongly after behavior onset in the shared PC space. This means the selected motor-cortical population has different average dynamics during attempted production and passive perception under this preprocessing.
2. The full-space condition-centroid distance is 2.873 standardized units. A whole-trial permutation restricted within `session × block × word` gave `p = 0.001996` with 500 permutations and 35 exchangeable strata. This is the smallest attainable non-zero p-value with the current permutation count, so the next run should use more permutations and a cross-validated distance.
3. The within-condition seven-word geometries have a moderate Spearman relationship (`rho = 0.432`). Thus, the two conditions are not simply identical word maps shifted by one global offset.
4. The first 10 trajectory PCs explain 33.4% of pooled variance. The three-dimensional condition subspaces have large principal angles (63.5°–84.6°), consistent with different dominant variance orientations.

The result does **not** isolate “speech intent.” Attempted production and listening also differ in motor commands, auditory input, task demands, timing, and overall neural gain. PCA is descriptive, and the current Euclidean distance is not yet cross-validated.

## What to do next

1. Repeat T15 with `spikePow` as a separate feature analysis.
2. Increase the permutation count and add cross-validated Mahalanobis distance.
3. Add leave-one-block-out and leave-one-word-out sensitivity checks.
4. Repeat the exact pipeline on T12, carefully preserving its special listening alignment at `delayTrialEpochs`.
5. Compare `i6v` with `55b` rather than pooling anatomical arrays immediately.
6. Add the paper-style attempted-fit PCA as a sensitivity view; keep the symmetric 14-centroid PCA as the primary two-condition visualization.

## Reproduce locally

Run the source notebook with the real-data configuration:

```bash
KUNZ_CONFIG=configs/t15_interleaved_binnedtx.yaml \
  jupyter nbconvert --to notebook --execute \
  notebooks/01_attempted_vs_passive_shared_pca.ipynb \
  --output t15_interleaved_binnedtx_executed.ipynb
```

The raw archive and extracted `.mat` files stay outside Git. The adapter, configuration, tests, and this report are versioned.
