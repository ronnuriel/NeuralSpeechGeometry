# Data directory

The repository does not redistribute the Kunz et al. recordings.

Place extracted `.mat` files in `data/raw/` or create symlinks there. Keep the original files immutable. `raw/`, `interim/`, and `processed/` are ignored by Git except for their placeholder files.

Source: [Dryad DOI 10.5061/dryad.gf1vhhn1j](https://doi.org/10.5061/dryad.gf1vhhn1j)

Recommended first download for attempted-versus-listening inference: `interleavedVerbalBehaviors.zip`. Use `isolatedVerbalBehaviors.zip` later for descriptive coverage of all seven behaviors.

Relevant interleaved files documented by the official analysis are:

- `t12.2024.04.11_interleavedVerbalBehaviors_raw.mat`
- `t15.2024.06.14_interleavedVerbalBehaviors_raw.mat` — default first target
- `t16.2024.07.17_interleavedVerbalBehaviors_raw.mat` — attempted **mimed**, not vocalized
- `t17.2024.12.09_interleavedVerbalBehaviors_raw.mat` — participant was anarthric

The isolated attempted/listening pairs are:

- `t12.2023.08.15_{attempted,listening}_raw.mat`
- `t15.2024.04.07_{attempted,listening}_raw.mat`
- `t16.2024.03.04_{attempted,listening}_raw.mat`
- `t17.2024.12.09_{attempted,listening}_raw.mat`

Before loading full arrays, run the audit notebook. For every source file, record:

- participant, session, and block;
- MATLAB version and top-level fields;
- neural feature name and units;
- axis order and sampling/bin rate;
- alignment-event field and trial time window;
- raw behavior labels and word labels;
- bad-channel or rejected-trial indicators; and
- SHA-256 checksum.

The canonical Python representation is documented in the root README and enforced by `NeuralDataset.validate()`.

## Expected public `.mat` schema

The Dryad README and official Stanford code document these fields; the audit must still verify each downloaded file:

| Field | Expected meaning |
|---|---|
| `binnedTX` | `time × feature`, threshold-crossing counts |
| `spikePow` | `time × feature`, spike-band power |
| `blockNum` | block identity for each continuous time bin |
| `goTrialEpochs` | trial go-period start/end indices |
| `delayTrialEpochs` | trial delay-period start/end indices |
| `trialCues` | cue ID for each trial |
| `cueList` | cue text/behavior labels |
| `chanSets`, `chanSetNames` | array/region channel mappings |
| `binSize` | bin duration in ms (observed values include 10 and 20) |

Loader guardrails:

- MATLAB cue, epoch, and channel indices are 1-based; convert exactly once.
- Epoch endpoints are inclusive in MATLAB.
- `cueList` can be a row or column cell array; normalize explicitly.
- Cast integer neural arrays to float before centering or smoothing.
- Exclude cue 1 / `DO_NOTHING` from the seven-word analysis.
- Expected words are `ban`, `choice`, `day`, `feel`, `kite`, `though`, and `were`.
- Use `binSize`; never infer timing from participant identity.
- Do not smooth across block boundaries.
- For isolated passive-listening files, the roughly 1.5 s audio stimulus can precede the stored go epoch. Verify the condition-specific event semantics instead of aligning every behavior naively to `goTrialEpochs`.
- T16's interleaved attempted condition is mimed, not vocalized.

Useful official reference: [`analysis/subfunctions/utils.py`](https://github.com/nptl-stanford/inner_speech/blob/main/analysis/subfunctions/utils.py).
