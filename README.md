# EAST Reflectometry Density-Profile Reconstruction

Physics-based reconstruction of an EAST electron-density profile from raw Q/V/W-band microwave-reflectometry I/Q signals, with PyTorch-accelerated spectral processing and CPU-side X-mode density inversion.

This repository contains the **single-time-point reference implementation** associated with the manuscript:

> **Millisecond-scale physics-based reconstruction of plasma density profiles with CPU–GPU heterogeneous acceleration on EAST**
> Yaling Zhang, Jia Huang, Chenguang Wan, *et al.*

> [!IMPORTANT]
> This is research code rather than a turnkey EAST data product.
>
> This repository does **not** include the original binary files (`Q_V_W.bin`) acquired by the EAST reflectometer. The bundled plasma HDF5 file contains only five preprocessed and aligned Q/V/W sweeps for one example time point. An associated vacuum-reference example is also included under `examples/data/`.
>
> Complete raw discharges, paper evaluation data, and the online acquisition/control-system integration are not distributed here. Confirm authorization for redistribution of the EAST-derived sample before public release.

## What this repository implements

For one EAST discharge and one requested time point, the package:

1. reads either an eight-channel binary acquisition segment or a preprocessed Q/V/W HDF5 sample;
2. aligns raw frequency sweeps using the trigger channel, or accepts five already aligned sweeps;
3. constructs or loads complex Q-, V-, and W-band I/Q signals;
4. performs batched sliding-window spectral estimation with PyTorch;
5. extracts beat frequency and spectral-width uncertainty using a weighted centroid;
6. subtracts vacuum references and converts beat frequency to group delay;
7. automatically selects a valid frequency interval using error-CUSUM and SNR criteria;
8. applies the physics-based X-mode inversion on the CPU; and
9. saves the reconstructed radial density profile as a MATLAB file.

The implementation supports CPU execution and automatically selects CUDA in `run_single_timepoint` when a compatible PyTorch/CUDA installation is available.

## Scope relative to the paper

The manuscript describes a complete CPU–GPU heterogeneous workflow, including batch processing and deployment in the EAST reflectometer system. This public package currently focuses on the **core, single-profile reconstruction path**.

Included here:

- raw Q/V/W I/Q loading for a selected time point;
- five-sweep spectral processing;
- GPU-capable beat-frequency extraction;
- automatic valid-interval screening;
- CPU-side density inversion and profile post-processing;
- MATLAB output and diagnostic plotting utilities;
- reusable MATLAB-vs-candidate profile evaluation, summary export, and accuracy plots; and
- a small EAST #151858 five-sweep HDF5 sample with its vacuum-reference data.

Not included here:

- EAST data acquisition and network transport;
- the online pipeline scheduler and control-system communication;
- the paper's large-scale batch benchmark scripts;
- the experimentally inverted reference profiles or AI comparison profiles;
- a full-discharge raw I/Q dataset or expected-output regression fixture;
- an automated end-to-end reproduction test; or
- the full online deployment configuration used during the 2026 EAST campaign.

## Method overview

```text
Raw eight-channel I/Q                 Preprocessed Q/V/W HDF5
        │                                      │
        ▼                                      │
Trigger alignment and five-sweep extraction   │
        │                                      │
        └──────────────────┬───────────────────┘
                           ▼
GPU-capable PyTorch processing
  ├─ 128-sample frequency windows
  ├─ 64-sample step (50% overlap)
  ├─ five sweeps arranged as a 5 × 128 sample window
  ├─ Welch PSD with 256-sample frames and a Hamming window
  └─ weighted-centroid beat frequency above Pmax / e
        │
        ▼
CPU processing
  ├─ vacuum-reference subtraction
  ├─ group-delay calculation
  ├─ Q/V/W frequency-region merging
  ├─ zero-density/cyclotron-frequency boundary
  ├─ error-CUSUM endpoint e1
  ├─ sub-band SNR endpoint e2
  ├─ final endpoint = min(e1, e2)
  └─ recursive physics-based X-mode inversion
        │
        ▼
Radial electron-density profile R–ne
```

### Parameters used in the manuscript configuration

| Parameter                   |                  Value | Meaning                                             |
| --------------------------- | ---------------------: | --------------------------------------------------- |
| Reflectometer bands         |                Q, V, W | Combined probing range of approximately 33–110 GHz |
| Sampling rate               |               62.5 MHz | Raw I/Q digitization rate                           |
| Sweep period                |                 50 μs | VCO sweep repetition period                         |
| Effective sweep             |                 40 μs | Samples used from each sweep                        |
| Sweeps per profile          |                      5 | Multi-period spectral estimate                      |
| Frequency-window length     |            128 samples | Sliding window along each sweep                     |
| Window step                 |             64 samples | 50% overlap                                         |
| Welch frame / FFT length    |            256 samples | Spectral estimation frame                           |
| Spectral threshold          |           `Pmax / e` | Bins retained for the weighted centroid             |
| CUSUM parameters            | `k = 14`, `h = 14` | Error-based endpoint detection                      |
| SNR contrast ratio          |               `0.12` | Adaptive SNR threshold interpolation                |
| Consecutive low-SNR windows |                  `3` | Persistent-degradation criterion                    |
| Zero-density radius         |                 2.37 m | Current inversion initialization value              |

The path and numerical defaults are defined in [`src/east_profile_recon/config.py`](src/east_profile_recon/config.py).

## Paper-reported results

The values below are reported by the manuscript and are **not automatically reproduced by the current repository without the corresponding EAST data and calibration files**.

### Reconstruction accuracy

The evaluation targets are density profiles produced by the established offline EAST DPR reconstruction workflow. Numerical X-mode inversion is followed by interpolation onto a common radial grid; these experimentally inverted profiles are used as references.

| EAST discharge | Profiles | Mean-density accuracy | Profile-wise R² | RMSE (`10^19 m^-3`) |
| -------------- | -------: | --------------------: | ---------------: | --------------------: |
| #155039        |    2,000 |      0.9905 ± 0.0215 | 0.9954 ± 0.0385 |      0.0905 ± 0.1158 |
| #156666        |    2,000 |      0.9904 ± 0.0089 | 0.9983 ± 0.0031 |      0.0323 ± 0.0224 |
| #158837        |    2,000 |      0.9914 ± 0.0076 | 0.9993 ± 0.0045 |      0.0138 ± 0.0112 |

For the independent endpoint-screening test, the conservative `min(e1, e2)` rule agreed with manual annotations within two frequency windows for **91.67%** of 120 samples from 40 previously unseen discharges.

For the Type-I ELM interval of EAST #155039 (`5.56–5.96 s`, 800 valid profiles), the paper reports:

- 98.77% average pedestal-height agreement;
- 99.16% average mean-density accuracy; and
- average profile-wise `R² = 0.9978`.

The AI comparison for EAST #155039 uses the original authors' R–`ne` predictions and the same interpolation rule as the proposed method and reference profiles.

### Timing boundary

The reported **approximately 3.5 ms** value is the time for core reconstruction from an already prepared five-period I/Q input to the output density profile. It does **not** include all engineering latency from acquisition, transmission, storage, or control-system communication.

On the paper's offline platform (AMD Ryzen 7 9700X, NVIDIA RTX 5060 Ti, Windows 11, PyTorch 2.8.0 with CUDA 12.8), the median single-profile timing after two warm-up runs was:

| Stage                               |             Median |      IQR |
| ----------------------------------- | -----------------: | -------: |
| Five-period I/Q extraction          |           0.483 ms | 0.021 ms |
| Host-to-device transfer             |           0.160 ms | 0.018 ms |
| GPU-side computation                |           0.919 ms | 0.103 ms |
| Device-to-host transfer             |           0.192 ms | 0.027 ms |
| CPU-side computation                |           2.203 ms | 0.082 ms |
| **Core reconstruction total** | **3.474 ms** |       — |

The paper reports an 8.75× speedup over the traditional CPU-serial implementation for one profile and approximately 20× for large batches. These values are hardware-, software-, data-, and benchmark-protocol-dependent.

## Repository layout

```text
.
├── .gitignore
├── LICENSE                         MIT License
├── README.md
├── pyproject.toml                  Build metadata and console entry points
├── requirements.txt               Minimum runtime dependencies
├── examples/
│   ├── data/
│   │   ├── README.md               Example-data schema and usage notes
│   │   ├── data_151858_5s.h5       Five preprocessed plasma sweeps
│   │   └── vacuum_data.h5          Vacuum beat-frequency references
│   ├── load_preprocessed_data.py   Read the bundled HDF5 example pair
│   └── run_single_timepoint.py     Bundled #151858 reconstruction example
└── src/
    └── east_profile_recon/
        ├── __init__.py             Explicit, lazy package-level exports
        ├── config.py               Paths and numerical defaults
        ├── core.py                 Spectral processing, screening, inversion
        ├── evaluation.py           Profile metrics, exports, plots, and CLI
        ├── plotting.py             Reconstruction diagnostic plots
        ├── profile_recon.py        Module entry-point wrapper
        └── data/
            ├── __init__.py
            ├── h5_loader.py        MAT/HDF5 calibration-data loaders
            └── iq_loader.py        Raw eight-channel binary I/Q loader
```

## Requirements

- Python 3.10 or newer
- NumPy 1.24 or newer
- SciPy 1.10 or newer
- h5py 3.8 or newer
- PyTorch 2.2 or newer
- Matplotlib 3.7 or newer

The repository currently specifies minimum versions rather than a lock file. For scientific reproduction, record the exact Python, PyTorch, CUDA, NumPy, SciPy, h5py, GPU driver, and hardware versions used in your environment.

CUDA is optional. Install the correct CPU or CUDA PyTorch build for your platform before installing this package.

## Installation

From the repository root:

```bash
python -m pip install -e .
```

If a platform-specific PyTorch build and the remaining dependencies are already installed:

```bash
python -m pip install -e . --no-deps
```

Basic import:

```python
import east_profile_recon
```

> [!NOTE]
> The reconstruction plotting module selects the Matplotlib `TkAgg` backend during import, so a Tk-capable environment may be required for reconstruction even when figures are not displayed. Package exports are lazy: importing `east_profile_recon.data` for HDF5 loading does not import PyTorch or Matplotlib. See [Known limitations](#known-limitations).

## Configuration and paths

The defaults in [`src/east_profile_recon/config.py`](src/east_profile_recon/config.py) remain available for backward compatibility:

| Setting                      | Purpose                               |
| ---------------------------- | ------------------------------------- |
| `DEFAULT_DATA_ROOT`        | Root containing `{shot}/Q_V_W.bin`  |
| `DEFAULT_VACUUM_DATA_PATH` | Vacuum beat-frequency HDF5 file       |
| `DEFAULT_IT_PATH`          | Optional shot-to-IT lookup MAT file   |
| `DEFAULT_QVW_DATA_PATH`    | Optional preprocessed Q/V/W HDF5 file |
| `DEFAULT_OUTPUT_ROOT`      | MATLAB output root                    |

Editing these defaults is no longer required. The raw-data root, a direct raw I/Q file, a preprocessed Q/V/W HDF5 file, vacuum-reference file, output root, and computation device can be supplied through either the Python API or command-line options. A supplied `qvw_path`/`--qvw-file` bypasses raw extraction; `iq_file` and `qvw_path` are mutually exclusive.

## Input data

### 1. Raw Q/V/W binary

Expected path:

```text
{DEFAULT_DATA_ROOT}/{shot}/Q_V_W.bin
```

The current loader interprets the file as interleaved eight-channel `int16` samples:

```text
Q = channel 0 + j × channel 1
V = channel 2 + j × channel 3
W = channel 4 + j × channel 5
trigger = channel 6
```

Channel 7 is not used by the reconstruction. The current implementation uses NumPy's native `int16` byte order; the acquisition file must match the executing platform's interpretation.

The loader reads ten sweep periods around the requested time, finds the trigger maximum, and extracts five effective sweeps. Input files should therefore contain sufficient data after the selected trigger position.

### 2. Vacuum-reference HDF5

The file at `DEFAULT_VACUUM_DATA_PATH` must contain:

```text
fb_Q
fb_err_Q
fb_V
fb_err_V
fb_W
fb_err_W
```

Each vacuum array must align one-to-one with the sliding frequency windows produced for its band. Under the current 2,500-sample effective sweep, 128-sample window, and 64-sample step, the implementation produces 38 windows per band. The bundled [`examples/data/vacuum_data.h5`](examples/data/vacuum_data.h5) contains these six 38-value references plus intermediate diagnostic spectra.

### 3. Optional shot-to-IT MAT file

`get_IT` expects a variable named `data`:

```text
data[0, :]  EAST shot numbers
data[1, :]  corresponding IT values
```

The default reconstruction does not call `get_IT`; `IT` is supplied directly to `run_single_timepoint`.

### 4. Preprocessed plasma HDF5

`load_QVWdata` directly reads aligned plasma sweep data from an HDF5 file containing:

| Dataset        | Required form                            |
| -------------- | ---------------------------------------- |
| `t_point_np` | Two-dimensional sample-time array        |
| `Q_point_np` | Complex Q-band array with the same shape |
| `V_point_np` | Complex V-band array with the same shape |
| `W_point_np` | Complex W-band array with the same shape |

The loader verifies that all four datasets exist, are two-dimensional, and have identical shapes. The three signal arrays must be complex-valued.

The bundled [`examples/data/data_151858_5s.h5`](examples/data/data_151858_5s.h5) contains five 2,500-sample sweeps from EAST #151858 at approximately 5.0 s. Its `t_point_np` values are in milliseconds and its Q/V/W arrays are `complex64`:

```python
from east_profile_recon.data import load_QVWdata

time_ms, q_data, v_data, w_data = load_QVWdata(
    "examples/data/data_151858_5s.h5"
)
```

This loader bypasses raw-binary extraction and trigger alignment. Pass the file through `qvw_path` in the Python API or `--qvw-file` on the CLI to run the downstream spectral processing and inversion directly from these five sweeps.

## Quick start

### Python API

The bundled example can be reconstructed without a separate raw `Q_V_W.bin` file:

```python
from east_profile_recon.core import run_single_timepoint

output_file = run_single_timepoint(
    shot=151858,
    IT=10735,
    time_point=5.0,
    qvw_path="examples/data/data_151858_5s.h5",
    vacuum_path="examples/data/vacuum_data.h5",
    output_root="outputs",
    device=None,          # CUDA if available; otherwise CPU
)

print(output_file)
```

For a complete raw discharge, supply either `data_root` or a direct `iq_file` instead of `qvw_path`.

### Bundled examples

Inspect the HDF5 arrays without running reconstruction:

```bash
python examples/load_preprocessed_data.py
```

Run the downstream spectral processing and density inversion using the bundled EAST #151858 sample, `IT = 10735`, and `t = 5.0 s`:

```bash
python examples/run_single_timepoint.py
```

Both examples resolve their data paths relative to the `examples/` directory and therefore do not require environment variables. See [`examples/data/README.md`](examples/data/README.md) for the complete schema and checksums.

## Command-line interface

After installation, inspect all options with:

```bash
east-profile-recon --help
```

Run the bundled preprocessed example:

```bash
east-profile-recon \
  --shot 151858 \
  --it 10735 \
  --time 5.0 \
  --qvw-file "examples/data/data_151858_5s.h5" \
  --vacuum-file "examples/data/vacuum_data.h5" \
  --output-root "outputs" \
  --device auto
```

For complete raw data, use the conventional `{data-root}/{shot}/Q_V_W.bin` layout:

```bash
east-profile-recon \
  --shot 151858 \
  --it 10735 \
  --time 5.0 \
  --data-root "/path/to/IQ_data" \
  --vacuum-file "/path/to/vacuum.h5" \
  --output-root "outputs" \
  --device auto
```

Or provide the raw binary directly with `--iq-file`. `--iq-file` and `--qvw-file` are mutually exclusive.

The equivalent module entry point is:

```bash
python -m east_profile_recon.profile_recon --help
```

Available device values are `auto`, `cpu`, and `cuda`. `auto` selects CUDA when `torch.cuda.is_available()` is true and otherwise uses the CPU. Requesting `cuda` when CUDA is unavailable raises an explicit error.

The CLI defaults to EAST #151858, `IT = 10735`, and `t = 5.0 s`. Machine-specific raw-data, vacuum, and output path defaults remain available from `config.py`, but explicit paths are recommended for portable and reproducible runs.

## Output

The MATLAB file is written to:

```text
{output_root}/{shot}/{shot}_{time_point:.3f}s.mat
```

| MAT key  | Description                                                |
| -------- | ---------------------------------------------------------- |
| `shot` | EAST discharge number                                      |
| `IT`   | Current-related inversion parameter supplied by the caller |
| `time` | Requested reconstruction time in seconds                   |
| `R`    | Reconstructed major-radius coordinates in metres           |
| `Ne`   | Electron density in units of `10^19 m^-3`                |

`run_single_timepoint` returns the generated file path.

For in-memory inspection, `main_worker` returns a nested dictionary containing the final profile, processed 108-position arrays, and extracted Q/V/W I/Q data. It accepts the same `data_root`, `iq_file`, `qvw_path`, `vacuum_path`, and `device` controls as `run_single_timepoint`; its automatic device selection also falls back to the CPU when CUDA is unavailable.

## Accuracy evaluation

[`src/east_profile_recon/evaluation.py`](src/east_profile_recon/evaluation.py) provides reusable profile-series evaluation independently of the reconstruction command. It replaces the former path-specific analysis script with a library API and the `east-profile-evaluate` command.

### Evaluation input format

Both inputs must be MATLAB files readable by `scipy.io.loadmat`. Each selected variable must be a one-dimensional MATLAB structure array whose records contain:

| Structure field | Description                                                                                         |
| --------------- | --------------------------------------------------------------------------------------------------- |
| `nwidth`      | Frame-validity flag; a record is eligible when this value is finite and non-negative in both inputs |
| `R_fit`       | One-dimensional radial grid                                                                         |
| `ne_fit`      | Electron density on `R_fit`, normally in `10^19 m^-3`                                           |

For example, the expected MATLAB-side organization is equivalent to:

```matlab
profiles(k).nwidth
profiles(k).R_fit
profiles(k).ne_fit
```

Reference and candidate records are paired by array index. The evaluator uses the shorter series when the files contain different numbers of records and, by default, evaluates at most the first 2,000 eligible frames. If a variable name is not supplied, the first public variable in the MAT file is selected; specifying both names is recommended for reproducibility.

The evaluator deliberately does **not** interpolate. After removal of non-finite and non-positive density points, the two profiles must:

- have the same number of radial and density samples;
- contain at least 10 common valid points by default;
- use the same radial grid within an absolute tolerance of `1e-8`; and
- have a strictly monotonic radius coordinate.

Frames that fail these checks are skipped and counted by reason in the JSON summary. This fixed-grid requirement matches the manuscript evaluation, where profiles were interpolated onto a common radial grid before metric calculation. The per-profile reconstruction MAT file described in [Output](#output) contains `R` and `Ne`; convert a series of those outputs to the structure format above, or call the array-level API directly, before using this MAT-file evaluator.

### Metrics

For reference density `n_i`, candidate density `n̂_i`, and valid radial samples `R_i`, the evaluator reports:

| Metric                  | Definition                                                              |
| ----------------------- | ----------------------------------------------------------------------- |
| Mean-density accuracy   | `1 -                                                                    |
| Profile-wise R²        | `1 - Σ(n̂_i - n_i)² / Σ(n_i - mean(n))²`                         |
| RMSE                    | `sqrt(mean((n̂_i - n_i)²))`, in the same density unit as `ne_fit` |
| Mean relative error     | `mean(                                                                  |
| NIAE                    | `∫                                                                     |
| Maximum relative error  | `max(                                                                   |
| Mean pointwise accuracy | `mean(1 -                                                               |

Accuracy values are not clipped to `[0, 1]`; a strongly inaccurate candidate can therefore produce a negative accuracy. R² is recorded as `NaN` when the reference profile is effectively constant.

### Evaluation CLI

Inspect all options with:

```bash
east-profile-evaluate --help
```

Example comparison of an offline MATLAB reference series and a Python candidate series:

```bash
east-profile-evaluate \
  "/path/to/reference_profiles.mat" \
  "/path/to/candidate_profiles.mat" \
  --reference-variable profiles \
  --candidate-variable profiles \
  --max-profiles 2000 \
  --time-start 5.0 \
  --time-interval 0.0005 \
  --event-time 5.713 \
  --event-time 5.723 \
  --event-time 5.745 \
  --reference-label "Offline reference" \
  --candidate-label "Python reconstruction" \
  --output-dir evaluation_outputs
```

`--time-start` and `--time-interval` map record indices to plot times. Repeat `--event-time` to add event markers. Use `--no-plots` for data-only output or `--show` to display the generated figures after saving them.

Each run prints the primary metric summary and writes:

| File                           | Contents                                                                         |
| ------------------------------ | -------------------------------------------------------------------------------- |
| `evaluation_summary.json`    | Input metadata, frame counts, skipped-frame reasons, and summary statistics      |
| `evaluation_metrics.npz`     | Time, selected indices, per-frame metric arrays, and pooled valid density points |
| `accuracy_time_series.png`   | Profile-wise R² and mean-relative-error time series in separate panels          |
| `accuracy_distributions.png` | Metric histograms with density estimates when the data support them              |
| `accuracy_cdf.png`           | Mean-relative-error cumulative distribution with the 90th percentile             |
| `accuracy_scatter.png`       | Pooled reference-versus-candidate density scatter and 1:1 line                   |

Generated evaluation data and figures default to `evaluation_outputs/`, which is excluded by `.gitignore`.

### Evaluation Python API

```python
from east_profile_recon.evaluation import (
    evaluate_mat_files,
    save_evaluation_data,
    save_evaluation_plots,
)

result, metadata = evaluate_mat_files(
    "reference_profiles.mat",
    "candidate_profiles.mat",
    reference_variable="profiles",
    candidate_variable="profiles",
    max_profiles=2000,
    time_start=5.0,
    time_interval=0.0005,
)

output_dir = save_evaluation_data(result, "evaluation_outputs", metadata)
save_evaluation_plots(
    result,
    output_dir,
    reference_label="Offline reference",
    candidate_label="Python reconstruction",
    event_times=(5.713, 5.723, 5.745),
)

print(result.summary())
```

Use `evaluate_profile_series` directly when the two structure arrays are already loaded in memory.

## API guide

### Primary reconstruction interface

| Function                 | Module                      | Purpose                                            |
| ------------------------ | --------------------------- | -------------------------------------------------- |
| `run_single_timepoint` | `east_profile_recon.core` | Reconstruct and save one profile                   |
| `main_worker`          | `east_profile_recon.core` | Run one reconstruction without saving the MAT file |

### Evaluation interface

| Function                    | Module                            | Purpose                                                  |
| --------------------------- | --------------------------------- | -------------------------------------------------------- |
| `evaluate_profile_series` | `east_profile_recon.evaluation` | Evaluate two in-memory profile structure arrays          |
| `evaluate_mat_files`      | `east_profile_recon.evaluation` | Load and evaluate reference and candidate MAT files      |
| `save_evaluation_data`    | `east_profile_recon.evaluation` | Save compressed per-frame arrays and a JSON summary      |
| `save_evaluation_plots`   | `east_profile_recon.evaluation` | Save time-series, distribution, CDF, and scatter figures |

### Data utilities

| Function             | Module                                | Purpose                                        |
| -------------------- | ------------------------------------- | ---------------------------------------------- |
| `read_QVWdata`     | `east_profile_recon.data.iq_loader` | Read and align raw eight-channel I/Q data      |
| `load_vacuum_data` | `east_profile_recon.data.h5_loader` | Load configured vacuum references              |
| `get_IT`           | `east_profile_recon.data.h5_loader` | Look up IT from the configured MAT file        |
| `load_QVWdata`     | `east_profile_recon.data.h5_loader` | Load preprocessed plasma time and Q/V/W arrays |

### Reconstruction plotting utilities

- `plot_iq_data`
- `plot_spectrogram`
- `plot_strategy_comparison`
- `plot_all_bands`
- `plot_density_profile`

These functions are intended for interactive diagnostics. The current implementation uses `TkAgg` and calls `plt.show()`.

## Reproducibility checklist

To make a result traceable, record at least:

- repository version or archive checksum;
- EAST discharge number, time point, and IT value;
- raw I/Q or preprocessed plasma HDF5 identity and checksum;
- vacuum-reference file identity, calibration date, and checksum;
- all values in `config.py`;
- Python and dependency versions;
- CPU, GPU, CUDA runtime, and driver versions;
- whether CPU or CUDA was selected;
- warm-up count, repeat count, synchronization method, and timing boundary;
- output MAT checksum and the reference-profile provenance;
- evaluation reference/candidate MAT checksums and selected variable names; and
- evaluation frame limit, validity thresholds, radial tolerance, and time mapping.

The EAST data used in the manuscript belong to the EAST team and may be requested from the corresponding authors subject to the applicable data-access conditions.

## Known limitations

- Only one five-sweep plasma sample and one vacuum-reference file are included; full raw discharges, paper evaluation datasets, and expected-output fixtures are not distributed.
- Default paths in `config.py` remain machine-specific; portable runs should pass explicit API or CLI paths.
- The reconstruction CLI covers one time point per invocation and does not yet support batch manifests or external configuration files.
- Evaluation operates on completed profile series; it does not run or schedule batch reconstruction.
- The MAT-file evaluator expects `nwidth`/`R_fit`/`ne_fit` structure arrays readable by `scipy.io.loadmat`; MATLAB v7.3 HDF5 files and arbitrary schemas require conversion or a custom loader.
- There is no automated test suite, continuous integration, or locked reproduction environment.
- HDF5 dataset presence and basic shape consistency are validated, but raw binary size, trigger position, finite values, calibration provenance, and frequency monotonicity are not comprehensively checked before numerical processing.
- Reconstruction plotting still selects the interactive `TkAgg` backend when the plotting module is imported; data-only imports now remain independent of Matplotlib and PyTorch.
- The endpoint and frequency-mask parameters were calibrated for the EAST DPR configuration and require validation or recalibration for other hardware, sweep ranges, or noise conditions.
- The reported 3.5 ms timing covers core computation only, not complete closed-loop latency.
- Severe turbulence or spectral degradation can cause premature endpoint selection; the current public output does not include a calibrated confidence/validity flag.

Before using this code for quantitative scientific conclusions, validate the numerical pipeline with known synthetic signals and representative benchmark profiles from the target diagnostic configuration.

## Citation

If this repository contributes to published work, cite the associated manuscript. Until final bibliographic information is available, use the manuscript title and authors shown at the top of this README and update the entry after publication.

Suggested temporary BibTeX:

```bibtex
@unpublished{zhang2026millisecond,
  title  = {Millisecond-scale physics-based reconstruction of plasma density profiles with CPU--GPU heterogeneous acceleration on EAST},
  author = {Zhang, Yaling and Huang, Jia and Wan, Chenguang and others},
  year   = {2026},
  note   = {Manuscript}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
