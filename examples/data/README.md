# Bundled example data

This directory contains a minimal, single-time-point input pair for checking the HDF5 loaders and developing the spectral-processing workflow without access to a complete EAST discharge file.

| File | SHA-256 |
| ---- | ------- |
| `data_151858_5s.h5` | `aa8b0ee58eadc42f31d97b12ed304f8b594a3176a9459097b142316a32567efc` |
| `vacuum_data.h5` | `4c7299a3275318529347ccc29549bdd8cbaba728f341a40a3d14997a73e38ec2` |

## Files

### `data_151858_5s.h5`

Preprocessed plasma reflectometry data for EAST discharge **#151858** at approximately **5.0 s**.

File attributes:

| Attribute | Value |
| --------- | ----- |
| `shot` | `151858` |
| `time_point_s` | `5.0` |

Datasets:

| Dataset | Shape | Type | Description |
| ------- | ----- | ---- | ----------- |
| `t_point_np` | `(5, 2500)` | `float64` | Sample times in milliseconds |
| `Q_point_np` | `(5, 2500)` | `complex64` | Five effective Q-band sweeps |
| `V_point_np` | `(5, 2500)` | `complex64` | Five effective V-band sweeps |
| `W_point_np` | `(5, 2500)` | `complex64` | Five effective W-band sweeps |

Load it with:

```python
from east_profile_recon.data import load_QVWdata

time_ms, q_data, v_data, w_data = load_QVWdata(
    "examples/data/data_151858_5s.h5"
)
```

### `vacuum_data.h5`

Vacuum beat-frequency reference data for the 38 sliding windows produced in each band by the current manuscript configuration. `load_vacuum_data` reads:

```text
fb_Q
fb_err_Q
fb_V
fb_err_V
fb_W
fb_err_W
```

The file also retains intermediate spectral arrays for inspection, but they are not returned by the loader.

## Run the examples

From the repository root, after installing the package, inspect the arrays with:

```bash
python examples/load_preprocessed_data.py
```

Run the downstream reconstruction for EAST #151858 with `IT = 10735` and `t = 5.0 s` using:

```bash
python examples/run_single_timepoint.py
```

These files are intentionally small examples, not the full paper evaluation dataset and not an expected-output regression fixture. Confirm that redistribution of the EAST-derived sample and its vacuum calibration is authorized before publishing the repository.
