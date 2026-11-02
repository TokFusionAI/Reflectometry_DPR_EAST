from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

from ..config import DEFAULT_IT_PATH, DEFAULT_QVW_DATA_PATH, DEFAULT_VACUUM_DATA_PATH


VACUUM_DATASETS = (
    "fb_Q",
    "fb_err_Q",
    "fb_V",
    "fb_err_V",
    "fb_W",
    "fb_err_W",
)
QVW_DATASETS = ("t_point_np", "Q_point_np", "V_point_np", "W_point_np")


def _require_file(file_path, description):
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _read_datasets(file_path, dataset_names, description):
    path = _require_file(file_path, description)
    with h5py.File(path, "r") as h5_file:
        missing = [name for name in dataset_names if name not in h5_file]
        if missing:
            available = sorted(h5_file.keys())
            raise KeyError(
                f"Missing datasets in {path}: {missing}. "
                f"Available datasets: {available}"
            )
        return tuple(h5_file[name][:] for name in dataset_names)


def get_IT(shot, mat_file_path=DEFAULT_IT_PATH):
    mat_path = _require_file(mat_file_path, "IT lookup MAT file")
    mat_data = sio.loadmat(mat_path)
    data = mat_data["data"]
    mask = data[0, :] == shot
    if np.any(mask):
        IT_value = data[1, mask]
        IT_scalar = IT_value[0].item() if IT_value.size > 1 else IT_value.item()
        return int(IT_scalar)
    raise ValueError(f"No IT lookup value found for shot {shot}")


def load_vacuum_data(h5_file_path=DEFAULT_VACUUM_DATA_PATH):
    """Load Q/V/W vacuum beat-frequency references and their uncertainties."""
    arrays = _read_datasets(
        h5_file_path,
        VACUUM_DATASETS,
        "Vacuum-reference HDF5 file",
    )
    lengths = {array.size for array in arrays}
    if len(lengths) != 1 or any(array.ndim != 1 for array in arrays):
        shapes = dict(zip(VACUUM_DATASETS, (array.shape for array in arrays)))
        raise ValueError(
            "Vacuum-reference datasets must be one-dimensional and have equal "
            f"lengths; received {shapes}"
        )
    return arrays


def load_QVWdata(h5_file_path=DEFAULT_QVW_DATA_PATH):
    """Load aligned time and preprocessed five-sweep Q/V/W plasma arrays."""
    time_data, q_data, v_data, w_data = _read_datasets(
        h5_file_path,
        QVW_DATASETS,
        "Preprocessed Q/V/W HDF5 file",
    )
    arrays = (time_data, q_data, v_data, w_data)
    shapes = dict(zip(QVW_DATASETS, (array.shape for array in arrays)))
    if any(array.ndim != 2 for array in arrays) or len(set(shapes.values())) != 1:
        raise ValueError(
            "Preprocessed time and Q/V/W datasets must be two-dimensional and "
            f"share one shape; received {shapes}"
        )
    if not all(np.iscomplexobj(array) for array in (q_data, v_data, w_data)):
        raise ValueError("Q_point_np, V_point_np, and W_point_np must be complex arrays")
    return time_data, q_data, v_data, w_data
