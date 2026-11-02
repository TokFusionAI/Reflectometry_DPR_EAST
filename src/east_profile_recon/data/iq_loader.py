import numpy as np


def read_QVWdata(time_point: float, filename_IQ: str):
    """Extract aligned Q, V, and W plasma IQ data for one time point."""
    T_period = 50e-6
    T_scan = 40e-6
    fs_IQ = 62.5e6
    IQ_sign = np.array([1, 1, 1])
    V_IQ_delay = 0.2e-6
    time_trigger = -10e-3
    period = 5
    ts_IQ = 1 / fs_IQ
    scan_length = int(np.ceil(T_scan * fs_IQ))
    period_length = int(np.ceil(T_period * fs_IQ))
    period1 = period * 2
    datalength_IQ = int(np.ceil(T_period * fs_IQ * period1 * 8))  # int16 values per time point
    bytes_per_timepoint = datalength_IQ * 2  # Each int16 value occupies two bytes
    L = datalength_IQ // 8
    time_diff = time_point - time_trigger
    offset_samples = int(np.ceil(fs_IQ * time_diff) * 2 * 8)
    with open(filename_IQ, 'rb') as fid:
        fid.seek(offset_samples)
        data = fid.read(bytes_per_timepoint)
    # 1. Convert the binary buffer to a NumPy array.
    data_cpu = np.frombuffer(data, dtype=np.int16)
    data_cpu = data_cpu.reshape(-1, 8).T
    # 2. Locate the trigger-channel maximum.
    L = data_cpu.shape[1]
    # search_periods = 2
    # search_start_idx = max(0, L - search_periods * period_length)
    # index = int(np.argmax(data_cpu[6, search_start_idx:])) + search_start_idx
    index = int(np.argmax(data_cpu[6, :]))
    # 3. Calculate extraction indices with exact MATLAB parity.
    l_batch = int(index // period_length)  # Sweep containing the peak, matching MATLAB
    ind_batch = index - l_batch * period_length  # Peak position within the sweep
    ind_batch = ind_batch + int(V_IQ_delay * fs_IQ)  # Apply the V-band IQ delay
    # Start at the aligned position, then offset each of the requested sweeps.
    base_start = int(ind_batch + (period_length - scan_length))
    # Extract the current sweep and the following period - 1 sweeps.
    i_offsets = (np.arange(period) * period_length)[:, None]  # Sweep offsets: 0, T, 2T, ...
    j_offsets = np.arange(scan_length)[None, :]  # Sample offsets within each sweep
    indices = base_start + i_offsets + j_offsets  # Final indices, matching MATLAB
    # Clip indices to the available buffer range.
    indices = np.clip(indices, 0, L - 1).astype(np.int64)
    indices_flat = indices.ravel()
    # 4. Construct complex Q, V, and W signals.
    Q_point_np = (data_cpu[0, indices_flat].astype(np.float32) + 1j * IQ_sign[0] * data_cpu[1, indices_flat].astype(np.float32)).reshape(period, scan_length)
    V_point_np = (data_cpu[2, indices_flat].astype(np.float32) + 1j * IQ_sign[1] * data_cpu[3, indices_flat].astype(np.float32)).reshape(period, scan_length)
    W_point_np = (data_cpu[4, indices_flat].astype(np.float32) + 1j * IQ_sign[2] * data_cpu[5, indices_flat].astype(np.float32)).reshape(period, scan_length)
    t_point_np = (time_point + indices * ts_IQ) * 1000
    return t_point_np, Q_point_np, V_point_np, W_point_np
