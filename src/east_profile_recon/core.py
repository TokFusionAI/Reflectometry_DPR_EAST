import argparse
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from .config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_FILTERS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_VACUUM_DATA_PATH,
    ENDPOINT_CONSECUTIVE_BELOW,
    ENDPOINT_CUSUM_H,
    ENDPOINT_CUSUM_K,
    ENDPOINT_SNR_CONTRAST_RATIO,
    FFTPOINT,
    FREQ_OFFSET,
    FS,
    Q_F_BASE,
    T_SCAN as DEFAULT_T_SCAN,
    TIME_DELAY,
    V_F_BASE,
    WINDOW_SIZE,
    W_F_BASE,
)
from .plotting import (
    plot_iq_data,
    plot_spectrogram,
    plot_strategy_comparison,
    plot_all_bands,
    plot_density_profile
)
from .data.h5_loader import get_IT, load_QVWdata, load_vacuum_data
from .data.iq_loader import read_QVWdata

def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct one EAST electron-density profile from raw or "
            "preprocessed Q/V/W reflectometry I/Q data."
        )
    )
    parser.add_argument("--shot", type=int, default=151858, help="EAST shot number")
    parser.add_argument(
        "--it",
        dest="IT",
        type=int,
        default=10735,
        help="Current-related inversion parameter",
    )
    parser.add_argument(
        "--time",
        dest="time_point",
        type=float,
        default=5.0,
        help="Reconstruction time in seconds",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(DEFAULT_DATA_ROOT),
        help="Root containing <shot>/Q_V_W.bin",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--iq-file",
        type=Path,
        default=None,
        help="Direct path to Q_V_W.bin; overrides --data-root",
    )
    input_group.add_argument(
        "--qvw-file",
        type=Path,
        default=None,
        help="Preprocessed HDF5 file containing t_point_np and Q/V/W arrays",
    )
    parser.add_argument(
        "--vacuum-file",
        type=Path,
        default=Path(DEFAULT_VACUUM_DATA_PATH),
        help="Vacuum-reference HDF5 file",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(DEFAULT_OUTPUT_ROOT),
        help="Root directory for reconstructed MAT files",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Computation device (default: auto)",
    )
    return parser


def _resolve_device(device):
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is false. "
            "Use --device cpu or install a CUDA-enabled PyTorch build."
        )
    return device


def _resolve_iq_file(shot, data_root, iq_file):
    if iq_file is None:
        path = Path(data_root).expanduser() / str(shot) / "Q_V_W.bin"
    else:
        path = Path(iq_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Raw Q/V/W I/Q file not found: {path}")
    return path


def main(argv=None):
    args = _build_parser().parse_args(argv)
    return run_single_timepoint(
        shot=args.shot,
        IT=args.IT,
        time_point=args.time_point,
        data_root=args.data_root,
        iq_file=args.iq_file,
        qvw_path=args.qvw_file,
        vacuum_path=args.vacuum_file,
        output_root=args.output_root,
        device=args.device,
    )


def run_single_timepoint(
    shot=151858,
    IT=10735,
    time_point=5.0,
    data_root=DEFAULT_DATA_ROOT,
    iq_file=None,
    qvw_path=None,
    vacuum_path=DEFAULT_VACUUM_DATA_PATH,
    output_root=DEFAULT_OUTPUT_ROOT,
    device=None,
):
    device = _resolve_device(device)
    shot_str = str(shot)
    start_time = time.perf_counter()
    shot_result = main_worker(
        shot,
        IT,
        time_point,
        data_root=data_root,
        iq_file=iq_file,
        qvw_path=qvw_path,
        vacuum_path=vacuum_path,
        device=device,
    )
    result = shot_result[shot_str]['result']
    elapsed = time.perf_counter() - start_time
    print(f"Total processing time: {elapsed:.6f}s")
    R = result['final']['R']
    Ne = result['final']['ne']
    mat_data = {
        'shot': np.array(shot, dtype=np.int64),
        'IT': np.array(IT, dtype=np.int64),
        'time': np.array([time_point], dtype=np.float64),
        'R': R,
        'Ne': Ne,
    }
    savepath = Path(output_root).expanduser() / shot_str
    savepath.mkdir(parents=True, exist_ok=True)
    mat_filename = savepath / f"{shot}_{time_point:.3f}s.mat"
    sio.savemat(mat_filename, mat_data)
    print(f"Saved shot {shot} at {time_point:.3f}s to {mat_filename}")
    return str(mat_filename)


def main_worker(
    shot,
    IT,
    time_point,
    data_root=DEFAULT_DATA_ROOT,
    iq_file=None,
    qvw_path=None,
    vacuum_path=DEFAULT_VACUUM_DATA_PATH,
    device=None,
):
    device = _resolve_device(device)
    shot_results = {}
    shot_str = str(shot)
    print(f"=====================Processing shot {shot_str} time {time_point}s=====================")
    if iq_file is not None and qvw_path is not None:
        raise ValueError("Pass either iq_file or qvw_path, not both")

    vac_data = load_vacuum_data(vacuum_path)
    if qvw_path is not None:
        t_point_np, Q_point_np, V_point_np, W_point_np = load_QVWdata(qvw_path)
    else:
        filename_IQ = _resolve_iq_file(shot, data_root, iq_file)
        time_delay = TIME_DELAY
        t_point_np, Q_point_np, V_point_np, W_point_np = read_QVWdata(
            time_point - time_delay,
            str(filename_IQ),
        )
    # plot_iq_data(t_point_np, Q_point_np, V_point_np, W_point_np)
    Q_point = torch.from_numpy(Q_point_np).to(device)
    V_point = torch.from_numpy(V_point_np).to(device)
    W_point = torch.from_numpy(W_point_np).to(device)
    T_scan = DEFAULT_T_SCAN
    Q_f = np.array(Q_F_BASE) + FREQ_OFFSET
    V_f = np.array(V_F_BASE) + FREQ_OFFSET
    W_f = np.array(W_F_BASE) + FREQ_OFFSET
    scan_len = Q_point.shape[1]
    fq_Q = torch.linspace(float(Q_f[0]), float(Q_f[1]), scan_len, dtype=torch.float64).to(device) * 4.0
    fq_V = torch.linspace(float(V_f[0]), float(V_f[1]), scan_len, dtype=torch.float64).to(device) * 4.0
    fq_W = torch.linspace(float(W_f[0]), float(W_f[1]), scan_len, dtype=torch.float64).to(device) * 6.0
    pla_results = process_three_bands(Q_point, V_point, W_point, fq_Q, fq_V, fq_W, device)
    timepoint_result = process_cpu_stage(pla_results, vac_data, IT, time_point, T_scan, Q_f, V_f, W_f, shot_str)
    result = {
        'shot': shot_str,
        'time_point': time_point,
        'result': timepoint_result,
        'data_pla': {'Q': Q_point_np, 'V': V_point_np, 'W': W_point_np}
    }
    shot_results[shot_str] = result
    return shot_results

def process_three_bands(Q_point, V_point, W_point, fq_Q, fq_V, fq_W, device):
    """Process the Q, V, and W bands as one batch."""
    data_batch = torch.stack([Q_point, V_point, W_point], dim=0)
    fq_batch = torch.stack([fq_Q, fq_V, fq_W], dim=0)
    filters = torch.tensor(
        DEFAULT_FILTERS,
        dtype=torch.float64,
        device=device,
    )
    batch_results = refl_fb_analysis_batch(
        data_batch,
        WINDOW_SIZE,
        FFTPOINT,
        FS,
        filters,
        fq_batch,
    )
    return batch_results

def refl_fb_analysis_batch(data, window_size, fftpoint, fs, filters, f_band):
    """Analyze beat frequencies for a three-band batch.

    ``data`` has shape ``[3, period, scan_length]``.
    """
    step_size = window_size // 2
    freq_m, f, P, n_segments = sliding_window_batch(
        data, window_size, step_size, fftpoint, fs, f_band
    )
    fb, fb_err, P_max = find_fb_batch(f, P, filters)
    result = {
        'freq_m': freq_m,          # [3, n_segments]
        'fb': fb,                  # [3, n_segments]
        'fb_err': fb_err,          # [3, n_segments]
        'P': P,                    # [3, n_segments, fftpoint]
        'P_max': P_max,            # [3, n_segments]
        'f': f,                    # [fftpoint]
        'filters': filters,        # [3, 2]
        'f_band': f_band,          # [3, scan_length]
        'data': data               # [3, period, scan_length] 
    }
    return result

def sliding_window_batch(data, window_size, step_size, fftpoint, fs, f_band):
    """Apply batched sliding-window spectral processing.

    ``data`` has shape ``[3, period, scan_length]`` and ``f_band`` has
    shape ``[3, scan_length]``.
    """
    batch_size, period, scan_length = data.shape       
    n_segments = int((scan_length - window_size) / step_size) + 1
    f_band_windows = f_band.unfold(dimension=1, size=window_size, step=step_size)  # (batch, n_segments, window_size)
    all_freq_m = f_band_windows.mean(dim=2).to(torch.float64)  # (batch, n_segments)
    windows = data.unfold(dimension=2, size=window_size, step=step_size)  # (batch, period, n_segments, window_size)
    windows = windows.permute(0, 2, 1, 3)  # (batch, n_segments, period, window_size)
    flattened_len = period * window_size
    all_windows_flat = windows.reshape(batch_size * n_segments, flattened_len)
    window_means = all_windows_flat.mean(dim=1, keepdim=True)
    all_windows_centered = (all_windows_flat - window_means).to(torch.complex128)
    # Welch PSD implementation using framing + FFT
    f, P =torch_welch(all_windows_centered, fs=fs, nperseg=fftpoint)
    f_final = f / 1e6
    P_shifted = torch.fft.fftshift(P, dim=1)
    P_reshaped = P_shifted.reshape(batch_size, n_segments, fftpoint)
    return all_freq_m, f_final, P_reshaped, n_segments

def find_fb_batch(f, P, filters):
    """Estimate beat frequency and uncertainty for all bands.

    ``f`` has shape ``[fftpoint]``, ``P`` has shape
    ``[3, n_segments, fftpoint]``, and ``filters`` has shape ``[3, 2]``.
    """
    lower = filters[:, 0].unsqueeze(1)  # (batch,1)
    upper = filters[:, 1].unsqueeze(1)  # (batch,1)
    f_g = f.unsqueeze(0)  # (1, fftpoint)
    mask_freq = (f_g > lower) & (f_g < upper)  # (batch, fftpoint)
    mask_freq = mask_freq.unsqueeze(1)  # (batch,1,fftpoint)
    P_band = P * mask_freq
    P1_max = P_band.max(dim=2).values
    threshold = P1_max * torch.exp(torch.tensor(-1.0, dtype=P1_max.dtype))
    mask_signal = P_band >= threshold.unsqueeze(2)
    P_final = torch.where(mask_signal, P_band, torch.zeros_like(P_band))
    sum_P = P_final.sum(dim=2)
    sum_P_safe = sum_P + 1e-12
    fb = (f.unsqueeze(0).unsqueeze(0) * P_final).sum(dim=2) / sum_P_safe
    diff = f.unsqueeze(0).unsqueeze(0) - fb.unsqueeze(2)
    var_num = (diff ** 2 * P_final).sum(dim=2)
    fb_err = torch.sqrt(2 * var_num / sum_P_safe)  
    return fb, fb_err, P1_max

def torch_welch(signals, fs, nperseg, noverlap=None):
    """Estimate Welch power spectral density with PyTorch."""
    if noverlap is None:
        noverlap = nperseg // 2
    hop = nperseg - noverlap
    N, L = signals.shape
    is_complex = torch.is_complex(signals)
    if L < nperseg:
        pad = nperseg - L
        signals = torch.nn.functional.pad(signals, (0, pad))
    frames = signals.unfold(dimension=1, size=nperseg, step=hop) 
    frames = frames - frames.mean(dim=-1, keepdim=True)
    window = torch.hamming_window(nperseg, periodic=True,
                                device=signals.device, dtype=signals.real.dtype)
    frames_win = frames * window
    if is_complex:
        S = torch.fft.fft(frames_win, dim=-1)
        df = fs / nperseg
        freqs = torch.linspace(-(nperseg//2 - 1) * df, (nperseg//2) * df, nperseg,
                             device=signals.device, dtype=torch.float64)
    else:
        S = torch.fft.rfft(frames_win, dim=-1)                    # One-sided spectrum
        freqs = torch.fft.rfftfreq(nperseg, d=1.0/fs, device=signals.device, dtype=torch.float64)
    psd = (S.abs() ** 2)
    U = (window ** 2).sum()
    psd = psd / (U * fs)          
    P_avg = psd.mean(dim=1)      
    if not is_complex:
        if nperseg % 2 == 0:     
            P_avg[..., 1:-1] *= 2.0
        else:                   
            P_avg[..., 1:] *= 2.0
    return freqs, P_avg

def process_cpu_stage(pla_results, vacuum_data, IT, time_point, T_scan, Q_f, V_f, W_f, shot_str):
    """Run the CPU-side inversion and profile post-processing stage."""
    rand_number = 20
    rand_number_check = 0
    R_zerone = 2.37
    e = 1.602e-19
    me = 9.1096e-31
    freq_cut = 105
    Bt_n0 = (IT * 1.7) / (4086 * R_zerone)
    freq_zero = e * Bt_n0 / (2 * np.pi * me) / 1e9
    dfdt_Q = (Q_f[1] - Q_f[0]) * 4 / T_scan
    dfdt_V = (V_f[1] - V_f[0]) * 4 / T_scan
    dfdt_W = (W_f[1] - W_f[0]) * 6 / T_scan        
    (
        fb_vac_Q,
        fb_err_vac_Q,
        fb_vac_V,
        fb_err_vac_V,
        fb_vac_W,
        fb_err_vac_W,
    ) = vacuum_data
    fb = pla_results['fb'].cpu().numpy()
    fb_err = pla_results['fb_err'].cpu().numpy()
    freq_m = pla_results['freq_m'].cpu().numpy()
    P_max = pla_results['P_max'].cpu().numpy()
    P = pla_results['P'].cpu().numpy()
    f = pla_results['f'].cpu().numpy()
    spectral_filters = pla_results['filters'].cpu().numpy()
    fb_pla_Q, fb_pla_V, fb_pla_W = fb
    fb_err_Q, fb_err_V, fb_err_W = fb_err
    fq_Q, fq_V, fq_W = freq_m
    P_max_Q, P_max_V, P_max_W = P_max
    P_Q, P_V, P_W = P
    # plot_spectrogram(f, P, savepath=None, filename=None)
    tau_Q = ((fb_pla_Q - fb_vac_Q) * 1e6) / dfdt_Q
    tau_err_Q = np.sqrt(fb_err_Q**2 + fb_err_vac_Q**2) * 1e6 / dfdt_Q
    tau_V = ((fb_pla_V - fb_vac_V) * 1e6) / dfdt_V
    tau_err_V = np.sqrt(fb_err_V**2 + fb_err_vac_V**2) * 1e6 / dfdt_V
    tau_W = ((fb_pla_W - fb_vac_W) * 1e6) / dfdt_W
    tau_err_W = np.sqrt(fb_err_W**2 + fb_err_vac_W**2) * 1e6 / dfdt_W
    n_windows = int(fb_pla_Q.shape[0])
    mask_V_part = np.arange(1, n_windows-1)
    f1 = float(fq_V[0])
    f2 = float(fq_V[-1])
    mask_Q_part = fq_Q < f1
    mask_W_part = fq_W > f2
    mask_W_part[:4] = False
    mask_V_bool = np.zeros(fq_V.size, dtype=bool)
    mask_V_bool[mask_V_part] = True
    freq_all = np.concatenate([fq_Q[mask_Q_part], fq_V[mask_V_part], fq_W[mask_W_part]]).astype(float)
    P_max_all = np.concatenate([P_max_Q[mask_Q_part], P_max_V[mask_V_part], P_max_W[mask_W_part]]).astype(float)
    fb_all = np.concatenate([fb_pla_Q[mask_Q_part], fb_pla_V[mask_V_part], fb_pla_W[mask_W_part]]).astype(float)
    fb_err_all = np.concatenate([fb_err_Q[mask_Q_part], fb_err_V[mask_V_part], fb_err_W[mask_W_part]]).astype(float)
    tau_all = np.concatenate([tau_Q[mask_Q_part], tau_V[mask_V_part], tau_W[mask_W_part]]).astype(float)
    tau_err_all = np.concatenate([tau_err_Q[mask_Q_part], tau_err_V[mask_V_part], tau_err_W[mask_W_part]]).astype(float)
    P_all = np.concatenate([P_Q[mask_Q_part,:], P_V[mask_V_part,:], P_W[mask_W_part,:]], axis=0).astype(float)
    mask_middle = freq_all > freq_zero
    freq_middle = freq_all[mask_middle]
    P_max_middle = P_max_all[mask_middle]
    fb_middle = fb_all[mask_middle]
    fb_err_middle = fb_err_all[mask_middle]
    tau_middle = tau_all[mask_middle]
    tau_err_middle = tau_err_all[mask_middle]
    P_middle = P_all[mask_middle,:]
    orig_mid_idx = np.nonzero(mask_middle)[0]
    all_to_mid = np.full(freq_all.size, -1, dtype=np.int64)
    all_to_mid[orig_mid_idx] = np.arange(orig_mid_idx.size, dtype=np.int64)
    q_count = np.count_nonzero(mask_Q_part)
    v_count = np.count_nonzero(mask_V_part)
    q_mid_pos = np.nonzero(orig_mid_idx < q_count)[0]
    v_mid_pos = np.nonzero((orig_mid_idx >= q_count) & (orig_mid_idx < q_count + v_count))[0]
    indQ = np.array([int(q_mid_pos[-1])], dtype=np.int64) if q_mid_pos.size else np.array([], dtype=np.int64)
    indV = np.array([int(v_mid_pos[-1])], dtype=np.int64) if v_mid_pos.size else np.array([], dtype=np.int64)
    ind1 = auto_select1(fb_err_middle)
    q_end = int(indQ[0]) if (hasattr(indQ, '__len__') and len(indQ) > 0) else None
    v_end = int(indV[0]) if (hasattr(indV, '__len__') and len(indV) > 0) else None
    ind2 = ind1
    spectral_masks = (
        (f[None, :] > spectral_filters[:, 0, None])
        & (f[None, :] < spectral_filters[:, 1, None])
    )
    plasma_global_noise = calculate_global_noise(
        (
            (P_Q, P_max_Q, fq_Q, mask_Q_part, spectral_masks[0]),
            (P_V, P_max_V, fq_V, mask_V_bool, spectral_masks[1]),
            (P_W, P_max_W, fq_W, mask_W_part, spectral_masks[2]),
        ),
        freq_zero,
    )
    if (q_end is not None) and (ind1 <= q_end):
        ind2_middle = auto_select2(
            P_Q[mask_Q_part, :],
            spectrum_mask=spectral_masks[0],
            noise_scalar=plasma_global_noise
        )
        if 0 <= ind2_middle < q_count:
            ind2 = int(all_to_mid[int(ind2_middle)])
    elif (v_end is not None) and (ind1 <= v_end) and (q_end is None or ind1 > q_end):
        ind2_middle = auto_select2(
            P_V[mask_V_part, :],
            spectrum_mask=spectral_masks[1],
            noise_scalar=plasma_global_noise
        )
        if 0 <= ind2_middle < v_count:
            ind2 = int(all_to_mid[q_count + int(ind2_middle)])
    elif (v_end is not None) and (ind1 > v_end):
        ind2_middle = auto_select2(
            P_W[mask_W_part, :],
            spectrum_mask=spectral_masks[2],
            noise_scalar=plasma_global_noise
        )
        w_count = np.count_nonzero(mask_W_part)
        if 0 <= ind2_middle < w_count:
            ind2 = int(all_to_mid[q_count + v_count + int(ind2_middle)])
    ind1 = int(ind1)
    ind2 = ind1 if int(ind2) < 0 else int(ind2)
    final_ind = min(ind1, ind2)
    tau_valid = tau_middle[:final_ind + 1]
    tau_err_valid = tau_err_middle[:final_ind + 1]
    freq_valid = freq_middle[:final_ind + 1]
    # inds1 = np.arange(0, ind1 + 1)
    # inds2 = np.arange(0, ind2 + 1)
    # final_inds = np.arange(0, final_ind + 1)
    # raw_ind1 = find_w_indices(inds1, mask_middle, mask_Q_part, mask_V_part, mask_W_part)
    # raw_ind2 = find_w_indices(inds2, mask_middle, mask_Q_part, mask_V_part, mask_W_part)
    # raw_inds = find_w_indices(final_inds, mask_middle, mask_Q_part, mask_V_part, mask_W_part)
    # plot_strategy_comparison(
    # f, raw_ind1, raw_ind2, raw_inds, 
    # fq_Q, P_Q.T, fb_pla_Q, fb_err_Q,
    # fq_V, P_V.T, fb_pla_V, fb_err_V,
    # fq_W, P_W.T, fb_pla_W, fb_err_W
    # )
    # Interpolate using the screened arrays; freq_valid corresponds to tau_valid.
    tau_zero = pchip_interpolate(freq_valid, tau_valid, freq_zero)
    tau_final = np.concatenate([np.array([tau_zero]), tau_valid])
    tau_err_final = np.concatenate([np.array([0]), tau_err_valid])
    freq_final = np.concatenate([np.array([freq_zero]), freq_valid])
    R1, N1 = refl_density(
        IT=IT,
        rand_number=rand_number,
        fq=freq_final,
        tau=tau_final,
        tau_err=tau_err_final,
        rand_number_check=rand_number_check,
        R_zero=R_zerone
    )
    final_index = find_original_index(freq_middle, freq_final, remove_missing=True)
    ind_Q = find_original_index(fq_Q, freq_final, remove_missing=True)
    ind_V = find_original_index(fq_V, freq_final, remove_missing=True)
    ind_W = find_original_index(fq_W, freq_final, remove_missing=True)
    R_final, N_final, indx, R_108, N_108 = process_data(
        R1=R1,
        N1=N1,
        final_index=final_index,
        ind_Q=ind_Q,
        ind_V=ind_V,
        ind_W=ind_W
    )
    # plot_all_bands(
    #     f,
    #     ind_Q, ind_V, ind_W,
    #     fq_Q, P_Q.T, fb_pla_Q, fb_err_Q,
    #     fq_V, P_V.T, fb_pla_V, fb_err_V, 
    #     fq_W, P_W.T, fb_pla_W, fb_err_W,
    # )
    # plot_density_profile(R_final, N_final, shot_str, time_point)
    res_dict = {
        'final': {
            'R': R_final,
            'ne': N_final,
            'tau': tau_final,
            'freq': freq_final,
            'IT': IT
        },
        'processed_data': {
            'R': R_108,
            'ne': N_108,
            'indx': indx
        },
        'time_point': time_point
    }
    return res_dict

def auto_select1(
    fb_err,
    cusum_k=ENDPOINT_CUSUM_K, cusum_h=ENDPOINT_CUSUM_H,
    global_anchor_ratio=0.15, anchor_min_window=5,
):
    floor = 0.1
    W_total = fb_err.size
    n_anchor = max(int(global_anchor_ratio * W_total), anchor_min_window)
    n_anchor = min(n_anchor, W_total)
    windows = np.lib.stride_tricks.sliding_window_view(fb_err, n_anchor)
    centers = np.median(windows, axis=1)
    mads = np.median(np.abs(windows - centers[:, None]), axis=1)
    sigmas = np.maximum(1.4826 * mads, floor)
    score_trace = centers + sigmas
    best_start = int(np.nanargmin(score_trace))
    best_center = centers[best_start]
    best_sigma_rob = sigmas[best_start]
    z = (fb_err - best_center) / best_sigma_rob
    scan_start = best_start
    z_scan = z[scan_start:]
    increments = np.minimum(z_scan - cusum_k, 3.0)
    cumulative = np.concatenate([[0.0], np.cumsum(increments)])
    S = cumulative - np.minimum.accumulate(cumulative)
    alarm_pos = np.nonzero(S[1:] >= cusum_h)[0]
    alarm_local = int(alarm_pos[0]) if alarm_pos.size else None
    if alarm_local is None:
        end = W_total - 1
    else:
        normal_pos = np.nonzero(z_scan[:alarm_local + 1] <= cusum_k)[0]
        end_local = int(normal_pos[-1]) if normal_pos.size else int(alarm_local)
        end = scan_start + end_local
    return np.int64(end)

def auto_select2(
    P,
    spectrum_mask,
    noise_scalar,
    snr_contrast_ratio=ENDPOINT_SNR_CONTRAST_RATIO,
    consecutive_below=ENDPOINT_CONSECUTIVE_BELOW,
    local_anchor_ratio=0.1,
    anchor_min_window=5,
):
    W = P.shape[0]
    masked_spectra = P[:, spectrum_mask]
    rows = np.arange(W)
    local_peak_idx = np.argmax(masked_spectra, axis=1)
    peak_max = masked_spectra[rows, local_peak_idx]
    neighbor_idx = local_peak_idx[:, None] + np.array([-1, 0, 1])
    valid_neighbor = (
        (neighbor_idx >= 0) & (neighbor_idx < masked_spectra.shape[1])
    )
    neighbor_idx = np.clip(neighbor_idx, 0, masked_spectra.shape[1] - 1)
    neighbor_power = masked_spectra[rows[:, None], neighbor_idx]
    peak = np.sum(
        np.where(valid_neighbor, neighbor_power, 0.0),
        axis=1,
    ) / np.count_nonzero(valid_neighbor, axis=1)
    snr = np.divide(
        peak,
        noise_scalar,
        out=np.zeros(W, dtype=float),
        where=peak_max > 0,
    )
    if W <= 1:
        return np.int64(max(W - 1, 0))
    win = min(max(int(W * local_anchor_ratio), anchor_min_window), W)
    windows = np.lib.stride_tricks.sliding_window_view(snr, win)
    centers = np.median(windows, axis=1)
    mads = 1.4826 * np.median(np.abs(windows - centers[:, None]), axis=1)
    scores = centers + mads
    min_score = np.nanmin(scores)
    best_candidates = np.nonzero(np.isclose(scores, min_score))[0]
    best_start = int(best_candidates[best_candidates.size // 2])
    best_end = min(best_start + win - 1, W - 1)
    background = snr[best_start:best_end + 1]
    bg_center = float(np.median(background))
    signal_center = float(np.percentile(snr[:best_start], 75)) if best_start > 0 else bg_center
    threshold = bg_center + snr_contrast_ratio * (signal_center - bg_center)
    below = snr < threshold
    crossing_start = None
    for start in range(1, best_start + 1):
        stop = start + consecutive_below
        if stop <= W and np.all(below[start:stop]):
            crossing_start = start
            break
    if crossing_start is not None:
        upper_idx = crossing_start - 1
    else:
        above = np.nonzero(snr[:best_start + 1] >= threshold)[0]
        upper_idx = int(above[-1]) if above.size else max(best_start - 1, 0)
    return np.int64(upper_idx)

def calculate_global_noise(band_data, freq_zero):
    window_noise = []
    for power, peak_max, frequency, window_mask, spectrum_mask in band_data:
        retained_mask = window_mask & (frequency > freq_zero)
        masked_power = power[retained_mask][:, spectrum_mask]
        background_mask = masked_power < peak_max[retained_mask, None] / np.e
        background_count = np.count_nonzero(background_mask, axis=1)
        background_sum = np.where(background_mask, masked_power, 0.0).sum(axis=1)
        window_noise.append(
            np.divide(
                background_sum,
                background_count,
                out=np.full(background_sum.shape, np.nan),
                where=background_count > 0,
            )
        )
    sorted_noise = np.sort(np.concatenate(window_noise))
    trim_each_tail = int(np.floor(sorted_noise.size * 0.10 + 0.5))
    retained_noise = sorted_noise[
        trim_each_tail:sorted_noise.size - trim_each_tail
        if trim_each_tail
        else sorted_noise.size
    ]
    return np.mean(retained_noise)

def pchip_interpolate(x_original, y_original, x_interp):
    """
    Optimized PCHIP interpolation for a single point (Pure CPU/NumPy version).
    """
    xi = float(x_interp)
    N = len(x_original)
    if N < 2:
        if N == 1:
            return float(y_original[0]) 
        else:
            return np.nan 
    idx = np.searchsorted(x_original, xi, side='right') - 1
    # Handle strictly out of bounds or edges
    idx = max(0, min(idx, N - 2))
    # Define window to fetch: [start, end]
    start = max(0, idx - 1)
    end = min(N, idx + 3)
    # Slice on CPU (no H2D transfer needed)
    x_sub = x_original[start:end]
    y_sub = y_original[start:end]
    # Re-calculate local index relative to slice
    i = idx - start
    # Handle small array case (N=2)
    if len(x_sub) < 3:
        # Linear interpolation
        return y_sub[0] + (y_sub[1]-y_sub[0])*(xi-x_sub[0])/(x_sub[1]-x_sub[0])
    # Compute h and delta
    h = x_sub[1:] - x_sub[:-1]
    delta = (y_sub[1:] - y_sub[:-1]) / h
    # Helper to compute derivative at local node k
    def get_deriv(k):
        # Internal point formula
        if 0 < k < len(delta):
            d1, d2 = delta[k-1], delta[k]
            if d1 * d2 <= 0:
                return 0.0
            w1 = 2*h[k] + h[k-1]
            w2 = h[k] + 2*h[k-1]
            return (w1 + w2) / (w1/d1 + w2/d2)
        # Left boundary of slice (corresponds to actual left boundary)
        elif k == 0:
            d = ((2*h[0] + h[1])*delta[0] - h[0]*delta[1]) / (h[0] + h[1])
            if d * delta[0] < 0: return 0.0
            return d
        # Right boundary of slice
        else:
            n = len(delta) - 1
            d = ((2*h[n] + h[n-1])*delta[n] - h[n]*delta[n-1]) / (h[n] + h[n-1])
            if d * delta[n] < 0: return 0.0
            return d
    # Calculate derivatives at x[i] and x[i+1]
    d0 = get_deriv(i)
    d1 = get_deriv(i+1)
    # Hermite interpolation
    x0, x1 = x_sub[i], x_sub[i+1]
    y0, y1 = y_sub[i], y_sub[i+1]
    dx = x1 - x0
    if dx == 0: return y0
    t = (xi - x0) / dx
    t2 = t*t
    t3 = t2*t
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2
    return float(h00*y0 + h10*dx*d0 + h01*y1 + h11*dx*d1)

def refl_density(IT, rand_number, fq, tau, tau_err, rand_number_check, R_zero):
    """Reconstruct one electron-density profile on the CPU."""
    R_inwall = 1.354
    c = 3.0e8
    e = 1.602e-19
    me = 9.1096e-31
    Bt_n0 = (IT * 1.7) / (4086 * R_zero)
    fq_zero = e * Bt_n0 / (2 * np.pi * me) / 1e9  # GHz
    fq_final = fq
    tau_final = tau
    tau_err_final = tau_err
    tau_vac = 2 * (R_zero - R_inwall) / c
    tau1 = tau_vac * 1e9 + tau_final
    tau1_err = tau_err_final
    fq0 = fq_zero 
    R_n0_val = R_zero
    if rand_number_check == 0:
        phase = cumulative_trapezoid_complete(
            y=tau1 / 1e9,
            x=2 * np.pi * fq_final * 1e9,
            initial=0
        )
        R_out, ne_out = refl_inversion_2(
            f0=fq0 * 1e9,
            x0=R_n0_val,
            n0=0,
            freq=fq_final[1:] * 1e9,
            phase=phase[1:],
            IT=IT
        )
        R1 = R_out
        N1 = ne_out / 1e19
    elif rand_number_check == 1:
        fq_input = fq_final[1:] * 1e9
        err_coef = np.random.rand(tau1_err.shape[0], rand_number) * 2 - 1
        tau_temp = tau1[:, None] + 1 * err_coef * tau1_err[:, None]
        x_ph = 2 * np.pi * fq_final * 1e9 # (N,)
        y_ph = tau_temp / 1e9             # (N, M)        
        dx = x_ph[1:] - x_ph[:-1]         # (N-1,)
        sy = y_ph[1:, :] + y_ph[:-1, :]   # (N-1, M)
        integrals = (dx[:, None] * sy) / 2 # Broadcasting 
        phase = np.empty_like(y_ph)
        phase[0, :] = 0
        phase[1:, :] = np.cumsum(integrals, axis=0)
        R1_temp, N1_temp = refl_inversion_batch(
            f0=fq0 * 1e9,
            x0=R_n0_val,
            n0=0,
            freq=fq_input,
            phase=phase[1:, :], 
            IT=IT
        )
        N1_temp = N1_temp / 1e19
        R1 = np.mean(R1_temp, axis=1)
        N1 = np.mean(N1_temp, axis=1)
    else:
        R1 = np.array([])
        N1 = np.array([])
    return R1, N1

def cumulative_trapezoid_complete(y, x, initial=0):
    """NumPy version"""
    N = y.shape[0]
    integrals = (x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2
    result = np.empty(N, dtype=y.dtype)
    result[0] = initial
    result[1:] = np.cumsum(integrals) + initial
    return result

def refl_inversion_2(f0, x0, n0, freq, phase, IT):
    """Run single-profile refractive-index inversion on the CPU."""
    K = freq.shape[0]
    c = 3.0e8
    x_hist = np.empty(K + 1, dtype=np.float64)
    Bt_hist = np.empty(K + 1, dtype=np.float64)
    ne_hist = np.empty(K + 1, dtype=np.float64)
    x_hist[0] = x0
    Bt_hist[0] = (IT * 1.7) / (4086 * x0)
    ne_hist[0] = find_ne(f0, Bt_hist[0])
    for k in range(K):
        f_k = freq[k] 
        x_view = x_hist[:k+1]
        Bt_view = Bt_hist[:k+1]
        ne_view = ne_hist[:k+1]        
        N_temp = X_refraction_index(f_k, Bt_view, ne_view)  # (k+1,)        
        phi_temp = 0.0
        if k > 0:
            integrand = N_temp * (4 * np.pi * f_k / c)
            phi_temp = -np.trapz(integrand, x=x_view)
        dphi = phase[k] - phi_temp
        N_last = N_temp[-1]
        if N_last == 0:
            N_last = 1e-12
        term = (2 * dphi * c / (4 * np.pi * f_k)) / N_last
        x_next = x_view[-1] - term
        x_hist[k+1] = x_next      
        Bt_next = (IT * 1.7) / (4086 * x_next)
        Bt_hist[k+1] = Bt_next
        ne_next = find_ne(f_k, Bt_next)
        ne_hist[k+1] = ne_next
    x3 = np.concatenate([np.array([x0]), (x_hist[:-1] + x_hist[1:]) / 2])
    n3 = np.abs(np.concatenate([np.array([n0]), (ne_hist[:-1] + ne_hist[1:]) / 2]))
    return x3, n3

def refl_inversion_batch(f0, x0, n0, freq, phase, IT):
    """Run a batched, parallel version of ``refl_inversion_2``.

    ``phase`` has shape ``(K, M)``.
    """
    K, M = phase.shape
    c = 3.0e8
    x_hist = np.empty((K + 1, M), dtype=np.float64)
    Bt_hist = np.empty((K + 1, M), dtype=np.float64)
    ne_hist = np.empty((K + 1, M), dtype=np.float64)
    x_hist[0, :] = x0
    Bt_hist[0, :] = (IT * 1.7) / (4086 * x0)
    ne_hist[0, :] = find_ne(f0, Bt_hist[0, :])
    for k in range(K):
        f_k = freq[k] 
        x_view = x_hist[:k+1, :]
        Bt_view = Bt_hist[:k+1, :]
        ne_view = ne_hist[:k+1, :]        
        N_temp = X_refraction_index(f_k, Bt_view, ne_view)  # (k+1, M)   
        phi_temp = np.zeros(M, dtype=np.float64)
        if k > 0:
            integrand = N_temp * (4 * np.pi * f_k / c)
            try:
                phi_temp = -np.trapezoid(integrand, x=x_view, axis=0)
            except AttributeError:
                phi_temp = -np.trapz(integrand, x=x_view, axis=0)
        dphi = phase[k, :] - phi_temp # (M,)
        N_last = N_temp[-1, :]
        N_last = np.where(N_last == 0, 1e-12, N_last)
        term = (2 * dphi * c / (4 * np.pi * f_k)) / N_last
        x_next = x_view[-1, :] - term
        x_hist[k+1, :] = x_next      
        Bt_next = (IT * 1.7) / (4086 * x_next)
        Bt_hist[k+1, :] = Bt_next
        ne_next = find_ne(f_k, Bt_next)
        ne_hist[k+1, :] = ne_next
    x3 = np.concatenate([x_hist[0:1, :], (x_hist[:-1, :] + x_hist[1:, :]) / 2], axis=0)
    n3 = np.abs(np.concatenate([ne_hist[0:1, :], (ne_hist[:-1, :] + ne_hist[1:, :]) / 2], axis=0))
    return x3, n3

def find_ne(f, Bt):
    """Calculate electron density from frequency and magnetic field."""
    e = 1.602e-19
    epsilon0 = 8.854e-12
    me = 9.1096e-31
    # c = 3.0e8
    Wce = e * Bt / me 
    WR = 2 * np.pi * f 
    X = WR * (WR - Wce)
    ne = epsilon0 * me * X / e ** 2 
    return ne

def X_refraction_index(f, Bt, ne):
    """Calculate the X-mode refractive index."""
    e = 1.602e-19
    epsilon0 = 8.854e-12
    me = 9.1096e-31
    # c = 3.0e8
    ne = np.asarray(ne, dtype=complex) 
    Wpe = np.sqrt(e ** 2 * ne / (epsilon0 * me)) 
    Wce = e * Bt / me 
    X = (np.real(Wpe) ** 2 - np.imag(Wpe) ** 2) / ((2 * np.pi * f) ** 2) 
    Y = Wce / (2 * np.pi * f) 
    denominator = 1 - X - Y ** 2
    N1 = 1 - (X * (1 - X)) / denominator 
    mask = np.real(N1) < 0
    N1[mask] = 0
    N = np.sqrt(N1) 
    return N

def find_original_index(freq, freq_final, remove_missing: bool = False):
    """Map final frequencies back to their original NumPy indices."""
    ff = freq_final[1:]
    idx_min = np.full(ff.shape, -1, dtype=np.int64)
    if ff.size == 0 or freq.size == 0:
        return idx_min
    mask_final = ~np.isnan(ff)
    mask_freq = ~np.isnan(freq)
    if not np.any(mask_final) or not np.any(mask_freq):
        return idx_min
    freq_keep = freq[mask_freq]
    diff = np.abs(freq_keep[:, None] - ff[None, :])
    closest_in_keep = np.argmin(diff, axis=0)  # index in freq_keep
    min_dists = diff[closest_in_keep, np.arange(diff.shape[1])]
    rel_tol = 1e-6
    abs_tol = 1e-8
    tol = np.maximum(np.abs(ff) * rel_tol, abs_tol)
    valid = (min_dists <= tol) & mask_final
    if np.any(valid):
        orig_indices = np.nonzero(mask_freq)[0]
        idx_min[valid] = orig_indices[closest_in_keep[valid]]
    if remove_missing:
        if np.any(idx_min >= 0):
            return idx_min[idx_min >= 0]
        else:
            return np.array([], dtype=np.int64)
    return idx_min

def process_data(R1, N1, final_index, ind_Q, ind_V, ind_W):
    """Map the reconstructed profile to the fixed 108-position grid."""
    Q_LEN = 32 
    V_LEN = 38 
    W_LEN = 38 
    TOTAL_LEN = Q_LEN + V_LEN + W_LEN
    QV_LEN = Q_LEN + V_LEN 
    final_index = final_index[final_index >= 0]
    ind_Q_len = ind_Q.size
    ind_QV_len = ind_Q_len + ind_V.size
    # Calculate masks (NumPy)
    mask_a = final_index < ind_Q_len
    mask_b = (final_index >= ind_Q_len) & (final_index < ind_QV_len)
    mask_c = final_index >= ind_QV_len
    # Extract indices
    a_inds = final_index[mask_a]
    b_inds = final_index[mask_b]
    c_inds = final_index[mask_c]
    a_points = np.array([], dtype=np.int64)
    b_points = np.array([], dtype=np.int64)
    c_points = np.array([], dtype=np.int64)
    if a_inds.size > 0:
        a_points = ind_Q[a_inds]  # Q: 0~31
    if b_inds.size > 0:
        b_points = ind_V[b_inds - ind_Q_len] + Q_LEN  # V: 32~69
    if c_inds.size > 0:
        c_points = ind_W[c_inds - ind_QV_len] + QV_LEN  # W: 70~107
    valid_a = (a_points >= 0) & (a_points < TOTAL_LEN)
    a_points = a_points[valid_a]
    a_len = a_points.size
    valid_b = (b_points >= 0) & (b_points < TOTAL_LEN)
    b_points = b_points[valid_b]
    b_len = b_points.size
    valid_c = (c_points >= 0) & (c_points < TOTAL_LEN)
    c_points = c_points[valid_c]
    c_len = c_points.size
    # Initialize 108-point arrays (NumPy)
    R_108 = np.zeros(TOTAL_LEN, dtype=np.float64)
    N_108 = np.zeros(TOTAL_LEN, dtype=np.float64)
    point = 1  # Start from index 1 of R1/N1
    if a_len > 0:
        R_108[a_points] = R1[point: point + a_len]
        N_108[a_points] = N1[point: point + a_len]
        point += a_len
    if b_len > 0:
        R_108[b_points] = R1[point: point + b_len]
        N_108[b_points] = N1[point: point + b_len]
        point += b_len
    if c_len > 0:
        R_108[c_points] = R1[point: point + c_len]
        N_108[c_points] = N1[point: point + c_len]
        point += c_len
    idx0 = -1
    if a_len > 0:
        idx0 = a_points[0]
    elif b_len > 0:
        idx0 = b_points[0]
    elif c_len > 0:
        idx0 = c_points[0]
    if idx0 > 0:
        R_108[idx0 - 1] = 2.3700
        N_108[idx0 - 1] = 0.0
    mask_small = R_108 < 1.95
    R_108[mask_small] = 0.0
    N_108[mask_small] = 0.0
    mask_nonzero = R_108 > 0
    R_final = R_108[mask_nonzero]
    N_final = N_108[mask_nonzero]
    bounds = np.zeros(6, dtype=np.int64)
    if a_len > 0:
        bounds[0] = a_points[0]
        bounds[1] = a_points[-1]
    if b_len > 0:
        bounds[2] = b_points[0]
        bounds[3] = b_points[-1]
    if c_len > 0:
        bounds[4] = c_points[0]
        bounds[5] = c_points[-1]
    return R_final, N_final, bounds, R_108, N_108

def find_w_indices(ind, mask_middle, mask_Q_part, mask_V_part, mask_W_part):
    """Map indices from ``freq_middle`` back to the original W-band indices."""
    orig_mid_idx = np.nonzero(mask_middle)[0]
    ind_arr = np.asarray(ind)
    ind_arr = ind_arr[(ind_arr >= 0) & (ind_arr < orig_mid_idx.size)]
    pos_in_all = orig_mid_idx[ind_arr]
    q_len = np.count_nonzero(mask_Q_part)
    v_len = np.count_nonzero(mask_V_part)
    w_start = q_len + v_len
    w_orig_idx = np.nonzero(mask_W_part)[0]
    w_mask = pos_in_all >= w_start
    w_pos = pos_in_all[w_mask]
    rel = (w_pos - w_start).astype(int)
    rel = rel[(rel >= 0) & (rel < w_orig_idx.size)]
    inds_w = w_orig_idx[rel]
    return np.asarray(inds_w, dtype=np.int64)

def _asnumpy(value):
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "get"):
        return value.get()
    return np.asarray(value)
