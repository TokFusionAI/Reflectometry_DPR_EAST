import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def _asnumpy(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "get"):
        return value.get()
    return np.asarray(value)

def plot_iq_data(time_iq, q_data, v_data, w_data):
    """Plot Q, V, and W complex IQ time traces in a 3-by-1 layout."""
    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'lines.linewidth': 0.8,
    })
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15 / 2.54, 12 / 2.54),
        sharex=True,
        gridspec_kw={'hspace': 0},
    )
    time_flat = np.asarray(time_iq).flatten()
    q_flat = np.asarray(q_data).flatten() / 10000.0
    v_flat = np.asarray(v_data).flatten() / 10000.0
    w_flat = np.asarray(w_data).flatten() / 10000.0
    axes[0].plot(time_flat, np.real(q_flat), label='Re(Q)')
    axes[0].plot(time_flat, np.imag(q_flat), 'r', label='Im(Q)', alpha=0.8)
    axes[0].set_title(r'$\times 10^4$', loc='left', fontsize=8, pad=3)
    axes[0].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.8, edgecolor='none')
    axes[0].tick_params(direction='in', top=True, right=True, bottom=True, labelbottom=False)
    axes[1].plot(time_flat, np.real(v_flat), label='Re(V)')
    axes[1].plot(time_flat, np.imag(v_flat), 'r', label='Im(V)', alpha=0.8)
    axes[1].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.8, edgecolor='none')
    axes[1].tick_params(direction='in', top=True, right=True, bottom=True, labelbottom=False)
    axes[2].plot(time_flat, np.real(w_flat), label='Re(W)')
    axes[2].plot(time_flat, np.imag(w_flat), 'r', label='Im(W)', alpha=0.8)
    axes[2].set_xlabel('Time (ms)')
    axes[2].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.8, edgecolor='none')
    axes[2].tick_params(direction='in', top=True, right=True, bottom=True, labelbottom=True)
    fig.suptitle('I/Q Signal', fontsize=9)
    fig.supylabel('I/Q Signal (a.u.)', fontsize=8)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.show()
    return fig, axes

def plot_spectrogram(f, power, savepath=None, filename=None):
    """Plot power spectral density for the Q, V, and W bands."""
    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'lines.linewidth': 0.8,
    })
    colors_hex = [
        '#2166ACFF', '#4393C3FF', '#92C5DEFF', '#D1E5F0FF',
        '#FDDBC7FF', '#F4A582FF', '#D6604DFF', '#B71C2CFF',
    ]
    colors_rgb = [color[:7] for color in colors_hex]
    current_cmap = mcolors.LinearSegmentedColormap.from_list('custom_nf', colors_rgb)
    f_cpu = _asnumpy(f)
    power_cpu = _asnumpy(power)
    batch_size, n_segments, _ = power_cpu.shape
    segments = np.arange(n_segments)
    power_db = 10 * np.log10(np.abs(power_cpu) + 1e-12)
    fig, axes = plt.subplots(3, 1, figsize=(15 / 2.54, 12 / 2.54), sharex=False)
    band_names = ['Q Band', 'V Band', 'W Band']
    for i in range(batch_size):
        image = axes[i].pcolormesh(
            segments,
            f_cpu,
            power_db[i].T,
            shading='gouraud',
            cmap=current_cmap,
            vmin=np.percentile(power_db[i], 5),
            vmax=np.percentile(power_db[i], 99.5),
        )
        axes[i].set_ylabel('Freq. (MHz)')
        axes[i].set_xlabel('Window Segment Index')
        axes[i].text(
            0.98,
            0.85,
            band_names[i],
            transform=axes[i].transAxes,
            ha='right',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
        )
        fig.colorbar(image, ax=axes[i], pad=0.02)
        axes[i].tick_params(direction='in', top=True, right=True)
        axes[i].ticklabel_format(useOffset=False, style='plain', axis='x')
    fig.align_ylabels(axes)
    plt.tight_layout()
    if savepath is not None and filename is not None:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, filename), format='png', bbox_inches='tight', dpi=300)
    plt.show()
    return fig, axes

def plot_strategy_comparison(
    f, ind1, ind2, ind_w,
    fq_q, power_q, fb_q, fb_q_err,
    fq_v, power_v, fb_v, fb_v_err,
    fq_w, power_w, fb_w, fb_w_err,
):
    """Compare two screening strategies and the final result in a 1-by-3 layout."""
    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    })
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18 / 2.54, 6 / 2.54),
        sharex=False,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    power_w_safe = np.maximum(power_w, 1e-10)
    errorbar_format = {
        'fmt': '-o',
        'color': 'black',
        'markersize': 0.8,
        'linewidth': 0.7,
        'capsize': 1.2,
        'capthick': 0.5,
        'elinewidth': 0.5,
    }
    labels = ['(a)', '(b)', '(c)']
    strategy_indices = [ind1, ind2, ind_w]
    for i, axis in enumerate(axes):
        axis.imshow(
            np.log(power_w_safe),
            extent=[fq_w[0], fq_w[-1], f[0], f[-1]],
            aspect='auto',
            origin='lower',
            interpolation='bilinear',
            cmap='jet',
        )
        current_indices = strategy_indices[i]
        if current_indices is not None and len(current_indices) > 0:
            sorted_indices = np.sort(np.asarray(current_indices))
            valid_indices = sorted_indices[sorted_indices < len(fq_w)]
            if len(valid_indices) > 0:
                axis.errorbar(
                    fq_w[valid_indices],
                    fb_w[valid_indices],
                    yerr=fb_w_err[valid_indices],
                    **errorbar_format,
                )
        axis.text(
            0.5,
            -0.12,
            labels[i],
            transform=axis.transAxes,
            fontweight='bold',
            va='top',
            ha='center',
            color='black',
        )
        axis.tick_params(direction='in', top=True, right=True)
    fig.supxlabel('Frequency (GHz)', fontsize=8, y=0.05)
    fig.supylabel('Beat Frequency (MHz)', fontsize=8, x=0.03)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.05, bottom=0.15, left=0.1, right=0.98, top=0.95)
    plt.show()
    return fig, axes

def plot_all_bands(
    f, inds_q, inds_v, inds_w,
    fq_q, power_q, fb_q, fb_q_err,
    fq_v, power_v, fb_v, fb_v_err,
    fq_w, power_w, fb_w, fb_w_err,
    savepath=None,
    filename=None,
):
    """Plot power spectra and screened ranges for all bands in a 1-by-3 layout."""
    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    })
    fig, axes = plt.subplots(1, 3, figsize=(20 / 2.54, 10 / 2.54), sharex=False, sharey=True)
    band_data = {
        'Q': {
            'power': np.maximum(power_q, 1e-10),
            'fq': fq_q,
            'fb': fb_q,
            'fb_err': fb_q_err,
            'indices': inds_q,
        },
        'V': {
            'power': np.maximum(power_v, 1e-10),
            'fq': fq_v,
            'fb': fb_v,
            'fb_err': fb_v_err,
            'indices': inds_v,
        },
        'W': {
            'power': np.maximum(power_w, 1e-10),
            'fq': fq_w,
            'fb': fb_w,
            'fb_err': fb_w_err,
            'indices': inds_w,
        },
    }
    errorbar_format = {
        'fmt': '-o',
        'color': 'black',
        'markersize': 0.8,
        'linewidth': 0.7,
        'capsize': 1.2,
        'capthick': 0.5,
        'elinewidth': 0.5,
    }
    for axis, (band, data) in zip(axes, band_data.items()):
        axis.imshow(
            np.log(data['power']),
            extent=[data['fq'][0], data['fq'][-1], f[0], f[-1]],
            aspect='auto',
            origin='lower',
            interpolation='bilinear',
            cmap='jet',
        )
        indices = data['indices']
        if indices is not None and len(indices) > 0:
            sorted_indices = np.sort(np.asarray(indices))
            valid_indices = sorted_indices[sorted_indices < len(data['fq'])]
            if len(valid_indices) > 0:
                axis.errorbar(
                    data['fq'][valid_indices],
                    data['fb'][valid_indices],
                    yerr=data['fb_err'][valid_indices],
                    **errorbar_format,
                )
        axis.text(
            0.02,
            0.95,
            f'{band} Band',
            transform=axis.transAxes,
            fontsize=7,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
        )
        axis.tick_params(direction='in', top=True, right=True)
    fig.supxlabel('Frequency (GHz)', fontsize=8, y=0.05)
    fig.supylabel('Beat Frequency (MHz)', fontsize=8, x=0.03)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.23, bottom=0.15, left=0.1, right=0.98, top=0.95)
    if savepath is not None and filename is not None:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, filename), format='png', bbox_inches='tight', dpi=300)
    plt.show()
    return fig, axes

def plot_density_profile(
    radius,
    electron_density,
    shot,
    time_point,
    savepath=None,
    filename=None,
):
    """Plot the reconstructed radial electron-density profile."""
    plt.rcParams.update({
        'font.size': 6,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    })
    fig, axis = plt.subplots(figsize=(15 / 2.54, 10 / 2.54))
    radius_np = _asnumpy(radius)
    density_np = _asnumpy(electron_density)
    if radius_np is not None and density_np is not None:
        axis.plot(radius_np, density_np, '-o', markerfacecolor='none', linewidth=1, alpha=0.7)
    axis.set_xlabel('R (m)')
    axis.set_xlim(1.95, 2.4)
    axis.set_ylabel(r'$n_e$ ($10^{19}$ m$^{-3}$)')
    axis.set_title(f'Shot {shot} at {time_point:.2f}s')
    axis.tick_params(direction='in', top=True, right=True)
    plt.tight_layout()
    if savepath is not None and filename is not None:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, filename), format='png', bbox_inches='tight', dpi=300)
    plt.show()
    return fig, axis
