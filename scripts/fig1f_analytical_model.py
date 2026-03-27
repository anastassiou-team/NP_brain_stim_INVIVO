#!/usr/bin/env python3
"""
Figure 1 e–f: Analytical model of bipolar electrical stimulation.

Point-source approximation in a homogeneous resistive medium showing:
  • Panel f  – Ve equipotential contour map with probes at multiple angles
  • Extra 1  – Monopolar vs bipolar Ve contour comparison
  • Extra 2  – Ve amplitude profiles along probe channels
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config.plotting import (
    apply_nature_style, remove_top_right_spines,
    mm_to_inch, DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH,
)
from config.paths import FIGURES_OUTPUT
from src.analytical_model import (
    ve_monopolar, ve_bipolar, probe_coordinates,
    ve_along_probe, simulate_timeseries, instantaneous_amplitude,
)

# ── physical / electrode parameters ────────────────────────────────
RHO = 3.0             # extracellular resistivity (Ω·m)
I0 = 100e-6           # stimulation current (A)
FREQ = 8.0            # stimulation frequency (Hz)

PLUS_STIM  = [0.0, 0.0, 300e-6]   # anode position (m)
MINUS_STIM = [0.0, 0.0, 0.0]      # cathode position (m)

PROBE_LENGTH = 850e-6              # recording probe length (m)
N_ELEC = 40                        # number of channels
PROBE_OFFSET = [100e-6, 0.0, -200e-6]  # probe tip offset from cathode

PROBE_ANGLES = [0.0, np.pi / 20, np.pi / 12, np.pi / 8]   # tilt from z-axis
ANGLE_LABELS = ['0°', '9°', '15°', '22.5°']

# spatial mesh
MESH_MARGIN = 400e-6   # margin around electrodes (m)
MESH_STEP   = 5e-6     # grid resolution (m)

# unit conversion
UM = 1e6   # metres → µm
MV = 1e3   # volts  → mV

# probe colours (black → light grey, matching original notebook)
PROBE_COLORS = ['k', 'dimgray', 'darkgray', 'lightgray']
# highlight channel colours
CHANNEL_COLORS = ['tab:blue', 'tab:green', 'tab:red', 'tab:cyan']
CHANNEL_IDX = None  # set after N_ELEC is known


def _build_mesh():
    x_lo = min(PLUS_STIM[0], MINUS_STIM[0]) - MESH_MARGIN
    x_hi = max(PLUS_STIM[0], MINUS_STIM[0]) + MESH_MARGIN
    z_lo = min(PLUS_STIM[2], MINUS_STIM[2]) - MESH_MARGIN
    z_hi = max(PLUS_STIM[2], MINUS_STIM[2]) + MESH_MARGIN
    x = np.arange(x_lo, x_hi, MESH_STEP)
    z = np.arange(z_lo, z_hi, MESH_STEP)
    return np.meshgrid(x, z)


def _build_probes():
    return [probe_coordinates(PROBE_LENGTH, N_ELEC, PROBE_OFFSET, a)
            for a in PROBE_ANGLES]


def _highlight_indices():
    return [0, N_ELEC // 3, N_ELEC // 2, N_ELEC - 1]


# ────────────────────────────────────────────────────────────────────
# Figure helpers
# ────────────────────────────────────────────────────────────────────

def _draw_probes(ax, probes, ch_idx):
    """Overlay probe electrodes on an axis (coordinates in µm)."""
    for i, coords in enumerate(probes):
        ax.plot(coords[:, 0] * UM, coords[:, 2] * UM,
                '-o', color=PROBE_COLORS[i], linewidth=0.6,
                markersize=2.5, zorder=3)
        for j in ch_idx:
            ax.plot(coords[j, 0] * UM, coords[j, 2] * UM,
                    'o', color=CHANNEL_COLORS[CHANNEL_IDX.index(j)],
                    markersize=3.5, zorder=4)


def _draw_stim_electrodes(ax):
    ax.plot(PLUS_STIM[0] * UM, PLUS_STIM[2] * UM,
            's', color='red', markersize=4, zorder=5)
    ax.plot(MINUS_STIM[0] * UM, MINUS_STIM[2] * UM,
            's', color='blue', markersize=4, zorder=5)


# ────────────────────────────────────────────────────────────────────
# Panel generators
# ────────────────────────────────────────────────────────────────────

def panel_f(ax, xx, zz, V_bi, probes, ch_idx):
    """
    Panel f: bipolar Ve contour map with multi-angle probes.
    """
    levels = np.arange(-1.0, 1.1, 0.05)
    ax.contour(xx * UM, zz * UM, V_bi, levels,
               cmap='coolwarm', linewidths=0.6)
    _draw_probes(ax, probes, ch_idx)
    _draw_stim_electrodes(ax)
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('z (µm)')
    ax.set_title('bipolar ES – Ve contour')
    remove_top_right_spines(ax)


def panel_mono_vs_bi(ax_mono, ax_bi, xx, zz, V_mono, V_bi, probes, ch_idx):
    """
    Extra figure: side-by-side monopolar vs bipolar Ve contours.
    """
    levels = np.arange(-1.0, 1.1, 0.05)

    ax_mono.contour(xx * UM, zz * UM, V_mono, levels,
                    cmap='coolwarm', linewidths=0.6)
    _draw_probes(ax_mono, probes, ch_idx)
    _draw_stim_electrodes(ax_mono)
    ax_mono.set_xlabel('x (µm)')
    ax_mono.set_ylabel('z (µm)')
    ax_mono.set_title('monopolar')
    remove_top_right_spines(ax_mono)

    ax_bi.contour(xx * UM, zz * UM, V_bi, levels,
                  cmap='coolwarm', linewidths=0.6)
    _draw_probes(ax_bi, probes, ch_idx)
    _draw_stim_electrodes(ax_bi)
    ax_bi.set_xlabel('x (µm)')
    ax_bi.set_title('bipolar')
    remove_top_right_spines(ax_bi)


def panel_ve_profiles(axes, probes, ch_idx):
    """
    Extra figure: Ve amplitude vs channel for each probe angle.

    Top row: monopolar.  Bottom row: bipolar.
    One column per probe angle.
    """
    channels = np.arange(N_ELEC)

    for col, (coords, label) in enumerate(zip(probes, ANGLE_LABELS)):
        V_mono, V_bi = ve_along_probe(
            coords, PLUS_STIM, MINUS_STIM, I0, RHO)

        ax_m = axes[0, col]
        ax_b = axes[1, col]

        ax_m.plot(channels, V_mono * MV, '-', color='k', linewidth=0.8)
        ax_b.plot(channels, V_bi * MV, '-', color='k', linewidth=0.8)

        for j in ch_idx:
            c = CHANNEL_COLORS[CHANNEL_IDX.index(j)]
            ax_m.plot(channels[j], V_mono[j] * MV, 'o', color=c, markersize=3)
            ax_b.plot(channels[j], V_bi[j] * MV, 'o', color=c, markersize=3)

        ax_b.set_xlabel('channel')
        if col == 0:
            ax_m.set_ylabel('Ve (mV)')
            ax_b.set_ylabel('Ve (mV)')

        ax_m.set_title(f'θ = {label}')
        remove_top_right_spines(ax_m)
        remove_top_right_spines(ax_b)

    # shared y-limits
    all_ax = list(axes[0]) + list(axes[1])
    ymin = min(a.get_ylim()[0] for a in all_ax)
    ymax = max(a.get_ylim()[1] for a in all_ax)
    for a in all_ax:
        a.set_ylim(ymin, ymax)

    axes[0, 0].annotate('monopolar', xy=(0, 0.5),
                         xycoords='axes fraction', xytext=(-38, 0),
                         textcoords='offset points', rotation=90,
                         va='center', ha='center', fontsize=7,
                         fontweight='bold')
    axes[1, 0].annotate('bipolar', xy=(0, 0.5),
                         xycoords='axes fraction', xytext=(-38, 0),
                         textcoords='offset points', rotation=90,
                         va='center', ha='center', fontsize=7,
                         fontweight='bold')


def panel_amplitude_vs_distance(axes, probes):
    """
    Extra figure: instantaneous Ve amplitude vs distance from anode.

    Left: monopolar.  Right: bipolar.
    Each probe angle is a separate trace.
    """
    ax_mono, ax_bi = axes

    for i, (coords, label) in enumerate(zip(probes, ANGLE_LABELS)):
        V_mono_static, V_bi_static = ve_along_probe(
            coords, PLUS_STIM, MINUS_STIM, I0, RHO)

        # time-domain → Hilbert envelope
        V_mono_t, _ = simulate_timeseries(V_mono_static, FREQ)
        V_bi_t, _ = simulate_timeseries(V_bi_static, FREQ)
        A_mono = np.median(instantaneous_amplitude(V_mono_t), axis=-1)
        A_bi = np.median(instantaneous_amplitude(V_bi_t), axis=-1)

        dist_plus = np.sqrt(np.sum(
            (coords - np.array(PLUS_STIM)) ** 2, axis=1)) * UM

        ax_mono.plot(dist_plus, A_mono * MV, '-o',
                     color=PROBE_COLORS[i], linewidth=0.8,
                     markersize=2, label=label)
        ax_bi.plot(dist_plus, A_bi * MV, '-o',
                   color=PROBE_COLORS[i], linewidth=0.8,
                   markersize=2, label=label)

    ax_mono.set_title('monopolar')
    ax_bi.set_title('bipolar')
    for ax in (ax_mono, ax_bi):
        ax.set_xlabel('distance from anode (µm)')
        ax.legend(title='probe angle', fontsize=5, title_fontsize=5)
        remove_top_right_spines(ax)
    ax_mono.set_ylabel('Ve amplitude (mV)')


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    global CHANNEL_IDX
    apply_nature_style()

    # geometry
    xx, zz = _build_mesh()
    probes = _build_probes()
    CHANNEL_IDX = _highlight_indices()

    # static fields on the mesh
    V_mono = ve_monopolar(xx, zz, PLUS_STIM, I0, RHO)
    V_bi = ve_bipolar(xx, zz, PLUS_STIM, MINUS_STIM, I0, RHO)

    # ── Figure 1f (main output) ────────────────────────────────────
    fig_f, ax_f = plt.subplots(
        figsize=(mm_to_inch(SINGLE_COLUMN_WIDTH),
                 mm_to_inch(SINGLE_COLUMN_WIDTH)))
    panel_f(ax_f, xx, zz, V_bi, probes, CHANNEL_IDX)
    fig_f.tight_layout()
    os.makedirs(os.path.join(FIGURES_OUTPUT, 'fig1'), exist_ok=True)
    path_f = os.path.join(FIGURES_OUTPUT, 'fig1', 'figure1f_bipolar_model.png')
    fig_f.savefig(path_f, dpi=300, bbox_inches='tight')
    fig_f.savefig(path_f.replace('.png', '.pdf'), bbox_inches='tight')
    print(f'Saved {path_f}')

    # ── Extra 1: monopolar vs bipolar contours ─────────────────────
    fig1, (ax_m, ax_b) = plt.subplots(
        1, 2,
        figsize=(mm_to_inch(DOUBLE_COLUMN_WIDTH),
                 mm_to_inch(SINGLE_COLUMN_WIDTH)))
    panel_mono_vs_bi(ax_m, ax_b, xx, zz, V_mono, V_bi, probes, CHANNEL_IDX)
    fig1.tight_layout()
    path1 = os.path.join(FIGURES_OUTPUT, 'fig1', 'figure1_extra_mono_vs_bipolar.png')
    fig1.savefig(path1, dpi=300, bbox_inches='tight')
    print(f'Saved {path1}')

    # ── Extra 2: Ve profiles along channels ────────────────────────
    fig2, axes2 = plt.subplots(
        2, len(PROBE_ANGLES),
        figsize=(mm_to_inch(DOUBLE_COLUMN_WIDTH),
                 mm_to_inch(SINGLE_COLUMN_WIDTH)),
        sharex=True)
    panel_ve_profiles(axes2, probes, CHANNEL_IDX)
    fig2.tight_layout()
    path2 = os.path.join(FIGURES_OUTPUT, 'fig1', 'figure1_extra_ve_profiles.png')
    fig2.savefig(path2, dpi=300, bbox_inches='tight')
    print(f'Saved {path2}')

    # ── Extra 3: amplitude vs distance ─────────────────────────────
    fig3, (ax_am, ax_ab) = plt.subplots(
        1, 2,
        figsize=(mm_to_inch(DOUBLE_COLUMN_WIDTH),
                 mm_to_inch(0.5 * SINGLE_COLUMN_WIDTH)))
    panel_amplitude_vs_distance((ax_am, ax_ab), probes)
    fig3.tight_layout()
    path3 = os.path.join(FIGURES_OUTPUT, 'fig1', 'figure1_extra_amplitude_distance.png')
    fig3.savefig(path3, dpi=300, bbox_inches='tight')
    print(f'Saved {path3}')

    plt.close('all')


if __name__ == '__main__':
    main()
