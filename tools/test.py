"""Read headerless simulation data and compute a phase-shift dispersion image."""

from pathlib import Path

import numpy as np
from numpy.fft import rfft, rfftfreq


def read_binary_data(filename, nx, nt, npd, offsets, fs):
    """Read time-major little-endian float32 data and strip the two PML zones.

    The file layout is sample-major: for each time sample, it contains all
    ``nx`` spatial traces in order. ``offsets`` gives the signed source-receiver
    offsets of the retained traces, also in file order.
    """
    if nx <= 0 or nt <= 0:
        raise ValueError("nx and nt must be positive integers.")
    if npd < 0 or 2 * npd >= nx:
        raise ValueError("npd must be non-negative and leave at least one trace.")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be a positive finite value.")

    n_receivers = nx - 2 * npd
    x = np.asarray(offsets, dtype=float)
    if x.ndim != 1 or x.size != n_receivers:
        raise ValueError(
            f"Provide {n_receivers} signed offsets for the retained traces; "
            f"received an array with shape {x.shape}."
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("All offsets must be finite.")

    data = np.fromfile(filename, dtype="<f4")
    expected_count = nt * nx
    if data.size != expected_count:
        raise ValueError(
            f"{filename} contains {data.size} float32 values; "
            f"expected nt*nx={expected_count}."
        )

    raw_gather = data.reshape((nt, nx), order="C")
    gather = raw_gather[:, npd:nx - npd]
    if not np.all(np.isfinite(gather)):
        raise ValueError("The binary data contains NaN or infinite values.")

    time = np.arange(nt, dtype=float) / fs
    return gather, time, x


def dispersion_imaging(gather, offsets, fs, c_min, c_max, dc):
    """Compute normalized phase-shift amplitude on a frequency-velocity grid.

    Returns ``frequency_grid, velocity_grid, amplitude``. Each returned array
    has shape ``(nt // 2 + 1, number_of_test_velocities)``.
    """
    gather = np.asarray(gather)
    x = np.asarray(offsets, dtype=float)
    if gather.ndim != 2:
        raise ValueError("gather must have shape (nt, number_of_receivers).")
    if x.ndim != 1 or x.size != gather.shape[1]:
        raise ValueError("There must be exactly one offset for each trace.")
    if not np.all(np.isfinite(x)):
        raise ValueError("All offsets must be finite.")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be a positive finite value.")
    if not np.isfinite(c_min) or not np.isfinite(c_max) or c_min <= 0 or c_max < c_min:
        raise ValueError("Velocities must satisfy 0 < c_min <= c_max.")
    if not np.isfinite(dc) or dc <= 0:
        raise ValueError("dc must be a positive finite value.")

    spectrum = rfft(gather, axis=0)
    frequencies = rfftfreq(gather.shape[0], d=1.0 / fs)
    omega = 2.0 * np.pi * frequencies
    magnitude = np.abs(spectrum)
    phase_spectrum = np.divide(
        spectrum.conj(),
        magnitude,
        out=np.zeros_like(spectrum),
        where=magnitude > 0,
    )

    velocities = np.arange(c_min, c_max + dc / 2.0, dc)
    if velocities.size == 0:
        raise ValueError("The requested velocity range has no test values.")

    frequency_grid = np.tile(frequencies[:, None], (1, velocities.size))
    velocity_grid = np.tile(velocities[None, :], (frequencies.size, 1))
    amplitude = np.empty_like(frequency_grid)

    # For each frequency, compensate the phase at every offset and stack traces.
    for i, angular_frequency in enumerate(omega):
        shifts = np.exp(
            -1j * angular_frequency * x[:, None] / velocities[None, :]
        )
        stacked = np.sum(phase_spectrum[i, :, None] * shifts, axis=0)
        amplitude[i, :] = np.abs(stacked) / gather.shape[1]

    return frequency_grid, velocity_grid, amplitude


def omega_from_c(c, alpha_w, rho_w, alpha, beta, rho, thickness, mode=0):
    """Compute theoretical angular frequency for phase velocity ``c``.

    Velocities outside ``alpha_w < c < min(alpha, beta)`` return NaN.
    ``c`` may be a scalar or an array.
    """
    if alpha_w <= 0 or alpha <= 0 or beta <= 0:
        raise ValueError("All wave speeds must be positive.")
    if rho_w <= 0 or rho <= 0 or thickness <= 0:
        raise ValueError("Densities and layer thickness must be positive.")
    if not isinstance(mode, (int, np.integer)) or mode < 0:
        raise ValueError("mode must be a non-negative integer.")

    scalar = np.isscalar(c)
    c = np.atleast_1d(np.asarray(c, dtype=np.float64))
    omega = np.full_like(c, np.nan)
    valid = (
        np.isfinite(c)
        & (c > alpha_w)
        & (c < alpha)
        & (c < beta)
    )
    if np.any(valid):
        cv = c[valid]
        a = cv**2 / alpha_w**2 - 1.0
        b = 1.0 - cv**2 / alpha**2
        q = 1.0 - cv**2 / beta**2
        d = 1.0 / alpha_w**2 - 1.0 / cv**2
        ratio = (
            rho * beta**4 * np.sqrt(a)
            / (rho_w * cv**4 * np.sqrt(b))
        )
        bracket = (
            4.0 * np.sqrt(q) * np.sqrt(b)
            - (cv**2 / beta**2 - 2.0) ** 2
        )
        phase = np.arctan(ratio * bracket) + mode * np.pi
        omega[valid] = phase / (thickness * np.sqrt(d))

    return float(omega[0]) if scalar else omega


def run_analysis():
    gather, time, offsets = read_binary_data(
        BINARY_FILE, NX, NT, NPD, OFFSETS, FS
    )

    trace_start, trace_end = TRACE_RANGE
    if trace_start < 1 or trace_end > gather.shape[1] or trace_start > trace_end:
        raise ValueError(
            f"TRACE_RANGE must be an inclusive 1-based range within "
            f"1..{gather.shape[1]}."
        )
    gather = gather[:, trace_start - 1:trace_end]
    offsets = offsets[trace_start - 1:trace_end]

    time_start, time_end = TIME_WINDOW
    if time_start < time[0] or time_end > time[-1] or time_start >= time_end:
        raise ValueError(
            f"TIME_WINDOW must satisfy {time[0]:g} <= start < end <= {time[-1]:g} seconds."
        )
    time_mask = (time >= time_start) & (time <= time_end)
    gather = gather[time_mask, :]
    time = time[time_mask]
    if gather.shape[0] < 2:
        raise ValueError("TIME_WINDOW must include at least two samples.")

    frequency, velocity, amplitude = dispersion_imaging(
        gather, offsets, FS, C_MIN, C_MAX, DC
    )
    frequency_axis = frequency[:, 0]
    frequency_min, frequency_max = FREQUENCY_RANGE
    if frequency_min < 0 or frequency_max > FS / 2 or frequency_min >= frequency_max:
        raise ValueError(
            f"FREQUENCY_RANGE must satisfy 0 <= min < max <= Nyquist "
            f"({FS / 2:g} Hz)."
        )
    frequency_mask = (
        (frequency_axis >= frequency_min) & (frequency_axis <= frequency_max)
    )
    if not np.any(frequency_mask):
        raise ValueError("FREQUENCY_RANGE contains no FFT frequency bins.")

    frequency_axis = frequency_axis[frequency_mask]
    velocity_axis = velocity[0, :]
    amplitude = amplitude[frequency_mask, :]
    if USE_CONTOURF and frequency_axis.size < 2:
        raise ValueError(
            "contourf needs at least two frequency bins; widen FREQUENCY_RANGE."
        )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    if USE_CONTOURF:
        image = ax.contourf(
            frequency_axis,
            velocity_axis,
            amplitude.T,
            levels=np.linspace(0.0, 1.0, CONTOUR_LEVELS),
            cmap="turbo",
            vmin=0.0,
            vmax=1.0,
        )
    else:
        image = ax.pcolormesh(
            frequency_axis,
            velocity_axis,
            amplitude.T,
            shading="auto",
            cmap="turbo",
            vmin=0.0,
            vmax=1.0,
        )
    ax.set_xlim(frequency_min, frequency_max)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity (m/s)")
    # ax.set_title("Phase-shift Dispersion Image")

    theory_c_min = max(THEORY_ALPHA_W, C_MIN)
    theory_c_max = min(THEORY_ALPHA, THEORY_BETA, C_MAX)
    if theory_c_min >= theory_c_max:
        raise ValueError(
            "Theoretical valid velocity range must overlap the imaged "
            "velocity range (alpha_w < c < min(alpha, beta))."
        )
    theory_c = np.linspace(
        max(THEORY_ALPHA_W * (1.0 + 1e-8), theory_c_min),
        min(
            THEORY_ALPHA * (1.0 - 1e-8),
            THEORY_BETA * (1.0 - 1e-8),
            theory_c_max,
        ),
        THEORY_VELOCITY_SAMPLES,
    )
    for mode in THEORY_MODES:
        theory_frequency = omega_from_c(
            theory_c,
            THEORY_ALPHA_W,
            THEORY_RHO_W,
            THEORY_ALPHA,
            THEORY_BETA,
            THEORY_RHO,
            THEORY_THICKNESS,
            mode=mode,
        ) / (2.0 * np.pi)
        curve_mask = (
            np.isfinite(theory_frequency)
            & (theory_frequency >= frequency_min)
            & (theory_frequency <= frequency_max)
        )
        if np.any(curve_mask):
            ax.plot(
                theory_frequency[curve_mask],
                theory_c[curve_mask],
                linewidth=1.8,
                label=f"Theory mode {mode}",
            )
    ax.legend(loc="best")
    fig.colorbar(image, ax=ax, label="Normalized phase-stack amplitude")
    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=200)
    print(
        f"Selected traces: {trace_start}-{trace_end} "
        f"({gather.shape[1]} traces)"
    )
    print(
        f"Selected time window: {time[0]:.6g}-{time[-1]:.6g} s "
        f"({gather.shape[0]} samples)"
    )
    print(
        f"Displayed frequency range: "
        f"{frequency_axis[0]:.6g}-{frequency_axis[-1]:.6g} Hz"
    )
    print(f"Saved dispersion image to: {OUTPUT_FILE}")
    plt.show()
    plt.close(fig)


# Set the simulation and imaging parameters here.
BINARY_FILE = Path("./output/ortho_freesurface_scholte/0/record/solid_vz.bin")
NX = 667                   # Total traces in the binary file, including PML
NT = 3281                  # Number of time samples
NPD = 33                   # PML traces removed from each side
FS = 1000.0                # Sampling frequency [Hz]
OFFSETS = np.arange(601) * 2 - 20  # Signed offsets [m], retained traces in file order
TRACE_RANGE = (251, 600)   # Inclusive 1-based range within the PML-trimmed gather
TIME_WINDOW = (0.3, 3.25)  # Inclusive time window [s]
C_MIN = 1500.0             # Minimum test phase velocity [m/s]
C_MAX = 1800.0             # Maximum test phase velocity [m/s]
DC = 1.0                   # Test phase-velocity increment [m/s]
FREQUENCY_RANGE = (0.0, 80.0)  # Display range [Hz]; <= Nyquist
USE_CONTOURF = False        # True: smooth contour image; False: raw-grid pcolormesh
CONTOUR_LEVELS = 102       # Number of contour levels when USE_CONTOURF is True
OUTPUT_FILE = Path("dispersion_image.png")

# Theoretical fluid-solid interface model parameters.
THEORY_ALPHA_W = 1500.0    # Fluid P-wave speed [m/s]
THEORY_RHO_W = 1000.0      # Fluid density [kg/m^3]
THEORY_ALPHA = 3200.0      # Solid P-wave speed [m/s]
THEORY_BETA = 1800.0       # Solid S-wave speed [m/s]
THEORY_RHO = 2000.0        # Solid density [kg/m^3]
THEORY_THICKNESS = 100.0   # Characteristic layer thickness H [m]
THEORY_MODES = (0, 1, 2, 3, 4)   # Mode orders to plot
THEORY_VELOCITY_SAMPLES = 4000


if __name__ == "__main__":
    run_analysis()
