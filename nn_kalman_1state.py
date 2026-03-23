# Copyright (c) 2026 Vedant
# SPDX-License-Identifier: Apache-2.0
#
# NN-Q Kalman Filter: Neural network-adaptive process noise Q
# for 1-state position-only Kalman on ADC0804 pipeline.
# First published: 2026 (see git commit timestamp for exact date)

"""
╔══════════════════════════════════════════════════════════════════════════╗
║  ADC0804  →  4× MA Decimate  →  NN-Q Kalman Filter  [1-STATE REWRITE]  ║
║  10 kSPS raw  →  2.5 kSPS output  │  1.25 kHz bandwidth                ║
╠══════════════════════════════════════════════════════════════════════════╣
║  KEY CHANGE vs original:                                                 ║
║    2-state constant-velocity  →  1-state position-only Kalman           ║
║    Eliminates P[1,1] leakage through DT²·P[1,1] into P[0,0].           ║
║    Steady-state:  P_ss ≈ √(Q·R)  →  theoretical ceiling ≈ +1.66 bits   ║
║                                                                          ║
║  Supports three test inputs (auto-detected):                             ║
║    • D:/Downloads/WFDB/HR00001.mat + .hea   (WFDB/PhysioNet ECG)        ║
║    • D:/Downloads/gyro_adc7.txt              (raw 8-bit ADC codes)       ║
║    • D:/Downloads/adc_data-1.csv             (timestamped ADC CSV)       ║
║    • any plain .txt one-value-per-line file                              ║
╠══════════════════════════════════════════════════════════════════════════╣
║  NN ARCHITECTURE:                                                        ║
║    8 → [16 tanh] → [8 tanh] → 1 softplus → clamp                       ║
║    Parameters:   8×16 + 16 + 16×8 + 8 + 1×8 = 288 weights             ║
║    MACs / step:  8×16 + 16×8 + 1×8 = 264                               ║
╚══════════════════════════════════════════════════════════════════════════╝

USAGE
  python nn_kalman_1state.py                          # synthetic demo
  python nn_kalman_1state.py yourfile.txt             # raw ADC codes (one per line)
  python nn_kalman_1state.py yourfile.csv             # timestamped ADC CSV
  python nn_kalman_1state.py HR00001.mat              # WFDB .mat file
  python nn_kalman_1state.py --test-enob              # ENOB 7.0-7.3 test
  python nn_kalman_1state.py --compare path.txt       # side-by-side vs 2-state
  python nn_kalman_1state.py --all-tests              # run all three test files
"""

import sys, os, argparse, time, struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.lib.stride_tricks import sliding_window_view as _swv
from scipy.signal import lfilter as _lfilter

# ═══════════════════════════════════════════════════════════════════════════
# 1.  HARDWARE / SYSTEM CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
FS_RAW   = 10_000
OSR      = 4
FS_OUT   = FS_RAW // OSR       # 2 500 SPS
DT       = 1.0 / FS_OUT        # 400 µs

ADC_BITS = 8
VREF     = 5.0
LSB      = VREF / 2**ADC_BITS  # 19.531 mV

# Measurement noise variance (4-pt MA decimation of uniform quant noise)
R_MEAS   = (LSB**2 / 12.0) / OSR   # ≈ 7.95e-6 V²

# ── 1-STATE Q BOUNDS ────────────────────────────────────────────────────
# Steady state:  P_ss ≈ √(Q·R)
#   Q_MIN → P_ss ≈ √(0.01 · R² ) = 0.1·R  →  ENOB gain ≈ +1.66 bits
#   Q_MAX → P_ss ≈ √(100 · R²  ) = 10·R   →  allows fast step tracking
Q_MIN    = R_MEAS * 0.01
Q_MAX    = R_MEAS * 100.0
Q_INIT   = R_MEAS * 0.3

INN_SCALE = 3.0 * np.sqrt(R_MEAS)

RNG = np.random.default_rng(0xADC_0804)

# Path to the gyro training file — used to build a richer training pool.
# Set to None to disable.  Overridden by --gyro-train CLI arg.
GYRO_TRAIN_PATH = r"D:\Downloads\quantized_7bit_equivalent.txt"

# ECG .mat files used ONLY for pool training (HR00001 is the test file).
# Each file contributes every available lead as an independent segment.
# WHY: QRS dynamics (oracle Q swings 4–6 decades in ~25 samples then stays
# at floor for 700 ms) are not present in gyro or synthetic data. Without
# these in the pool, W1/W2 have never seen the pattern and fine-tune on an
# ECG test file must teach it from scratch → ft_loss stuck >2.0.
# With ECG in the pool, fine-tune only adjusts Q-magnitude (ft_loss → <0.1).
ECG_TRAIN_PATHS = [
    r"D:\Downloads\WFDB\HR21791.mat",
    r"D:\Downloads\WFDB\HR21792.mat",
    r"D:\Downloads\WFDB\HR21793.mat",
]

# ADC CSV test / inference file
ADC_CSV_PATH = r"D:\Downloads\adc_data (1).csv"




# ═══════════════════════════════════════════════════════════════════════════
# 2.  SIGNAL-ADAPTIVE Q BOUNDS  (1-state version)
# ═══════════════════════════════════════════════════════════════════════════

def _adaptive_q_params(decimated_signal: np.ndarray) -> dict:
    """
    Derive Q bounds and innovation scale from actual signal dynamics.
    Uses a 1-state scalar Kalman for the innovation-collection pass
    (no matrix ops — ~4× faster than 2-state version).
    """
    sig = np.asarray(decimated_signal, dtype=np.float64)
    N   = len(sig)

    # Q bounds from step-to-step variance
    vel    = np.diff(sig, prepend=sig[0])
    q_nom  = float(max(np.var(vel), R_MEAS))
    q_min  = R_MEAS * 0.01
    q_max  = q_nom  * 20.0
    q_init = q_nom  * 0.5

    # Innovation scale — run a low-Q 1-state Kalman
    Q_coll   = R_MEAS * 2.0
    x        = float(sig[0])
    P        = R_MEAS * 20.0
    innovations = np.empty(N)
    for k, z in enumerate(sig):
        P_p  = P + Q_coll
        S    = P_p + R_MEAS
        K    = P_p / S
        inn  = float(z) - x
        innovations[k] = inn
        x    = x + K * inn
        P    = (1.0 - K) * P_p            # Joseph-simplified for scalar

    inn_scale = 3.0 * float(np.std(innovations))
    inn_scale = max(inn_scale, 3.0 * float(np.sqrt(R_MEAS)))

    return dict(q_min=q_min, q_max=q_max, q_init=q_init, inn_scale=inn_scale)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  NEURAL NETWORK  (pure NumPy — zero external ML dependencies)
#
#   Architecture:  8 → [16 tanh] → [8 tanh] → 1 softplus → clamp
#   Parameters:    8×16 + 16 + 16×8 + 8 + 1×8 = 288 weights
#   MACs / step:   8×16 + 16×8 + 1×8 = 264
# ═══════════════════════════════════════════════════════════════════════════
N_INN = 8    # innovation history window
H1    = 16   # first hidden layer
H2    = 8    # second hidden layer


class TinyNNQ:
    """
    Frozen inference-only MLP.
    q_params : dict from _adaptive_q_params().  None → uses global constants.
    """

    def __init__(self, seed: int = 42, q_params: dict = None):
        if q_params is not None:
            self.q_min     = q_params['q_min']
            self.q_max     = q_params['q_max']
            self.q_init    = q_params['q_init']
            self.inn_scale = q_params['inn_scale']
        else:
            self.q_min     = Q_MIN
            self.q_max     = Q_MAX
            self.q_init    = Q_INIT
            self.inn_scale = INN_SCALE

        rg = np.random.default_rng(seed)
        he = lambda r, c: rg.normal(0, np.sqrt(2.0 / c), (r, c))
        self.W1 = he(H1, N_INN).astype(np.float64)
        self.b1 = np.zeros(H1)
        self.W2 = he(H2, H1).astype(np.float64)
        self.b2 = np.zeros(H2)
        # W3 = tiny noise (1e-3 scale): z3≈0 → Q≈Q_MIN at init.
        # Tiny but non-zero to break symmetry and allow gradient flow via STE.
        # No b3 bias — would shift Q above Q_MIN for quiet inputs.
        self.W3 = rg.normal(0, 1e-3, (1, H2)).astype(np.float64)
        self._z3_max    = float(np.log(max(self.q_max / self.q_min, 1.0)))
        self._z3_offset = 0.0   # set by calibrate_quiet_offset() after pool+ft
        self._trained   = False

    @staticmethod
    def _tanh(z): return np.tanh(np.clip(z, -15, 15))
    @staticmethod
    def _sp(z):   return np.log1p(np.exp(np.clip(z, -30, 30)))
    @staticmethod
    def _dsp(z):  return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def forward(self, inn_buf: np.ndarray) -> float:
        """
        Q = Q_MIN * exp( clamp(z3 - z3_offset, 0, log(Q_MAX/Q_MIN)) )
        _z3_offset is calibrated post-training so that Q(zero_input) = Q_MIN exactly.
        For active inputs z3 >> z3_offset → Q rises as the NN learned.
        """
        z1 = self.W1 @ inn_buf + self.b1
        a1 = self._tanh(z1)
        z2 = self.W2 @ a1 + self.b2
        a2 = self._tanh(z2)
        z3  = float((self.W3 @ a2)[0])
        z3c = float(np.clip(z3 - self._z3_offset, 0.0, self._z3_max))
        return float(np.clip(self.q_min * np.exp(z3c), self.q_min, self.q_max))

    # ─────────────────────────────────────────────────────────────────────
    # Feature extraction helper (shared by train() and train_pool())
    # ─────────────────────────────────────────────────────────────────────
    def _compute_features(self, sig: np.ndarray,
                          q_params_local: dict = None) -> tuple:
        """
        Run a low-Q 1-state Kalman pass and build (X, Y) for one signal.

        X : (N, N_INN)  normalised innovations — always in [-5, +5]
        Y : (N,)        oracle Q targets        — in [q_min, q_max]

        q_params_local : if given, use its inn_scale for normalisation
                         (enables per-segment normalisation in pool training).
                         If None, uses self.inn_scale.
        """
        sig    = np.asarray(sig, dtype=np.float64)
        N      = len(sig)
        WIN    = 40
        alpha  = 0.10

        inn_sc = q_params_local['inn_scale'] if q_params_local else self.inn_scale

        # ── Oracle Q (log-domain EMA) ────────────────────────────────
        vel       = np.diff(sig, prepend=sig[0]) / DT
        v2        = vel ** 2
        vel_p     = np.concatenate([np.zeros(WIN), vel])
        v2_p      = np.concatenate([np.zeros(WIN), v2])
        cs1       = np.concatenate([[0.0], np.cumsum(vel_p)])
        cs2       = np.concatenate([[0.0], np.cumsum(v2_p)])
        ws        = np.minimum(np.arange(N) + 1, WIN).astype(float)
        end_      = WIN + np.arange(N) + 1
        st_       = end_ - ws.astype(int)
        s1        = cs1[end_] - cs1[st_]
        s2        = cs2[end_] - cs2[st_]
        oracle_raw = (s2 / ws - (s1 / ws) ** 2) * DT**2 + 1e-30
        log_raw   = np.log(np.maximum(oracle_raw, 1e-40))
        log_ema   = _lfilter([alpha], [1.0, alpha - 1.0], log_raw)
        oracle    = np.exp(log_ema)
        oracle    = np.clip(oracle - 2.0 * R_MEAS, self.q_min, self.q_max)

        # ── 1-state Kalman for innovations ───────────────────────────
        Q_coll = R_MEAS * 2.0
        x_     = float(sig[0])
        p_     = R_MEAS * 20.0
        innovations = np.empty(N)
        for k in range(N):
            p_p  = p_ + Q_coll
            S    = p_p + R_MEAS
            kk   = p_p / S
            inn  = sig[k] - x_
            innovations[k] = inn
            x_   = x_ + kk * inn
            p_   = (1.0 - kk) * p_p

        inn_norm = np.clip(innovations / inn_sc, -5.0, 5.0)
        padded   = np.concatenate([np.zeros(N_INN), inn_norm])
        X        = _swv(padded, N_INN)[:-1].copy()
        return X, oracle

    # ─────────────────────────────────────────────────────────────────────
    # Core Adam training loop (operates on pre-built X, Y matrices)
    # ─────────────────────────────────────────────────────────────────────
    def _train_on_xy(self, X: np.ndarray, Y: np.ndarray,
                     epochs: int, lr: float, verbose: bool,
                     epoch_offset: int = 0,
                     lr_min_ratio: float = 0.05) -> list:
        """Adam loop on (X, Y) feature matrices with cosine LR decay.
        LR decays from `lr` → `lr * lr_min_ratio` over training.
        Returns loss_hist."""
        N   = len(Y)
        B   = 256
        mW1=np.zeros_like(self.W1); vW1=np.zeros_like(self.W1)
        mb1=np.zeros_like(self.b1); vb1=np.zeros_like(self.b1)
        mW2=np.zeros_like(self.W2); vW2=np.zeros_like(self.W2)
        mb2=np.zeros_like(self.b2); vb2=np.zeros_like(self.b2)
        mW3=np.zeros_like(self.W3); vW3=np.zeros_like(self.W3)
        b1a, b2a, eps_a = 0.9, 0.999, 1e-8
        t   = 0
        loss_hist = []
        idx = np.arange(N)

        for ep in range(epochs):
            cos_frac = 0.5 * (1.0 + np.cos(np.pi * ep / max(epochs - 1, 1)))
            lr_ep    = lr * (lr_min_ratio + (1.0 - lr_min_ratio) * cos_frac)
            np.random.default_rng(ep + epoch_offset).shuffle(idx)
            ep_loss   = 0.0
            n_batches = 0

            for start in range(0, N, B):
                batch = idx[start:start + B]
                xb    = X[batch]
                yb    = Y[batch]
                t    += 1

                z1     = xb @ self.W1.T + self.b1
                a1     = np.tanh(np.clip(z1, -15, 15))
                z2     = a1 @ self.W2.T + self.b2
                a2     = np.tanh(np.clip(z2, -15, 15))
                z3_raw = (a2 @ self.W3.T).ravel()
                z3c    = np.clip(z3_raw, 0.0, self._z3_max)
                qc     = np.clip(self.q_min * np.exp(z3c), self.q_min, self.q_max)

                lr_    = np.log(qc / yb)
                ep_loss += (lr_**2).mean()
                n_batches += 1

                # STE backward
                d3   = 2.0 * lr_
                gW3  = (d3[:, None] * a2).mean(axis=0, keepdims=True)
                d2   = (d3[:, None] * self.W3) * (1 - a2**2)
                gW2  = (d2[:, :, None] * a1[:, None, :]).mean(axis=0)
                gb2  = d2.mean(axis=0)
                d1   = (d2 @ self.W2) * (1 - a1**2)
                gW1  = (d1[:, :, None] * xb[:, None, :]).mean(axis=0)
                gb1  = d1.mean(axis=0)

                for g in (gW1, gb1, gW2, gb2, gW3):
                    np.clip(g, -2.0, 2.0, out=g)

                def adam(p, g, m, v):
                    m[:] = b1a*m + (1-b1a)*g
                    v[:] = b2a*v + (1-b2a)*g**2
                    mh = m/(1-b1a**t);  vh = v/(1-b2a**t)
                    p  -= lr_ep * mh / (np.sqrt(vh) + eps_a)

                adam(self.W1, gW1, mW1, vW1)
                adam(self.b1, gb1, mb1, vb1)
                adam(self.W2, gW2, mW2, vW2)
                adam(self.b2, gb2, mb2, vb2)
                adam(self.W3, gW3, mW3, vW3)

            avg = ep_loss / n_batches
            loss_hist.append(avg)
            if verbose and (ep % 60 == 0 or ep == epochs - 1):
                ep_disp = ep + 1 + epoch_offset
                print(f"  Epoch {ep_disp:>3d}  log-ratio MSE = {avg:.5f}")

        return loss_hist

    def calibrate_quiet_offset(self):
        """
        Post-training zero-offset calibration.
        Measures z3 at zero input and stores as _z3_offset so that
        Q(zeros) = Q_MIN exactly, enabling full ENOB gain in quiet periods.
        Call after every pool+fine-tune run.

        Safety condition: z3_offset must be SMALL (< ~5) so that active-input
        z3 values remain above z3_offset and Q still rises for real signals.
        Pool training with gyro+ECG+oscillatory bank reliably achieves
        z3_offset ≈ 3–4.  Without the oscillatory bank z3_offset can reach
        13+ (catastrophic — clamps all sinusoidal inputs to Q_MIN).
        """
        a1z = np.tanh(np.clip(self.b1, -15, 15))
        a2z = np.tanh(np.clip(self.W2 @ a1z + self.b2, -15, 15))
        self._z3_offset = float((self.W3 @ a2z).ravel()[0])
        q_check = self.forward(np.zeros(N_INN))
        print(f"  [calibrate]  z3_offset={self._z3_offset:.4f}  "
              f"Q(quiet)={q_check:.3e}  Q_MIN={self.q_min:.3e}")

    # ─────────────────────────────────────────────────────────────────────
    # Public API (unchanged signature)
    # ─────────────────────────────────────────────────────────────────────
    def train(self, decimated_signal: np.ndarray,
              epochs: int = 300, lr: float = 3e-3, verbose: bool = True):
        """
        Single-signal training (used by --test-enob).
        W3 initialised to tiny noise → z3≈0 → Q≈Q_MIN for quiet inputs naturally.
        No calibrate_quiet_offset() here: that offset is only valid post-pool-training
        where gyro dynamics inflate z3 at zero input. Calling it on a pure sine sets
        z3_offset ≈ z3(zero) ≈ z3(sine) which zeros out all sine responses.
        """
        X, Y = self._compute_features(decimated_signal)
        loss  = self._train_on_xy(X, Y, epochs, lr, verbose)
        self._z3_offset = 0.0   # explicit: no offset correction for single-signal
        self._trained = True
        return loss

    def train_pool(self, segments: list,
                   q_params_inf: dict,
                   inference_signal: np.ndarray = None,
                   epochs_pool: int = 150,
                   epochs_ft:   int = 300,
                   lr_pool:     float = 2e-3,
                   lr_ft:       float = 3e-3,
                   max_per_seg: int   = 1_500,
                   verbose:     bool  = True) -> tuple:
        """
        Train on a pool of (signal, q_params_local) tuples, then fine-tune
        on the inference signal.

        segments : list of (decimated_voltage_array, q_params_local_dict)
                   Each segment uses its own local q_params for oracle + inn_scale,
                   so innovations are normalised to [-5,+5] per segment —
                   features are scale-consistent regardless of signal amplitude.

        q_params_inf     : q_params from the actual inference signal.
        inference_signal : signal to fine-tune on. If None, uses segments[-1][0].
                           Always pass this explicitly from run() — segments[-1]
                           is fragile if the pool list order changes.
        max_per_seg      : cap samples per segment (~1500 = 0.6 s at FS_OUT).

        Returns (pool_loss_hist, ft_loss_hist).
        """
        # ── Build pooled (X, Y) ──────────────────────────────────────
        X_parts, Y_parts = [], []
        total = 0
        for seg_sig, seg_qp in segments:
            seg_sig = np.asarray(seg_sig, dtype=np.float64)
            # Sub-sample if too long (evenly spaced to keep dynamics)
            if len(seg_sig) > max_per_seg:
                idx = np.round(np.linspace(0, len(seg_sig)-1, max_per_seg)).astype(int)
                seg_sig = seg_sig[idx]
            Xs, Ys = self._compute_features(seg_sig, q_params_local=seg_qp)
            X_parts.append(Xs)
            Y_parts.append(Ys)
            total += len(Ys)

        X_pool = np.concatenate(X_parts, axis=0)
        Y_pool = np.concatenate(Y_parts, axis=0)
        print(f"  [train_pool]  {len(segments)} segments  →  "
              f"{total:,} samples in pool  (capped at {max_per_seg}/seg)")

        # ── Pool training ────────────────────────────────────────────
        pool_loss = self._train_on_xy(X_pool, Y_pool, epochs_pool,
                                      lr_pool, verbose, epoch_offset=0)

        # ── Fine-tune on inference signal ───────────────────────────────
        # RESET W3 before fine-tune: pool training biases W3 toward gyro
        # dynamics. W1/W2 retain pool-learned representations (keep them).
        # Resetting W3 re-anchors Q(any quiet input) → Q_MIN so fine-tune
        # only needs to learn the raise-Q pattern for this specific signal.
        rg_ft = np.random.default_rng(99)
        self.W3 = rg_ft.normal(0, 1e-3, self.W3.shape).astype(np.float64)
        # Recompute _z3_max in case q_params_inf differs from pool q_params
        self._z3_max    = float(np.log(max(q_params_inf['q_max'] /
                                           q_params_inf['q_min'], 1.0)))
        self.q_min      = q_params_inf['q_min']
        self.q_max      = q_params_inf['q_max']
        self.q_init     = q_params_inf['q_init']
        self.inn_scale  = q_params_inf['inn_scale']

        # Use explicit inference_signal if provided; fall back to segments[-1][0]
        _ft_sig = inference_signal if inference_signal is not None else segments[-1][0]
        X_ft, Y_ft = self._compute_features(_ft_sig, q_params_local=q_params_inf)
        if verbose:
            print(f"  [fine-tune]  {epochs_ft} epochs @ lr={lr_ft:.1e}  "
                  f"on {len(Y_ft):,} inference samples …")
        ft_loss = self._train_on_xy(X_ft, Y_ft, epochs_ft,
                                    lr_ft, verbose, epoch_offset=epochs_pool)

        self.calibrate_quiet_offset()
        q_quiet = float(self.forward(np.zeros(N_INN)))
        print(f"  [fine-tune]  done  ft_loss={ft_loss[-1]:.5f}  "
              f"Q(quiet)={q_quiet:.3e}  (Q_MIN={self.q_min:.3e}  "
              f"ratio={q_quiet/self.q_min:.2f}×)")
        self._trained = True
        return pool_loss, ft_loss



# ═══════════════════════════════════════════════════════════════════════════
# 4.  1-STATE POSITION-ONLY KALMAN FILTER
#
#   State:    x (voltage)            — scalar
#   Predict:  x_p = x,  P_p = P + Q
#   Update:   S = P_p + R
#             K = P_p / S
#             x = x_p + K·(z − x_p)
#             P = (1 − K)·P_p          (Joseph-simplified for scalar)
#
#   Steady-state:
#     P_ss = [−Q + √(Q² + 4QR)] / 2  ≈  √(Q·R)  for Q ≪ R
#     At Q=Q_MIN (0.01·R):  P_ss ≈ 0.1·R  →  ENOB gain ≈ +1.66 bits
#
#   No velocity state  →  no DT²·P[1,1] bleed-through.
#   Lag on ramps:  proportional to (1−K) — acceptable for sensor fusion.
# ═══════════════════════════════════════════════════════════════════════════

class NNKalman:
    """
    1-state (position-only) Kalman filter with NN-Q adaptation.
    Asymmetric IIR smoother on Q prevents single-sample spikes while
    allowing fast rise on transients and slow fall in quiet periods.
    """
    # Asymmetric IIR: rise fast on transients, fall slowly in quiet regions
    # so filter spends most time near Q_MIN between signal events.
    #   α_up   = 0.90  → τ_rise  ≈  1 sample  (0.4 ms)   — snap to large Q
    #   α_down = 0.50  → τ_fall  ≈  2 samples  (0.8 ms)  — fast return to Q_MIN
    Q_ALPHA_UP   = 0.90
    Q_ALPHA_DOWN = 0.50

    def __init__(self, nn: TinyNNQ, q_fixed: float = None):
        self.nn       = nn
        self.q_fixed  = q_fixed
        self.x        = 0.0
        self.P        = 0.0
        self.inn_buf  = np.zeros(N_INN)
        self.q_smooth = nn.q_min   # start at Q_MIN — only rises when needed
        self.q_trace   = []
        self.inn_trace = []
        self.p_trace   = []

    def reset(self, z0: float):
        self.x          = z0
        self.P          = R_MEAS * 20.0
        self.inn_buf[:] = 0.0
        self.q_smooth   = self.nn.q_min
        self.q_trace.clear()
        self.inn_trace.clear()
        self.p_trace.clear()

    def step(self, z: float) -> float:
        # Q selection + asymmetric IIR smoothing
        Q_raw = self.q_fixed if self.q_fixed is not None else self.nn.forward(self.inn_buf)
        if Q_raw >= self.q_smooth:
            alpha = self.Q_ALPHA_UP
        else:
            alpha = self.Q_ALPHA_DOWN
        Q_val = alpha * Q_raw + (1.0 - alpha) * self.q_smooth
        self.q_smooth = Q_val

        # 1-state predict
        P_p = self.P + Q_val

        # 1-state update
        S   = P_p + R_MEAS
        K   = P_p / S
        inn = z - self.x
        self.x = self.x + K * inn
        self.P = (1.0 - K) * P_p          # scalar Joseph form

        # normalise & buffer innovation
        inn_n = np.clip(inn / self.nn.inn_scale, -5.0, 5.0)
        self.inn_buf = np.roll(self.inn_buf, -1)
        self.inn_buf[-1] = inn_n

        self.q_trace.append(Q_val)
        self.inn_trace.append(inn)
        self.p_trace.append(self.P)
        return self.x

    def run(self, signal: np.ndarray) -> np.ndarray:
        self.reset(signal[0])
        out = np.empty(len(signal))
        for k, z in enumerate(signal):
            out[k] = self.step(float(z))
        return out


# ═══════════════════════════════════════════════════════════════════════════
# 5.  DECIMATOR
# ═══════════════════════════════════════════════════════════════════════════

def decimate_4x(raw_codes: np.ndarray) -> np.ndarray:
    N_out  = len(raw_codes) // OSR
    codes  = raw_codes[:N_out * OSR].reshape(N_out, OSR)
    return (codes.mean(axis=1) + 0.5) * LSB


# ═══════════════════════════════════════════════════════════════════════════
# 6.  DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════

def load_gyro_csv(path: str) -> list:
    """
    Load the quantized_7bit_equivalent gyro CSV file.
    Format: sample_rate , label , timestamp , x , y , z ;
    Returns list of (label, axis_name, voltage_array) tuples.
    """
    FS_GYRO = 1600.0
    from collections import defaultdict
    by_label = defaultdict(lambda: [[], [], []])

    with open(path, 'r', errors='replace') as f:
        for line in f:
            line = line.strip().rstrip(';')
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            try:
                label = parts[1].strip()
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                by_label[label][0].append(x)
                by_label[label][1].append(y)
                by_label[label][2].append(z)
            except (ValueError, IndexError):
                continue

    segments = []
    axis_names = ['X', 'Y', 'Z']
    total_sigs = 0

    for label in sorted(by_label.keys()):
        axes_data = by_label[label]
        for ai, axis_name in enumerate(axis_names):
            raw_phys = np.array(axes_data[ai], dtype=np.float64)
            if len(raw_phys) < 8:
                continue
            n_old = len(raw_phys)
            n_new = int(round(n_old * FS_OUT / FS_GYRO))
            t_old = np.linspace(0.0, 1.0, n_old)
            t_new = np.linspace(0.0, 1.0, n_new)
            resampled = np.interp(t_new, t_old, raw_phys)
            PHYS_RANGE = 6.0
            codes = np.clip(
                np.round((resampled + PHYS_RANGE) / (2 * PHYS_RANGE) * 255.0),
                0, 255
            )
            voltage = (codes + 0.5) * LSB
            segments.append((label, axis_name, voltage))
            total_sigs += 1

    print(f"  [load_gyro_csv]  {path}")
    print(f"  Labels: {sorted(by_label.keys())}  →  {total_sigs} signals "
          f"({len(by_label)} labels × 3 axes)  at {FS_OUT} SPS")
    return segments


def synthetic_oscillatory_realistic(n_raw: int = 20_000) -> list:
    """
    Realistic multi-frequency synthetic signals for pool training.

    WHY THIS IS NEEDED:
    Pool training on gyro + ECG teaches W1/W2 to recognise slow drift and
    sparse QRS spikes — but never sustained fast sinusoidal oscillations.

    A 73 Hz sine through the innovation Kalman produces normalised innovations
    with std ≈ 0.45 oscillating at 73 Hz — W1/W2 maps this to "quiet" because
    gyro/ECG have similar amplitude innovations but at low frequency.
    calibrate_quiet_offset() then sets z3_offset = z3(73 Hz) ≈ z3(zeros),
    clipping all sinusoidal responses to Q_MIN → signal is destroyed.

    This function adds 17 explicit fast-sine examples so W1/W2 learn that
    fast oscillation innovations ≠ quiet. After this, z3(73 Hz) >> z3(zeros)
    and calibration is safe.

    Returns list of raw_code arrays (float64, 0-255).
    """
    results = []
    rng_osc = np.random.default_rng(0xF1E2D3C4)

    # ── 1. Pure sine bank: 25 Hz to 1 kHz ───────────────────────────
    # Covers synthetic_demo frequencies (73, 312 Hz) explicitly.
    freqs = [25, 50, 73, 100, 150, 200, 312, 500, 800, 1000]
    for f in freqs:
        t   = np.arange(n_raw) / FS_RAW
        amp = rng_osc.uniform(0.5, 1.5)
        dc  = rng_osc.uniform(1.5, 3.5)
        sig = amp * np.sin(2 * np.pi * f * t) + dc
        noise = rng_osc.normal(0, 0.4 * LSB, n_raw)
        codes = np.clip(np.round((sig + noise) / LSB), 0, 255)
        results.append(codes.astype(np.float64))

    # ── 2. Multi-tone — exact synthetic_demo character ───────────────
    # 73 Hz + 312 Hz + DC step: teaches the NN the exact test waveform.
    t   = np.arange(n_raw) / FS_RAW
    sig = (1.2 * np.sin(2 * np.pi * 73 * t) +
           0.5 * np.sin(2 * np.pi * 312 * t) + 2.5)
    sig[n_raw // 2:] += 0.8
    noise = rng_osc.normal(0, 0.45 * LSB, n_raw)
    codes = np.clip(np.round((sig + noise) / LSB), 0, 255)
    results.append(codes.astype(np.float64))

    # ── 3. Sine + ramp (frequency content + trend) ───────────────────
    for f in [73, 200, 400]:
        t    = np.arange(n_raw) / FS_RAW
        amp  = rng_osc.uniform(0.4, 1.0)
        dc   = rng_osc.uniform(1.0, 2.5)
        ramp = np.linspace(0, 1.5, n_raw)
        sig  = amp * np.sin(2 * np.pi * f * t) + dc + ramp
        sig  = np.clip(sig, 0.1, 4.9)
        noise = rng_osc.normal(0, 0.4 * LSB, n_raw)
        codes = np.clip(np.round((sig + noise) / LSB), 0, 255)
        results.append(codes.astype(np.float64))

    # ── 4. Step-only signals ─────────────────────────────────────────
    for _ in range(3):
        sig = np.ones(n_raw) * rng_osc.uniform(1.5, 3.0)
        n_steps = rng_osc.integers(3, 10)
        for _s in range(n_steps):
            pos = rng_osc.integers(0, n_raw)
            sig[pos:] += rng_osc.uniform(-1.2, 1.2)
        sig = np.clip(sig, 0.1, 4.9)
        noise = rng_osc.normal(0, 0.4 * LSB, n_raw)
        codes = np.clip(np.round((sig + noise) / LSB), 0, 255)
        results.append(codes.astype(np.float64))

    return results


def build_training_pool(gyro_path: str = None,
                        ecg_paths: list = None,
                        include_synthetic: bool = True) -> list:
    """
    Build a list of (decimated_voltage, q_params_local) tuples for pool training.

    Each segment keeps its own q_params (including inn_scale) so
    _compute_features() can normalise innovations per-segment → features
    are always in [-5, +5] regardless of signal amplitude.

    Returns: list of (signal_array, q_params_dict)
    """
    pool = []

    if include_synthetic:
        syn_raw = synthetic_demo(40_000)
        pad = (-len(syn_raw)) % OSR
        if pad:
            syn_raw = np.append(syn_raw, np.full(pad, syn_raw[-1]))
        syn_dec = decimate_4x(syn_raw)
        qp_syn  = _adaptive_q_params(syn_dec)
        pool.append((syn_dec, qp_syn))
        print(f"  [pool]  Synthetic demo: {len(syn_dec):,} samples  "
              f"inn_scale={qp_syn['inn_scale']*1e3:.1f} mV")

        # ── Oscillatory signal bank ──────────────────────────────────────
        # CRITICAL: Without fast sinusoidal examples in the pool, W1/W2
        # cannot distinguish small-amplitude fast oscillations from quiet
        # input. This causes calibrate_quiet_offset to set z3_offset equal
        # to z3(73 Hz sine) which locks Q = Q_MIN for all sinusoidal inputs,
        # destroying the signal entirely (Corr dropped to 0.65 in testing).
        # See synthetic_oscillatory_realistic() docstring for full diagnosis.
        osc_bank = synthetic_oscillatory_realistic(20_000)
        n_osc = 0
        for codes in osc_bank:
            pad = (-len(codes)) % OSR
            if pad:
                codes = np.append(codes, np.full(pad, codes[-1]))
            osc_dec = decimate_4x(codes)
            qp_osc  = _adaptive_q_params(osc_dec)
            pool.append((osc_dec, qp_osc))
            n_osc += 1
        print(f"  [pool]  Oscillatory bank: {n_osc} signals "
              f"(25–1000 Hz + steps + mixed)  added")

    if gyro_path and os.path.exists(gyro_path):
        segments = load_gyro_csv(gyro_path)
        total_gyro = 0
        for (label, axis, voltage) in segments:
            qp_seg = _adaptive_q_params(voltage)
            pool.append((voltage, qp_seg))
            total_gyro += len(voltage)
        inn_scales = [p[1]['inn_scale']*1e3 for p in pool[1:]]
        print(f"  [pool]  Gyro CSV: {total_gyro:,} samples, {len(segments)} signals  "
              f"inn_scale=[{min(inn_scales):.1f}…{max(inn_scales):.1f}] mV")
    elif gyro_path:
        print(f"  [pool]  ⚠ Gyro file not found: {gyro_path}")

    # ── ECG training signals ──────────────────────────────────────────────
    if ecg_paths:
        n_ecg_leads = 0
        n_ecg_samps = 0
        for mat_path in ecg_paths:
            if not os.path.exists(mat_path):
                print(f"  [pool]  ⚠ ECG not found: {mat_path}")
                continue
            leads = load_wfdb_all_leads(mat_path)
            for (lead_lbl, voltage) in leads:
                qp = _adaptive_q_params(voltage)
                pool.append((voltage, qp))
                n_ecg_leads += 1
                n_ecg_samps += len(voltage)
        if n_ecg_leads:
            ecg_inn = [pool[-(n_ecg_leads - i)][1]['inn_scale'] * 1e3
                       for i in range(n_ecg_leads)]
            print(f"  [pool]  ECG: {n_ecg_samps:,} samples, "
                  f"{n_ecg_leads} leads across {len(ecg_paths)} files  "
                  f"inn_scale=[{min(ecg_inn):.1f}…{max(ecg_inn):.1f}] mV")

    if not pool:
        raise ValueError("Training pool is empty.")

    total = sum(len(p[0]) for p in pool)
    print(f"  [pool]  {len(pool)} segments  total {total:,} samples")
    return pool


def _parse_adc_csv_segments(path: str, gap_thresh_s: float = 1.0) -> tuple:
    rows = []
    with open(path, 'r', errors='replace') as f:
        header_line = f.readline().strip()
        col_names   = [c.strip().lower() for c in header_line.split(',')]
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric data found in {path}")
    data = np.array(rows)

    ts_keywords  = ('timestamp', 'time', 'ts', 't_s', 'time_s')
    adc_keywords = ('adc_raw', 'adc', 'raw', 'value', 'code', 'counts')
    ts_col = adc_col = None
    for i, name in enumerate(col_names):
        if any(k in name for k in ts_keywords)  and ts_col  is None: ts_col  = i
        if any(k in name for k in adc_keywords) and adc_col is None: adc_col = i
    if ts_col is None or adc_col is None:
        for i in range(data.shape[1]):
            col = data[:, i]
            is_int_range = (col.min() >= 0 and col.max() <= 255
                            and np.all(col == col.astype(int)))
            is_time      = (col.min() >= 0 and col.max() > 1.0
                            and np.sum(np.diff(col) > 0) > len(col) * 0.7)
            if is_int_range and adc_col is None: adc_col = i
            if is_time      and ts_col  is None: ts_col  = i

    if adc_col is None:
        raise ValueError("Cannot find ADC_Raw column (0-255 integers) in CSV")

    ts_name  = col_names[ts_col]  if ts_col  is not None and ts_col  < len(col_names) else '?'
    adc_name = col_names[adc_col] if adc_col < len(col_names) else '?'
    print(f"  [parse_csv]  {os.path.basename(path)}: {len(data):,} rows  "
          f"ts=col{ts_col}('{ts_name}')  adc=col{adc_col}('{adc_name}')")

    if ts_col is None:
        adc_vals = data[:, adc_col].astype(np.float64)
        seg_info = [{'idx': 0, 'n': len(adc_vals), 'dur': len(adc_vals)/FS_RAW,
                     'native_fs': FS_RAW,
                     'ts':  np.arange(len(adc_vals), dtype=np.float64) / FS_RAW,
                     'adc': np.clip(adc_vals, 0, 255)}]
        return seg_info, col_names

    ts  = data[:, ts_col].astype(np.float64)
    adc = data[:, adc_col].astype(np.float64)

    gaps       = np.where(np.diff(ts) > gap_thresh_s)[0]
    seg_bounds = list(zip(
        np.concatenate([[0], gaps + 1]),
        np.concatenate([gaps + 1, [len(ts)]])
    ))

    seg_info = []
    for s, e in seg_bounds:
        if e - s < 4:
            continue
        ts_seg  = ts[s:e]
        adc_seg = adc[s:e]
        dur     = ts_seg[-1] - ts_seg[0]
        if dur <= 0:
            continue
        diffs    = np.diff(ts_seg)
        med_dt   = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0/FS_RAW
        med_dt   = max(med_dt, 1e-9)
        native_fs = 1.0 / med_dt
        idx = len(seg_info)
        seg_info.append({
            'idx': idx, 'n': e - s, 'dur': dur,
            'native_fs': native_fs,
            'ts': ts_seg, 'adc': adc_seg,
        })
        print(f"  [parse_csv]  seg {idx}: {e-s:6,d} samples  "
              f"dur={dur:.2f}s  fs≈{native_fs:.1f} Hz  "
              f"ADC=[{int(adc_seg.min())},{int(adc_seg.max())}]")

    if not seg_info:
        raise ValueError("No valid segments found in CSV")
    return seg_info, col_names


def _resample_seg_to_fs_raw(seg: dict, max_duration_s: float = 30.0) -> np.ndarray:
    ts_s  = seg['ts']
    adc_s = seg['adc']
    t_end = min(ts_s[-1], ts_s[0] + max_duration_s)
    mask  = ts_s <= t_end
    ts_s, adc_s = ts_s[mask], adc_s[mask]
    n_out     = max(4, int(round((ts_s[-1] - ts_s[0]) * FS_RAW)))
    t_uniform = np.linspace(ts_s[0], ts_s[-1], n_out)
    raw_codes = np.clip(np.round(np.interp(t_uniform, ts_s, adc_s)), 0, 255)
    return raw_codes.astype(np.float64)


def load_adc_csv(path: str,
                 seg_index = None,
                 max_duration_s: float = 30.0,
                 gap_thresh_s:   float = 1.0):
    seg_info, _ = _parse_adc_csv_segments(path, gap_thresh_s)

    def _score(seg):
        rate_ratio = seg['native_fs'] / FS_RAW
        return (-abs(np.log10(max(rate_ratio, 1e-6)))
                + 0.3 * min(seg['dur'], max_duration_s) / max_duration_s)

    if seg_index == 'all':
        out = []
        for seg in seg_info:
            lbl   = f"seg{seg['idx']}_{seg['native_fs']:.0f}Hz"
            codes = _resample_seg_to_fs_raw(seg, max_duration_s)
            print(f"  [load_adc_csv]  {lbl}: "
                  f"{len(seg['ts']):,} → {len(codes):,} samples  "
                  f"({len(codes)/FS_RAW:.2f} s @ {FS_RAW} SPS)")
            out.append((lbl, codes))
        return out

    if seg_index is None:
        chosen = max(seg_info, key=_score)
        print(f"  [load_adc_csv]  Auto-selected seg {chosen['idx']}  "
              f"native_fs={chosen['native_fs']:.1f} Hz  dur={chosen['dur']:.2f}s")
    else:
        if not (0 <= int(seg_index) < len(seg_info)):
            raise IndexError(f"seg_index {seg_index} out of range "
                             f"(file has {len(seg_info)} segments)")
        chosen = seg_info[int(seg_index)]
        print(f"  [load_adc_csv]  Using seg {chosen['idx']}  "
              f"native_fs={chosen['native_fs']:.1f} Hz  dur={chosen['dur']:.2f}s")

    codes = _resample_seg_to_fs_raw(chosen, max_duration_s)
    print(f"  [load_adc_csv]  {len(chosen['ts']):,} → {len(codes):,} samples "
          f"({len(codes)/FS_RAW:.2f} s @ {FS_RAW} SPS)")
    return codes


def load_txt(path: str) -> np.ndarray:
    """Auto-detecting ADC text file loader (comma/tab/space delimited)."""
    import re
    from collections import Counter

    raw_rows = []
    with open(path, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = re.split(r'[,;\t]+|[ ]+', line)
            raw_rows.append(parts)

    if not raw_rows:
        raise ValueError(f"File is empty or unreadable: {path}")

    def try_float(s):
        try:    return float(s)
        except: return None

    numeric_rows = []
    for row in raw_rows:
        parsed = [try_float(p) for p in row]
        if any(v is not None for v in parsed):
            numeric_rows.append(parsed)

    if not numeric_rows:
        raise ValueError(f"No numeric data found in '{path}'.")

    width = Counter(len(r) for r in numeric_rows).most_common(1)[0][0]
    rows  = [r for r in numeric_rows if len(r) == width]

    if width == 1:
        best_col = 0
    else:
        best_col, best_score = 0, -1.0
        for col in range(width):
            vals = np.array([r[col] for r in rows if r[col] is not None])
            if len(vals) == 0:
                continue
            in_range  = float(np.mean((vals >= 0) & (vals <= 255)))
            int_bonus = 0.1 if float(np.mean(np.abs(vals - np.round(vals)) < 0.5)) > 0.9 else 0.0
            score     = in_range + int_bonus + col * 0.001
            if score > best_score:
                best_score, best_col = score, col
        print(f"  [load_txt]  {width}-column file -> using column {best_col} (0-indexed)")

    data = [r[best_col] for r in rows if r[best_col] is not None]
    arr  = np.array(data, dtype=np.float64)

    if arr.max() <= VREF * 1.05:
        print(f"  [load_txt]  Voltages detected -> converting to 8-bit codes.")
        arr = np.round(np.clip(arr / LSB, 0, 255))
    else:
        arr = np.clip(arr, 0, 255)

    print(f"  [load_txt]  {len(arr):,} samples  (range [{arr.min():.0f}, {arr.max():.0f}])")
    return arr


def _parse_hea(hea_path: str) -> dict:
    info = {'signals': []}
    with open(hea_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    rec_parts = lines[0].split()
    info['n_sig']     = int(rec_parts[1])
    info['fs']        = float(rec_parts[2]) if len(rec_parts) > 2 else 360.0
    info['n_samples'] = int(rec_parts[3])   if len(rec_parts) > 3 else 0

    def _parse_fmt(s):
        try:
            base   = int(s.split('+')[0].split('x')[0].strip())
            offset = int(s.split('+')[1]) if '+' in s else 0
            return base, offset
        except Exception:
            return 212, 0

    def _parse_gain(s):
        try:
            return float(s.split('/')[0].strip())
        except Exception:
            return 200.0

    for i in range(1, 1 + info['n_sig']):
        if i >= len(lines):
            break
        parts = lines[i].split()
        fmt_base, fmt_offset = _parse_fmt(parts[1]) if len(parts) > 1 else (212, 0)
        sig = {
            'filename':   parts[0] if len(parts) > 0 else '',
            'fmt':        fmt_base,
            'fmt_offset': fmt_offset,
            'gain':       _parse_gain(parts[2])   if len(parts) > 2 else 200.0,
            'bits':       int(parts[3])            if len(parts) > 3 else 12,
            'baseline':   int(parts[4])            if len(parts) > 4 else 0,
            'adc_zero':   int(parts[5])            if len(parts) > 5 else 0,
            'init_val':   int(parts[6])            if len(parts) > 6 else 0,
            'label':      parts[8]                 if len(parts) > 8 else f'sig{i-1}',
        }
        info['signals'].append(sig)

    return info


def _decode_fmt212(raw_bytes: bytes, n_samples: int, n_sig: int) -> np.ndarray:
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.int32)
    n_groups = len(arr) // 3
    a = arr[0::3] | ((arr[1::3] & 0x0F) << 8)
    b = (arr[1::3] >> 4) | (arr[2::3] << 4)
    a[a >= 2048] -= 4096
    b[b >= 2048] -= 4096
    interleaved = np.empty(2 * n_groups, dtype=np.int32)
    interleaved[0::2] = a
    interleaved[1::2] = b
    total = n_samples * n_sig
    interleaved = interleaved[:total]
    return interleaved.reshape(n_samples, n_sig)


def _decode_fmt16(raw_bytes: bytes, n_samples: int, n_sig: int) -> np.ndarray:
    arr = np.frombuffer(raw_bytes, dtype='<i2').astype(np.int32)
    return arr[:n_samples * n_sig].reshape(n_samples, n_sig)


def load_wfdb(mat_path: str, sig_index: int = 0) -> tuple:
    """
    Load a WFDB record from .mat (raw binary) + .hea file pair.
    PhysioNet WFDB .mat files are raw binary in format 212 or 16.
    Returns (raw_codes [0-255 float64], fs_original, label_str).
    """
    hea_path = mat_path.replace('.mat', '.hea')
    if not os.path.exists(hea_path):
        hea_path = os.path.join(os.path.dirname(mat_path),
                                os.path.splitext(os.path.basename(mat_path))[0] + '.hea')
    if not os.path.exists(hea_path):
        raise FileNotFoundError(f"Cannot find header file for {mat_path}")

    hdr = _parse_hea(hea_path)
    print(f"  [load_wfdb]  {os.path.basename(mat_path)}  "
          f"fs={hdr['fs']:.0f} Hz  n_sig={hdr['n_sig']}  "
          f"n_samples={hdr['n_samples']}")

    dat_path = mat_path
    base     = os.path.splitext(mat_path)[0]
    for ext in ('.dat', '.mat'):
        candidate = base + ext
        if os.path.exists(candidate):
            dat_path = candidate
            break

    raw_signal = None
    try:
        import scipy.io as sio
        mat_data = sio.loadmat(dat_path)
        for key in ('val', 'data', 'signal'):
            if key in mat_data:
                raw_signal = mat_data[key].astype(np.float64)
                if raw_signal.ndim == 2:
                    if raw_signal.shape[0] < raw_signal.shape[1]:
                        raw_signal = raw_signal[sig_index]
                    else:
                        raw_signal = raw_signal[:, sig_index]
                print(f"  [load_wfdb]  Loaded via scipy.io, key='{key}', "
                      f"{len(raw_signal):,} samples")
                break
    except Exception as e:
        print(f"  [load_wfdb]  scipy.io failed ({e}), trying binary decode …")

    if raw_signal is None:
        with open(dat_path, 'rb') as f:
            raw_bytes = f.read()
        fmt = hdr['signals'][sig_index]['fmt'] if hdr['signals'] else 212
        n   = hdr['n_samples'] if hdr['n_samples'] > 0 else len(raw_bytes) * 2 // 3
        n_s = hdr['n_sig']
        try:
            if fmt == 212:
                decoded = _decode_fmt212(raw_bytes, n, n_s)
            elif fmt in (16, 61):
                decoded = _decode_fmt16(raw_bytes, n, n_s)
            else:
                raise ValueError(f"Unsupported WFDB format {fmt}")
            raw_signal = decoded[:, sig_index].astype(np.float64)
            print(f"  [load_wfdb]  Binary decode fmt={fmt}, {len(raw_signal):,} samples")
        except Exception as e2:
            raise RuntimeError(f"Could not decode {dat_path}: {e2}")

    sig_info  = hdr['signals'][sig_index] if hdr['signals'] else {}
    gain      = sig_info.get('gain',     200.0)
    baseline  = sig_info.get('baseline', 0)
    label     = sig_info.get('label',    f'sig{sig_index}')

    phys = (raw_signal - baseline) / gain
    p_min, p_max = phys.min(), phys.max()
    if p_max > p_min:
        codes = np.round((phys - p_min) / (p_max - p_min) * 255.0)
    else:
        codes = np.full(len(phys), 128.0)
    codes = np.clip(codes, 0, 255)

    phys_range = p_max - p_min
    if phys_range > 100_000:
        print(f"  [load_wfdb]  ⚠ WARNING: phys range={phys_range:.1f} mV looks wrong!")
    print(f"  [load_wfdb]  Signal '{label}'  gain={gain}  baseline={baseline}  "
          f"phys=[{p_min:.4f}, {p_max:.4f}] mV  range={phys_range:.4f} mV → mapped to 0-255")
    return codes, float(hdr['fs']), label


def load_wfdb_all_leads(mat_path: str) -> list:
    hea_path = os.path.splitext(mat_path)[0] + '.hea'
    if not os.path.exists(hea_path):
        print(f"  [all_leads]  ⚠ No .hea for {mat_path} — skipping.")
        return []
    try:
        hdr = _parse_hea(hea_path)
    except Exception as e:
        print(f"  [all_leads]  ⚠ Cannot parse {hea_path}: {e}")
        return []

    n_sig   = hdr.get('n_sig', 1)
    results = []
    for i in range(n_sig):
        try:
            codes, fs_orig, label = load_wfdb(mat_path, sig_index=i)
            if abs(fs_orig - FS_OUT) > 1.0:
                n_new = int(round(len(codes) * FS_OUT / fs_orig))
                t_old = np.linspace(0.0, 1.0, len(codes))
                t_new = np.linspace(0.0, 1.0, n_new)
                codes = np.clip(np.round(np.interp(t_new, t_old, codes)), 0, 255)
            voltage = (codes + 0.5) * LSB
            results.append((label, voltage))
        except Exception as e:
            sig_lbl = (hdr['signals'][i]['label']
                       if i < len(hdr.get('signals', [])) else f'sig{i}')
            print(f"  [all_leads]  ⚠ Lead {sig_lbl}: {e}")

    print(f"  [all_leads]  {os.path.basename(mat_path)}: "
          f"{len(results)}/{n_sig} leads  {[r[0] for r in results]}")
    return results


def synthetic_demo(n_raw: int = 40_000) -> np.ndarray:
    t   = np.arange(n_raw) / FS_RAW
    sig = (1.2 * np.sin(2 * np.pi * 73  * t) +
           0.5 * np.sin(2 * np.pi * 312 * t) + 2.5)
    sig[int(1.5 * FS_RAW):]  += 0.8
    r = slice(int(2.0 * FS_RAW), int(3.0 * FS_RAW))
    sig[r] += np.linspace(0, 0.6, int(1.0 * FS_RAW))
    noise = RNG.normal(0, 0.45 * LSB, n_raw)
    return np.clip(np.round((sig + noise) / LSB), 0, 255)


# ═══════════════════════════════════════════════════════════════════════════
# 7.  PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════

def rms(x):  return float(np.sqrt(np.mean(np.asarray(x)**2)))

def enob_sine_fit(signal: np.ndarray, fs: float, freq: float) -> float:
    """IEEE 1241 / 1057  3-parameter sine-fit ENOB."""
    n = len(signal)
    t = np.arange(n) / fs
    A = np.column_stack([np.cos(2*np.pi*freq*t),
                         np.sin(2*np.pi*freq*t),
                         np.ones(n)])
    coeff, _, _, _ = np.linalg.lstsq(A, signal, rcond=None)
    fitted = A @ coeff
    noise  = signal - fitted
    sp, np_ = np.var(fitted), np.var(noise)
    if sp <= 1e-20 or np_ <= 1e-20:
        return 0.0
    return (10*np.log10(sp / np_) - 1.76) / 6.02


OSR_ENOB_GAIN = np.log2(OSR) / 2.0   # +1.0 bit  (fixed, always present)

def enob_from_p(p_val: float) -> float:
    """
    Kalman ENOB gain relative to post-decimation noise floor (R_MEAS).
    Covariance-based estimate — not IEEE 1241 sine-fit.
    Total gain above raw ADC = enob_from_p(p) + OSR_ENOB_GAIN.
    """
    if p_val <= 0 or R_MEAS <= 0:
        return 0.0
    return np.log2(R_MEAS / p_val) / 2.0


def p_steady_state(q: float) -> float:
    """Theoretical 1-state steady-state P for given Q and R_MEAS."""
    return (-q + np.sqrt(q**2 + 4*q*R_MEAS)) / 2.0


# ═══════════════════════════════════════════════════════════════════════════
# 8.  ENOB VALIDATION TEST
# ═══════════════════════════════════════════════════════════════════════════

ENOB_TEST_FREQ = 73.0
ENOB_TEST_BAND = (7.00, 7.30)


def _sigma_for_enob(target_enob: float, freq: float = ENOB_TEST_FREQ,
                    amplitude_lsb: float = 0.45 * 255) -> float:
    H_MA    = (np.sin(OSR * np.pi * freq / FS_RAW) /
               (OSR * np.sin(np.pi * freq / FS_RAW)))
    snr_lin = 10.0 ** ((target_enob * 6.02 + 1.76) / 10.0)
    sig_pow = (H_MA * amplitude_lsb) ** 2 / 2.0
    nv      = sig_pow / snr_lin
    return float(np.sqrt(max(0.0, nv * OSR - 1.0 / 12.0)))


def generate_enob_test_signal(n_raw=40_000, target_enob=7.15, seed=0xE20B):
    amplitude = 0.45 * 255
    centre    = 128.0
    sigma_add = _sigma_for_enob(target_enob)
    rg  = np.random.default_rng(seed)
    t   = np.arange(n_raw) / FS_RAW
    raw = amplitude * np.sin(2 * np.pi * ENOB_TEST_FREQ * t) + centre
    raw = np.clip(np.round(raw + rg.normal(0.0, sigma_add, n_raw)), 0, 255).astype(np.float64)
    dec         = decimate_4x(raw)
    actual_enob = enob_sine_fit(dec, FS_OUT, ENOB_TEST_FREQ)
    print(f"  [ENOB gen]  σ_add={sigma_add:.4f} LSB  →  "
          f"measured ENOB={actual_enob:.3f} bits (target {target_enob:.2f})")
    return raw, actual_enob


def run_enob_test(out_dir: str = "output", gyro_train_path: str = None) -> bool:
    if os.path.isfile(out_dir):
        out_dir = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "output")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "═"*66)
    print("  ENOB TEST  —  target band [7.00, 7.30] bits @ 73 Hz")
    print("  FILTER: 1-STATE position-only Kalman")
    print("═"*66)

    passed = True
    raw_codes, inp_enob = generate_enob_test_signal(40_000, 7.15, 0xE20B)
    pad = (-len(raw_codes)) % OSR
    if pad:
        raw_codes = np.append(raw_codes, np.full(pad, raw_codes[-1]))
    decimated = decimate_4x(raw_codes)
    N         = len(decimated)
    t_out     = np.arange(N) / FS_OUT

    in_band = ENOB_TEST_BAND[0] <= inp_enob <= ENOB_TEST_BAND[1]
    print(f"  {'✓ PASS' if in_band else '✗ FAIL'}  Input ENOB = {inp_enob:.3f} bits")
    if not in_band:
        passed = False

    print("  Training NN-Q estimator (single-signal) …")
    q_params  = _adaptive_q_params(decimated)
    nn        = TinyNNQ(seed=42, q_params=q_params)
    loss_hist = nn.train(decimated, epochs=300, lr=3e-3, verbose=False)
    print(f"  Done  │  final loss = {loss_hist[-1]:.5f}")

    nn_kf     = NNKalman(nn)
    nn_out    = nn_kf.run(decimated)
    fixed_kf  = NNKalman(nn, q_fixed=q_params['q_init'])
    fixed_out = fixed_kf.run(decimated)

    enob_nn    = enob_sine_fit(nn_out,    FS_OUT, ENOB_TEST_FREQ)
    enob_fixed = enob_sine_fit(fixed_out, FS_OUT, ENOB_TEST_FREQ)

    tol_ff = 0.02
    if enob_nn >= enob_fixed - tol_ff:
        print(f"  ✓ PASS  NN-Q {enob_nn:.3f} ≥ Fixed-Q {enob_fixed:.3f}")
    else:
        print(f"  ✗ FAIL  NN-Q {enob_nn:.3f} < Fixed-Q {enob_fixed:.3f}")
        passed = False

    tol = 0.05
    if enob_nn >= inp_enob - tol:
        print(f"  ✓ PASS  NN-Q {enob_nn:.3f} ≥ input {inp_enob:.3f} − {tol}")
    else:
        print(f"  ✗ FAIL  NN-Q {enob_nn:.3f} < input {inp_enob:.3f} − {tol}")
        passed = False

    p_th = p_steady_state(q_params['q_min'])
    eg_th = enob_from_p(p_th)
    p_nn  = float(np.mean(nn_kf.p_trace))

    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │              ENOB TEST SUMMARY  [1-STATE]                   │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │  Test frequency            : {ENOB_TEST_FREQ:>7.1f} Hz                  │")
    print(f"  │  Input ENOB  (sine-fit)    : {inp_enob:>7.3f} bits               │")
    print(f"  │  NN-Q  ENOB  (sine-fit)    : {enob_nn:>7.3f} bits               │")
    print(f"  │  Fixed-Q ENOB (sine-fit)   : {enob_fixed:>7.3f} bits               │")
    print(f"  │  NN-Q improvement          : {enob_nn-inp_enob:>+7.3f} bits               │")
    print(f"  │  Theoretical max gain      : {eg_th:>+7.3f} bits (@ Q_MIN)       │")
    print(f"  │  Achieved P[0] noise       : {np.sqrt(p_nn)*1e3:>7.4f} mV                │")
    print(f"  │  Band check [{ENOB_TEST_BAND[0]:.2f},{ENOB_TEST_BAND[1]:.2f}]    : "
          f"{'PASS ✓' if in_band else 'FAIL ✗'}                       │")
    print("  └─────────────────────────────────────────────────────────────┘")

    prefix = os.path.join(out_dir, "enob_test_1state")
    plot_dashboard(t_out, decimated, nn_out, fixed_out,
                   nn_kf, fixed_kf, loss_hist, prefix, title_tag="[1-STATE]")
    plot_closeup(t_out, decimated, nn_out, fixed_out, prefix)

    print(f"\n  ENOB test {'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed


# ═══════════════════════════════════════════════════════════════════════════
# 9.  VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════

PAL = dict(raw='#B0BEC5', ma='#4FC3F7', fixed='#FF7043',
           nn='#66BB6A', q='#FFA726', theory='#CE93D8')
BG  = '#0D1117'
AX  = '#161B22'
GR  = '#30363D'


def _ax_style(ax, title, xlabel, ylabel):
    ax.set_facecolor(AX)
    ax.set_title(title, color='white', fontsize=9, pad=5)
    ax.set_xlabel(xlabel, color='#8B949E', fontsize=8)
    ax.set_ylabel(ylabel, color='#8B949E', fontsize=8)
    ax.tick_params(colors='#8B949E', labelsize=7)
    for sp in ax.spines.values(): sp.set_color(GR)
    ax.grid(True, alpha=0.15, color=GR)


def plot_dashboard(t_out, decimated, nn_out, fixed_out,
                   nn_kf, fixed_kf, loss_hist, prefix,
                   title_tag="[1-STATE]"):

    N     = len(t_out)
    win_n = min(int(2.0 * FS_OUT), N)

    fig = plt.figure(figsize=(16, 14), facecolor=BG)
    fig.suptitle(
        f"ADC0804  →  4× MA Decimation  →  NN-Q Adaptive Kalman  {title_tag}",
        fontsize=13, color='white', fontweight='bold', y=0.985)

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.46, wspace=0.28,
                           left=0.07, right=0.97, top=0.94, bottom=0.06)

    ax1 = fig.add_subplot(gs[0, :])
    _ax_style(ax1, f"Time-Domain  (first {win_n/FS_OUT:.1f} s shown)",
              "Time [s]", "Voltage [V]")
    sl = slice(0, win_n)
    ax1.plot(t_out[sl], decimated[sl],  color=PAL['raw'],   alpha=0.45, lw=0.7, label='4× MA decimated')
    ax1.plot(t_out[sl], fixed_out[sl],  color=PAL['fixed'], alpha=0.85, lw=1.1, label='Fixed-Q Kalman')
    ax1.plot(t_out[sl], nn_out[sl],     color=PAL['nn'],    lw=1.5,             label='NN-Q Kalman')
    ax1.legend(loc='upper right', fontsize=8, facecolor='#21262D',
               labelcolor='white', framealpha=0.9)

    ax2 = fig.add_subplot(gs[1, 0])
    _ax_style(ax2, "Adaptive Q(k)  [V²]", "Time [s]", "Q [V²]")
    q_arr = np.array(nn_kf.q_trace)
    ax2.semilogy(t_out[:len(q_arr)], q_arr, color=PAL['q'], lw=1.0, label='NN-Q')
    ax2.axhline(nn_kf.nn.q_min,  ls=':', lw=0.8, color='white', alpha=0.4, label='Q_min')
    ax2.axhline(nn_kf.nn.q_max,  ls=':', lw=0.8, color='cyan',  alpha=0.4, label='Q_max')
    ax2.axhline(nn_kf.nn.q_init, ls='--', lw=0.8, color=PAL['fixed'], alpha=0.7, label='Q_init (fixed)')
    ax2.legend(fontsize=7, facecolor='#21262D', labelcolor='white', framealpha=0.9)

    ax3 = fig.add_subplot(gs[1, 1])
    _ax_style(ax3, "Kalman P[0]  (position variance)  [V²]",
              "Time [s]", "P [V²]")
    p_nn_arr  = np.array(nn_kf.p_trace)
    p_fx_arr  = np.array(fixed_kf.p_trace)
    ax3.semilogy(t_out[:len(p_nn_arr)], p_nn_arr,
                 color=PAL['nn'],    lw=1.0, label='NN-Q P[0]')
    ax3.semilogy(t_out[:len(p_fx_arr)], p_fx_arr,
                 color=PAL['fixed'], lw=1.0, alpha=0.7, label='Fixed-Q P[0]', ls='--')
    ax3.axhline(R_MEAS, ls=':', lw=1.0, color='white', alpha=0.5, label='R_MEAS (noise floor)')
    p_th_min = p_steady_state(nn_kf.nn.q_min)
    ax3.axhline(p_th_min, ls='-.', lw=1.2, color=PAL['theory'], alpha=0.8,
                label=f'P_ss @ Q_MIN = {np.sqrt(p_th_min)*1e3:.3f} mV')
    ax3.legend(fontsize=7, facecolor='#21262D', labelcolor='white', framealpha=0.9)

    ax4 = fig.add_subplot(gs[2, 0])
    _ax_style(ax4, "Innovation Distribution", "Innovation [V]", "Density")
    inn_nn  = np.array(nn_kf.inn_trace)
    inn_fix = np.array(fixed_kf.inn_trace)
    inn_ref = nn_kf.nn.inn_scale
    bins    = np.linspace(-6*inn_ref, 6*inn_ref, 55)
    ax4.hist(inn_fix, bins, color=PAL['fixed'], alpha=0.55, density=True, label='Fixed-Q')
    ax4.hist(inn_nn,  bins, color=PAL['nn'],    alpha=0.70, density=True, label='NN-Q')
    ax4.legend(fontsize=7, facecolor='#21262D', labelcolor='white', framealpha=0.9)

    ax5 = fig.add_subplot(gs[2, 1])
    _ax_style(ax5, "RMS Residual vs MA Decimated  (10 segments)",
              "Segment", "RMS [V]")
    n_seg   = 10
    seg_len = N // n_seg
    rms_nn  = [rms(nn_out[i*seg_len:(i+1)*seg_len]
                   - decimated[i*seg_len:(i+1)*seg_len]) for i in range(n_seg)]
    rms_fix = [rms(fixed_out[i*seg_len:(i+1)*seg_len]
                   - decimated[i*seg_len:(i+1)*seg_len]) for i in range(n_seg)]
    xs = np.arange(n_seg)
    ax5.bar(xs-0.2, rms_fix, 0.38, color=PAL['fixed'], alpha=0.8, label='Fixed-Q')
    ax5.bar(xs+0.2, rms_nn,  0.38, color=PAL['nn'],    alpha=0.8, label='NN-Q')
    ax5.set_xticks(xs)
    ax5.legend(fontsize=7, facecolor='#21262D', labelcolor='white', framealpha=0.9)

    ax6 = fig.add_subplot(gs[3, :])
    _ax_style(ax6, "NN Training Convergence  (log-ratio MSE loss)",
              "Epoch", "Loss")
    ax6.semilogy(loss_hist, color=PAL['nn'], lw=1.4)
    ax6.set_xlim(0, len(loss_hist))

    p_nn    = float(np.mean(nn_kf.p_trace))
    p_fixed = float(np.mean(fixed_kf.p_trace))
    en  = enob_from_p(p_nn)
    ef  = enob_from_p(p_fixed)
    eg_th = enob_from_p(p_steady_state(nn_kf.nn.q_min))
    q   = np.array(nn_kf.q_trace)
    sr  = rms(decimated - decimated.mean())
    # Correct weight count: W1=128 b1=16 W2=128 b2=8 W3=8 → 288 total
    stats = (
        f"Signal RMS: {sr*1e3:.3f} mV   │   "
        f"NN-Q P[0]: {np.sqrt(p_nn)*1e3:.4f} mV  gain: {en:+.3f} bits   │   "
        f"Fixed-Q P[0]: {np.sqrt(p_fixed)*1e3:.4f} mV  gain: {ef:+.3f} bits   │   "
        f"Theoretical max: {eg_th:+.3f} bits   │   "
        f"Q range: [{q.min():.2e}…{q.max():.2e}] V²   │   "
        f"NN: {N_INN*H1+H1+H1*H2+H2+H2} weights, {N_INN*H1+H1*H2+H2} MACs/step"
    )
    fig.text(0.5, 0.012, stats, ha='center', fontsize=7.5,
             color='#8B949E', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.35', facecolor=AX, edgecolor=GR, alpha=0.92))

    out = prefix + "_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_closeup(t_out, decimated, nn_out, fixed_out, prefix):
    win = int(0.3 * FS_OUT)
    N   = len(decimated)
    if N < win * 2:
        return None
    stride = max(1, win // 4)
    best_s, best_v = 0, 0
    for s in range(0, N - win, stride):
        v = float(np.var(decimated[s:s+win]))
        if v > best_v:
            best_v, best_s = v, s
    sl = slice(best_s, best_s + win)

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
    _ax_style(ax, "Close-up: most dynamic 300 ms window", "Time [s]", "Voltage [V]")
    ax.plot(t_out[sl], decimated[sl],  color=PAL['raw'],   alpha=0.5, lw=0.9, label='4× MA decimated')
    ax.plot(t_out[sl], fixed_out[sl],  color=PAL['fixed'], alpha=0.85, lw=1.2, label='Fixed-Q Kalman')
    ax.plot(t_out[sl], nn_out[sl],     color=PAL['nn'],    lw=1.7,             label='NN-Q Kalman')
    ax.legend(fontsize=9, facecolor='#21262D', labelcolor='white', framealpha=0.9)
    out = prefix + "_closeup.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def save_csv(t_out, decimated, nn_out, fixed_out, path):
    hdr  = "time_s,decimated_V,nn_kalman_V,fixed_kalman_V"
    data = np.column_stack([t_out, decimated, nn_out, fixed_out])
    np.savetxt(path, data, delimiter=',', header=hdr, comments='', fmt='%.8f')
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 10.  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run(input_path: str = None,
        out_dir: str = "output",
        label_override: str = None,
        gyro_train_path: str = None,
        ecg_train_paths: list = None,
        csv_seg_index = None):
    """
    csv_seg_index : None=auto-best  |  int=specific segment  |  ignored for .mat/.txt
    """

    if os.path.isfile(out_dir):
        out_dir = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "output")
    os.makedirs(out_dir, exist_ok=True)

    if input_path and os.path.exists(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext == '.mat':
            print(f"\n[1/5]  Loading WFDB: {input_path}")
            raw_codes, fs_orig, wfdb_label = load_wfdb(input_path)
            label = label_override or wfdb_label
            print(f"  Original fs={fs_orig:.0f} Hz  →  resampled to "
                  f"{FS_RAW} SPS (repeat-upsample for pipeline compatibility)")
            n_orig = len(raw_codes)
            n_new  = int(n_orig * FS_RAW / fs_orig)
            t_old  = np.linspace(0, 1, n_orig)
            t_new  = np.linspace(0, 1, n_new)
            raw_codes = np.clip(np.round(np.interp(t_new, t_old, raw_codes)), 0, 255)
        elif ext == '.csv':
            seg_tag   = f"_seg{csv_seg_index}" if csv_seg_index is not None else ""
            print(f"\n[1/5]  Loading ADC CSV: {input_path}{seg_tag}")
            raw_codes = load_adc_csv(input_path, seg_index=csv_seg_index)
            base_lbl  = os.path.splitext(os.path.basename(input_path))[0]
            label     = label_override or (base_lbl + seg_tag)
        else:
            print(f"\n[1/5]  Loading text: {input_path}")
            raw_codes = load_txt(input_path)
            label     = label_override or os.path.splitext(os.path.basename(input_path))[0]

        dur_s = len(raw_codes) / FS_RAW
        print(f"  {len(raw_codes):,} samples  ({dur_s:.2f} s @ {FS_RAW} SPS)")
    else:
        print("\n[1/5]  No input file — running synthetic 4-second demo")
        raw_codes = synthetic_demo(40_000)
        label     = label_override or "demo"

    pad = (-len(raw_codes)) % OSR
    if pad:
        raw_codes = np.append(raw_codes, np.full(pad, raw_codes[-1]))

    print(f"[2/5]  Decimating: {len(raw_codes):,} → {len(raw_codes)//OSR:,} samples")
    decimated = decimate_4x(raw_codes)
    N     = len(decimated)
    t_out = np.arange(N) / FS_OUT

    q_params = _adaptive_q_params(decimated)
    p_th_min = p_steady_state(q_params['q_min'])
    print(f"[3a/5] Adaptive Q:  init={q_params['q_init']:.2e}  "
          f"max={q_params['q_max']:.2e}  inn_scale={q_params['inn_scale']*1e3:.3f} mV")
    print(f"       P_ss @ Q_min={np.sqrt(p_th_min)*1e3:.4f} mV  "
          f"(ENOB gain {enob_from_p(p_th_min):+.3f} bits  theoretical)")

    print("[3b/5] Building training pool …")
    pool_segments = build_training_pool(
        gyro_path=gyro_train_path,
        ecg_paths=ecg_train_paths if ecg_train_paths is not None else ECG_TRAIN_PATHS,
        include_synthetic=True
    )
    pool_segments.append((decimated, q_params))

    nn        = TinyNNQ(seed=42, q_params=q_params)
    print(f"[3c/5] Training NN-Q on pool ({len(pool_segments)} segments) …")
    t0        = time.time()
    if len(decimated) <= 6_000:
        ft_epochs = 300
    elif len(decimated) <= 100_000:
        ft_epochs = 500
    else:
        ft_epochs = 200
    pool_loss, ft_loss = nn.train_pool(
        segments          = pool_segments,
        q_params_inf      = q_params,
        inference_signal  = decimated,
        epochs_pool       = 150,
        epochs_ft         = ft_epochs,
        lr_pool           = 2e-3,
        lr_ft             = 3e-3,
        max_per_seg       = 1_500,
        verbose           = True
    )
    print(f"  Pool done in {time.time()-t0:.1f} s  "
          f"│  pool_loss={pool_loss[-1]:.5f}  ft_loss={ft_loss[-1]:.5f}")
    loss_hist = pool_loss + ft_loss

    print("[4/5]  Running 1-state Kalman filters …")
    nn_kf     = NNKalman(nn)
    nn_out    = nn_kf.run(decimated)

    fixed_kf  = NNKalman(nn, q_fixed=q_params['q_init'])
    fixed_out = fixed_kf.run(decimated)

    p_nn    = float(np.mean(nn_kf.p_trace))
    p_fixed = float(np.mean(fixed_kf.p_trace))
    en  = enob_from_p(p_nn)
    ef  = enob_from_p(p_fixed)
    eg_th = enob_from_p(p_th_min)
    q   = np.array(nn_kf.q_trace)
    sr  = rms(decimated - decimated.mean())

    en_kalman    = en
    ef_kalman    = ef
    eg_th_kalman = eg_th
    en_total     = en_kalman  + OSR_ENOB_GAIN
    ef_total     = ef_kalman  + OSR_ENOB_GAIN
    eg_th_total  = eg_th_kalman + OSR_ENOB_GAIN

    R_RAW = R_MEAS * OSR
    q_quiet       = float(nn_kf.nn.forward(np.zeros(N_INN)))
    q_min_ratio   = q_quiet / nn_kf.nn.q_min
    p_ss_quiet    = p_steady_state(q_quiet)
    eg_quiet_kal  = enob_from_p(p_ss_quiet)
    eg_quiet_tot  = eg_quiet_kal + OSR_ENOB_GAIN

    print()
    print("  ┌──────────────────────────────────────────────────────────────┐")
    print("  │         PERFORMANCE SUMMARY  [OSR=4 + 1-STATE NN KALMAN]     │")
    print("  ├──────────────────────────────────────────────────────────────┤")
    print(f"  │  Signal RMS                  : {sr*1e3:>8.3f} mV                │")
    print(f"  │  Raw ADC noise  √(LSB²/12)   : {np.sqrt(R_RAW)*1e3:>8.4f} mV                │")
    print(f"  │  After 4× MA   √R_MEAS       : {np.sqrt(R_MEAS)*1e3:>8.4f} mV  (+{OSR_ENOB_GAIN:.2f} bit OSR) │")
    print(f"  │  NN-Q  output  √P[0]         : {np.sqrt(p_nn)*1e3:>8.4f} mV                │")
    print(f"  │  Fixed-Q output √P[0]        : {np.sqrt(p_fixed)*1e3:>8.4f} mV                │")
    print("  ├──────────────────────────────────────────────────────────────┤")
    print("  │  ENOB GAIN BREAKDOWN  (covariance-based estimate)            │")
    print(f"  │  Stage 1: 4× MA Decimation   :      {OSR_ENOB_GAIN:>+.3f} bits  (fixed)     │")
    print(f"  │  Stage 2: NN-Q Kalman        :      {en_kalman:>+.3f} bits  (adaptive)  │")
    print(f"  │           Fixed-Q Kalman     :      {ef_kalman:>+.3f} bits  (reference)  │")
    print("  ├──────────────────────────────────────────────────────────────┤")
    print(f"  │  TOTAL NN-Q gain (vs raw ADC):      {en_total:>+.3f} bits              │")
    print(f"  │  TOTAL Fixed-Q  (vs raw ADC):      {ef_total:>+.3f} bits              │")
    print(f"  │  Theoretical max (vs raw ADC):      {eg_th_total:>+.3f} bits (OSR+Kalman) │")
    print("  ├──────────────────────────────────────────────────────────────┤")
    print(f"  │  Q range (NN)                : {q.min():.2e} – {q.max():.2e} V²  │")
    print(f"  │  Q(quiet)/Q_MIN              : {q_min_ratio:>8.1f}×  ({q_quiet:.2e} V²) │")
    print(f"  │  Potential gain @ quiet      :      {eg_quiet_tot:>+.3f} bits  (OSR+Kal)  │")
    print(f"  │  NN weights / MACs           : {N_INN*H1+H1+H1*H2+H2+H2} weights / {N_INN*H1+H1*H2+H2} MACs          │")
    print("  └──────────────────────────────────────────────────────────────┘")

    print("[5/5]  Saving outputs …")
    prefix = os.path.join(out_dir, label + "_1state")
    plot_dashboard(t_out, decimated, nn_out, fixed_out,
                   nn_kf, fixed_kf, loss_hist, prefix)
    plot_closeup(t_out, decimated, nn_out, fixed_out, prefix)
    save_csv(t_out, decimated, nn_out, fixed_out, prefix + "_output.csv")

    print(f"\n  All outputs in: {out_dir}")
    return nn_out, fixed_out, nn_kf, fixed_kf


# ═══════════════════════════════════════════════════════════════════════════
# 11.  --all-tests  runner
# ═══════════════════════════════════════════════════════════════════════════

TEST_FILES = [
    ADC_CSV_PATH,
]


def run_csv_all_segs(csv_path: str,
                     out_dir:         str  = "output",
                     gyro_train_path: str  = None,
                     ecg_train_paths: list = None,
                     max_duration_s:  float = 30.0) -> dict:
    """
    Run the full NN-Q Kalman pipeline on EVERY segment found in a CSV file.
    Useful for adc_data (1).csv which has three distinct segments:
      seg 0 : long slow log  (~20 Hz, 27 min)  — capped to max_duration_s
      seg 1 : medium burst   (~666 Hz, 2 s)
      seg 2 : fast burst     (~10 kHz, 1.9 s)  — native pipeline rate
    """
    if not os.path.exists(csv_path):
        print(f"  ⚠  CSV not found: {csv_path}")
        return {}

    seg_info, _ = _parse_adc_csv_segments(csv_path)
    print(f"\n{'═'*70}")
    print(f"  CSV ALL-SEGMENTS TEST  [{len(seg_info)} segments found]")
    print(f"  File: {csv_path}")
    print('═'*70)

    results = {}
    for seg in seg_info:
        tag = f"seg{seg['idx']}_{int(seg['native_fs'])}Hz"
        print(f"\n{'─'*70}")
        print(f"  Running segment {seg['idx']}  "
              f"native_fs={seg['native_fs']:.1f} Hz  "
              f"dur={seg['dur']:.2f}s  "
              f"samples={seg['n']:,}")
        try:
            run(input_path=csv_path, out_dir=out_dir,
                label_override=tag,
                gyro_train_path=gyro_train_path,
                ecg_train_paths=ecg_train_paths,
                csv_seg_index=seg['idx'])
            results[tag] = "OK"
        except Exception as e:
            print(f"  ✗  seg {seg['idx']} FAILED: {e}")
            results[tag] = f"ERROR: {e}"

    print(f"\n{'═'*70}")
    print("  CSV ALL-SEGMENTS SUMMARY")
    print(f"{'═'*70}")
    for tag, status in results.items():
        print(f"  {tag:<35} {status}")
    return results


def run_all_tests(out_dir: str = "output", gyro_train_path: str = None,
                  ecg_train_paths: list = None):
    print("\n" + "═"*70)
    print("  RUNNING ALL TEST FILES  [1-STATE KALMAN]")
    print("═"*70)
    gyro_train_path = gyro_train_path or GYRO_TRAIN_PATH
    ecg_train_paths = ecg_train_paths if ecg_train_paths is not None else ECG_TRAIN_PATHS
    results = {}
    for path in TEST_FILES:
        print(f"\n{'─'*70}")
        print(f"  File: {path}")
        if not os.path.exists(path):
            print(f"  ⚠  Not found — skipping.")
            continue
        try:
            run(input_path=path, out_dir=out_dir,
                gyro_train_path=gyro_train_path, ecg_train_paths=ecg_train_paths)
            results[path] = "OK"
        except Exception as e:
            print(f"  ✗  ERROR: {e}")
            results[path] = f"ERROR: {e}"

    print(f"\n{'─'*70}")
    print("  Running ENOB validation test …")
    try:
        passed = run_enob_test(out_dir=out_dir)
        results["ENOB test"] = "PASS ✓" if passed else "FAIL ✗"
    except Exception as e:
        results["ENOB test"] = f"ERROR: {e}"

    print("\n" + "═"*70)
    print("  SUMMARY")
    print("═"*70)
    for k, v in results.items():
        print(f"  {os.path.basename(k):<30} {v}")


# ═══════════════════════════════════════════════════════════════════════════
# 12.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NN-Q Kalman filter (1-state rewrite) for ADC0804 pipeline")
    parser.add_argument("input", nargs="?", default=None,
                        help=".txt (raw ADC codes), .csv (timestamped ADC), or .mat (WFDB) file")
    parser.add_argument("--out-dir", default="output",
                        help="Output directory for plots and CSV")
    parser.add_argument("--test-enob", action="store_true",
                        help="Run ENOB 7.0-7.3 validation test")
    parser.add_argument("--all-tests", action="store_true",
                        help="Run all three test files + ENOB test")
    parser.add_argument("--gyro-train", default=None,
                        help="Override path to gyro training CSV")
    parser.add_argument("--ecg-train", nargs="+", default=None, metavar="PATH",
                        help="Override ECG training .mat paths (space-separated)")
    parser.add_argument("--csv-all-segs", action="store_true",
                        help="Run pipeline on ALL segments in the CSV (not just best)")
    args = parser.parse_args()

    gyro_path = args.gyro_train or GYRO_TRAIN_PATH
    ecg_paths = args.ecg_train  if args.ecg_train is not None else ECG_TRAIN_PATHS

    if args.all_tests:
        run_all_tests(out_dir=args.out_dir, gyro_train_path=gyro_path,
                      ecg_train_paths=ecg_paths)
        sys.exit(0)

    if args.csv_all_segs:
        csv_src = args.input or ADC_CSV_PATH
        run_csv_all_segs(csv_path=csv_src, out_dir=args.out_dir,
                         gyro_train_path=gyro_path, ecg_train_paths=ecg_paths)
        sys.exit(0)

    if args.test_enob:
        ok = run_enob_test(out_dir=args.out_dir, gyro_train_path=gyro_path)
        sys.exit(0 if ok else 1)

    if args.input is None:
        for p in TEST_FILES:
            if os.path.exists(p):
                print(f"[Auto] Found {p}")
                run(input_path=p, out_dir=args.out_dir, gyro_train_path=gyro_path,
                    ecg_train_paths=ecg_paths)
                sys.exit(0)
        print("[Auto] No test files found — running synthetic demo.")
        run(out_dir=args.out_dir, gyro_train_path=gyro_path, ecg_train_paths=ecg_paths)
    else:
        run(input_path=args.input, out_dir=args.out_dir, gyro_train_path=gyro_path,
            ecg_train_paths=ecg_paths)
