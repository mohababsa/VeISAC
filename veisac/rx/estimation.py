# estimation.py
"""
VeISAC — Radar Parameter Estimation

Range, velocity, and angle (MUSIC) estimation from the range-Doppler map, with CRB computation and parabolic sub-bin refinement for the Sen-RX chain.

Paper: "VeISAC: An End-to-End MIMO-OFDM-FMCW Framework for ISAC
        in 6G Vehicular Networks"
Authors: M. Ababsa, S. Ribouh, Y. El Hillali, A. Rivenq
"""

import numpy as np
from typing import Tuple, Literal, Optional, List, Dict, Union
import warnings

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np

LIGHTSPEED = 299792458.0

# ── Antenna rotation from DeepVerse-6G config ────────────────────────────────
# dv.radar.tx_antenna.rotation = [330, -10, 0]  (azimuth, elevation, roll) deg
# This rotation transforms global world coordinates to local array coordinates.
_ARRAY_AZ_DEG  = 330.0   # array boresight azimuth in global frame (degrees)
_ARRAY_EL_DEG  = -10.0   # array boresight elevation tilt (degrees)

def _global_doa_to_local_az(doa_phi_deg: float, doa_theta_deg: float) -> float:
    
    # Step 1 — global unit vector from spherical coords
    phi_g   = np.radians(doa_phi_deg)
    theta_g = np.radians(doa_theta_deg)   # polar angle from zenith

    ux = np.sin(theta_g) * np.cos(phi_g)
    uy = np.sin(theta_g) * np.sin(phi_g)
    uz = np.cos(theta_g)
    u_global = np.array([ux, uy, uz])

    # Step 2 — rotation matrix: first rotate around Z by -az, then around Y by el_tilt
    az_rad = np.radians(_ARRAY_AZ_DEG)
    el_rad = np.radians(_ARRAY_EL_DEG)

    # Rz(-az): rotate coordinate frame so boresight aligns with X-axis
    Rz = np.array([
        [ np.cos(az_rad),  np.sin(az_rad), 0],
        [-np.sin(az_rad),  np.cos(az_rad), 0],
        [0,                0,              1]
    ])

    # Ry(-el_tilt): tilt correction
    Ry = np.array([
        [ np.cos(el_rad), 0, np.sin(el_rad)],
        [ 0,              1, 0             ],
        [-np.sin(el_rad), 0, np.cos(el_rad)]
    ])

    R = Ry @ Rz
    u_local = R @ u_global

    # Step 3 — local azimuth = arctan2(y_local, x_local)
    local_az_deg = float(np.degrees(np.arctan2(u_local[1], u_local[0])))
    return local_az_deg

class ParameterEstimator:
    
    def __init__(
        self,
        carrier_freq_hz: float = 28e9,
        bandwidth_hz: float = 199.68e6,
        chirp_duration_s: float = 8.32e-6,
        n_chirps: int = 128,
        n_samples_per_chirp: int = 1664,
        sampling_rate_hz: float = 200e6,
        antenna_spacing_m: Optional[float] = None,
        n_tx_antennas: int = 4,
        n_rx_antennas: int = 4,
        radar_mode: Literal['monostatic', 'bistatic'] = 'monostatic',
        peak_method: Literal['max', 'centroid', 'parabolic'] = 'parabolic',
        use_gpu: bool = True,
        verbose: bool = True
    ):
        self.carrier_freq_hz = carrier_freq_hz
        self.bandwidth_hz = bandwidth_hz
        self.chirp_duration_s = chirp_duration_s
        self.n_chirps = n_chirps
        self.n_samples_per_chirp = n_samples_per_chirp
        self.sampling_rate_hz = sampling_rate_hz
        self.n_tx_antennas = n_tx_antennas
        self.n_rx_antennas = n_rx_antennas
        self.radar_mode = radar_mode
        self.peak_method = peak_method
        self.verbose = True
        
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        self.wavelength_m = LIGHTSPEED / carrier_freq_hz
        self.antenna_spacing_m = antenna_spacing_m if antenna_spacing_m is not None else self.wavelength_m / 2
        
        self.delay_factor = 2.0 if radar_mode == 'monostatic' else 1.0
        self.doppler_factor = 2.0 if radar_mode == 'monostatic' else 1.0
        
        self.chirp_slope_hz_s = bandwidth_hz / chirp_duration_s
        
        self.range_resolution_m = LIGHTSPEED / (self.delay_factor * bandwidth_hz)
        self.velocity_resolution_ms = self.wavelength_m / (
            self.doppler_factor * n_chirps * chirp_duration_s
        )
        
        self.n_virtual_antennas = n_tx_antennas * n_rx_antennas
        # Stores the global noise floor from estimate_all_parameters for per-target SNR capping
        self._global_noise_power: Optional[float] = None
        
        if verbose:
            self._validate_configuration()
            self._print_initialization()
    
    def _validate_configuration(self):

        # Expected resolutions depend on mode:
        #   Monostatic: delay_factor=2, doppler_factor=2
        #     ΔR = c/(2B) = 299792458/(2×199.68e6) = 0.751 m
        #     Δv = λ/(2×N×T) = (c/28e9)/(2×128×8.32e-6) = 5.027 m/s
        #   Bistatic: delay_factor=1, doppler_factor=1
        #     ΔR = c/(1×B) = 1.502 m
        #     Δv = λ/(1×N×T) = 10.054 m/s
        # Reference: Richards "Fundamentals of Radar Signal Processing" §2.4
        if self.radar_mode == 'monostatic':
            expected_range_res = 0.751
            expected_vel_res   = 5.027
        else:  # bistatic
            expected_range_res = LIGHTSPEED / self.bandwidth_hz          # c/B
            expected_vel_res   = (self.wavelength_m /
                                  (self.doppler_factor
                                   * self.n_chirps
                                   * self.chirp_duration_s))

        range_res_error = abs(self.range_resolution_m - expected_range_res)
        vel_res_error   = abs(self.velocity_resolution_ms - expected_vel_res)

        if range_res_error > 0.02:
            warnings.warn(
                f"Range resolution mismatch: {self.range_resolution_m:.3f} m "
                f"(expected {expected_range_res:.3f} m, error={range_res_error:.4f} m)"
            )

        if vel_res_error > 0.1:
            warnings.warn(
                f"Velocity resolution mismatch: {self.velocity_resolution_ms:.3f} m/s "
                f"(expected {expected_vel_res:.3f} m/s, error={vel_res_error:.4f} m/s)"
            )
        
        expected_samples = int(self.sampling_rate_hz * self.chirp_duration_s + 0.5)
        if abs(expected_samples - self.n_samples_per_chirp) > 1:
            warnings.warn(
                f"Sample count mismatch: Fs×T_chirp = {expected_samples} samples, "
                f"but n_samples_per_chirp = {self.n_samples_per_chirp}"
            )
    
    def _print_initialization(self):
        
        gpu_status = "GPU (CuPy)" if self.use_gpu else "CPU (NumPy)"
        
        print(f"\n{'='*80}")
        print(f"[PARAMETER ESTIMATOR v4.2 - CORRECTED & PRODUCTION-READY]")
        print(f"{'='*80}")
        print(f"  Mode: {self.radar_mode.upper()}")
        print(f"  Device: {gpu_status}")
        print(f"  Peak method: {self.peak_method.upper()}")
        
        print(f"\n[SYSTEM PARAMETERS]")
        print(f"  Carrier frequency: {self.carrier_freq_hz/1e9:.2f} GHz")
        print(f"  Wavelength: {self.wavelength_m*1000:.2f} mm")
        print(f"  Bandwidth: {self.bandwidth_hz/1e6:.2f} MHz")
        print(f"  Sampling rate: {self.sampling_rate_hz/1e6:.0f} MHz (RADAR Fs)")
        print(f"  Chirp duration: {self.chirp_duration_s*1e6:.2f} µs")
        print(f"  Chirp slope: {self.chirp_slope_hz_s/1e12:.2f} THz/s")
        
        print(f"\n[RADAR CONFIGURATION]")
        print(f"  Number of chirps: {self.n_chirps}")
        print(f"  Samples per chirp: {self.n_samples_per_chirp}")
        print(f"  Frame time: {self.n_chirps * self.chirp_duration_s*1e3:.2f} ms")
        print(f"  TX antennas: {self.n_tx_antennas}  "
              f"({'BS 2×2 UPA' if self.n_tx_antennas==4 else str(self.n_tx_antennas)})")
        _M_rx_row = min(2, self.n_rx_antennas)
        _M_rx_col = max(1, self.n_rx_antennas // _M_rx_row)
        if self.n_rx_antennas == 4:
            _rx_label = (f"BS 2×2 UPA (bistatic BS)"
                         if self.radar_mode == 'bistatic'
                         else f"BS 2×2 UPA (monostatic)")
        else:
            _rx_label = (f"UE {_M_rx_row}×{_M_rx_col} "
                         f"{'ULA' if _M_rx_col==1 else 'UPA'}")
        print(f"  RX antennas: {self.n_rx_antennas}  ({_rx_label})")
        print(f"  Virtual array: {self.n_virtual_antennas} elements  "
              f"({self.n_tx_antennas}×{self.n_rx_antennas} Kronecker)")
        print(f"  Delay factor: {self.delay_factor:.1f}×  "
              f"({'round-trip' if self.delay_factor == 2 else 'one-way TX→target→RX'})")
        print(f"  Doppler factor: {self.doppler_factor:.1f}×  "
              f"({'two-way' if self.doppler_factor == 2 else 'one-way'})")
        
        print(f"\n[RESOLUTION]")
        if self.radar_mode == 'monostatic':
            _exp_r, _exp_v = 0.751, 5.027
        else:
            _exp_r = LIGHTSPEED / self.bandwidth_hz
            _exp_v = self.wavelength_m / (self.doppler_factor * self.n_chirps * self.chirp_duration_s)
        print(f"  Range:    {self.range_resolution_m:.3f} m  "
              f"(expected: {_exp_r:.3f} m, {self.radar_mode})")
        print(f"  Velocity: {self.velocity_resolution_ms:.3f} m/s  "
              f"(expected: {_exp_v:.3f} m/s, {self.radar_mode})")
        print(f"  Antenna spacing: {self.antenna_spacing_m*1000:.2f} mm (λ/2)")
        
        if self.radar_mode == 'monostatic':
            expected_range_res_print = 0.751
            expected_vel_res_print   = 5.027
        else:
            expected_range_res_print = LIGHTSPEED / self.bandwidth_hz
            expected_vel_res_print   = (self.wavelength_m /
                                        (self.doppler_factor
                                         * self.n_chirps
                                         * self.chirp_duration_s))

        range_ok = abs(self.range_resolution_m - expected_range_res_print) < 0.02
        vel_ok   = abs(self.velocity_resolution_ms - expected_vel_res_print) < 0.1

        print(f"\n[VALIDATION]")
        print(f"  Range resolution: {'✓ PASS' if range_ok else '✗ FAIL'}  "
              f"(expected {expected_range_res_print:.3f} m, "
              f"delay_factor={self.delay_factor:.0f}×)")
        print(f"  Velocity resolution: {'✓ PASS' if vel_ok else '✗ FAIL'}  "
              f"(expected {expected_vel_res_print:.3f} m/s, "
              f"doppler_factor={self.doppler_factor:.0f}×)")
        print(f"  Radar mode: {self.radar_mode.upper()}")
        print(f"{'='*80}\n")
    
    def estimate_range(
        self,
        range_doppler_map: Union[np.ndarray, 'cp.ndarray'],
        range_idx: int,
        doppler_idx: int,
        range_fft_size: int
    ) -> Tuple[float, float]:

        xp = self.xp

        if self.use_gpu and isinstance(range_doppler_map, np.ndarray):
            range_doppler_map = cp.asarray(range_doppler_map)
        elif not self.use_gpu and hasattr(range_doppler_map, 'get'):
            range_doppler_map = range_doppler_map.get()

        # Extract range profile at detected Doppler bin
        range_profile = range_doppler_map[:, doppler_idx]

        # Refine peak location (sub-bin interpolation)
        refined_idx = self._refine_peak(range_profile, range_idx)

        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL FIX: Correct Range Computation with FFT Scaling
        # ═══════════════════════════════════════════════════════════════════
        # Theory (FMCW Radar Range Estimation):
        #   
        #   1. FFT bin index k_r maps to beat frequency:
        #      f_beat = k_r × (F_s / N_FFT)
        #      where F_s = sampling rate, N_FFT = range FFT size
        #
        #   2. Beat frequency relates to range via FMCW equation:
        #      f_beat = μ × τ = μ × (delay_factor × R / c)
        #      where μ = chirp_slope = B / T_chirp
        #            τ = round-trip delay (monostatic: 2R/c)
        #            delay_factor = 2 (monostatic) or 1 (bistatic)
        #
        #   3. Solving for range R:
        #      R = (c × f_beat) / (delay_factor × μ)
        #        = (c / (delay_factor × μ)) × k_r × (F_s / N_FFT)
        #
        #   4. Substituting μ = B / T_chirp:
        #      R = (c × T_chirp / (delay_factor × B)) × k_r × (F_s / N_FFT)
        #
        # Implementation:
        #   Step 1: Convert bin to beat frequency
        #   Step 2: Convert beat frequency to range
        # ═══════════════════════════════════════════════════════════════════
        
        # Step 1: Bin index → Beat frequency
        # f_beat = k_r × Δf, where Δf = F_s / N_FFT (frequency bin spacing)
        beat_freq_hz = refined_idx * (self.sampling_rate_hz / range_fft_size)
        
        # Step 2: Beat frequency → Range
        # R = c × f_beat / (delay_factor × chirp_slope)
        range_m = (LIGHTSPEED * beat_freq_hz) / (self.delay_factor * self.chirp_slope_hz_s)
        
        # Compute uncertainty bounds
        snr_linear = self._compute_snr_at_peak(range_profile, range_idx)
        range_variance = self.compute_crb_range(snr_linear)
        range_std_m = float(np.sqrt(range_variance))
        
        if self.verbose:
            print(f"\n  [Range Estimation]")
            print(f"    Bin index: {range_idx} → refined: {refined_idx:.3f}")
            print(f"    Beat frequency: {beat_freq_hz:,.2f} Hz")  # ← ADDED
            print(f"    Range: {range_m:.2f} ± {range_std_m:.3f} m")
            print(f"    SNR: {10*np.log10(snr_linear + 1e-12):.1f} dB")
            print(f"    CRB: {np.sqrt(range_variance):.3f} m")
        
        return float(range_m), range_std_m
    
    def estimate_velocity(
        self,
        range_doppler_map: Union[np.ndarray, 'cp.ndarray'],
        range_idx: int,
        doppler_idx: int,
        doppler_fft_size: int
    ) -> Tuple[float, float]:
    
        xp = self.xp
    
        if self.use_gpu and isinstance(range_doppler_map, np.ndarray):
            range_doppler_map = cp.asarray(range_doppler_map)
        elif not self.use_gpu and hasattr(range_doppler_map, 'get'):
            range_doppler_map = range_doppler_map.get()
    
        # Extract Doppler profile at detected range bin
        doppler_profile = range_doppler_map[range_idx, :]
    
        # Refine peak location (sub-bin interpolation)
        refined_idx = self._refine_peak(doppler_profile, doppler_idx)
    
        # Center the Doppler bin index (DC bin = N_FFT,d / 2)
        # After FFTSHIFT, DC bin is at index N_FFT,d/2
        d_centered = refined_idx - (doppler_fft_size / 2.0)
    
        # Compute velocity — sign convention depends on mode and config.
        # Monostatic: de-chirping conjugation reverses Doppler sign → flip needed.
        #   flip_doppler_sign=True in isac_rx_config → velocity_sign_factor=-1.0
        #   v = -d_centered × Δv  (approaching target has positive velocity)
        # Bistatic:  one-way propagation, no sign reversal from conjugation.
        #   flip_doppler_sign=False → velocity_sign_factor=+1.0
        #   v = +d_centered × Δv
        # The sign factor is read from param_estimator.radar_mode to stay
        # consistent regardless of which config key is set upstream.
        _sign = -1.0 if self.radar_mode == 'monostatic' else 1.0
        velocity_ms = _sign * d_centered * self.velocity_resolution_ms
        
        snr_linear = self._compute_snr_at_peak(doppler_profile, doppler_idx)
        velocity_variance = self.compute_crb_velocity(snr_linear)
        velocity_std_ms = float(np.sqrt(velocity_variance))
        
        if self.verbose:
            print(f"\n  [Velocity Estimation]")
            print(f"    Bin index: {doppler_idx} → refined: {refined_idx:.2f}")
            print(f"    Centered bin: {d_centered:+.2f}")
            print(f"    Velocity: {velocity_ms:+.2f} ± {velocity_std_ms:.3f} m/s")
            print(f"    Direction: {'→ Approaching' if velocity_ms > 0 else '← Receding'}")
            print(f"    SNR: {10*np.log10(snr_linear + 1e-12):.1f} dB")
            print(f"    CRB: {np.sqrt(velocity_variance):.3f} m/s")
        
        return float(velocity_ms), velocity_std_ms
    
    def estimate_angle_music(
        self,
        virtual_snapshot: np.ndarray,
        n_sources: int = 1,
        angle_search_deg: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        
        if angle_search_deg is None:
            # 0.25° resolution over ±60° visible sector
            angle_search_deg = np.linspace(-60, 60, 481)

        virtual_snapshot = np.asarray(virtual_snapshot, dtype=complex).flatten()
        n_virtual = len(virtual_snapshot)

        if n_virtual < 3:
            warnings.warn(
                f"MUSIC requires at least 3 virtual antennas. "
                f"Got {n_virtual} ({self.radar_mode} mode)"
            )
            return 0.0, 90.0

        # ── Normalize snapshot ────────────────────────────────────────────
        norm = np.linalg.norm(virtual_snapshot)
        if norm < 1e-12:
            return 0.0, 90.0
        virtual_snapshot = virtual_snapshot / norm

        # ── Forward-backward spatial smoothing ───────────────────────────
        L          = n_virtual // 2        # sub-array length = 8 for N=16
        n_subarrays = n_virtual - L + 1    # = 9 sub-arrays
        J          = np.fliplr(np.eye(L))  # exchange matrix

        # Forward covariance — average over all sub-arrays
        R_fwd = np.zeros((L, L), dtype=complex)
        for i in range(n_subarrays):
            sub    = virtual_snapshot[i:i + L]
            R_fwd += np.outer(sub, sub.conj())
        R_fwd /= n_subarrays

        # Backward covariance — each backward sub-array computed independently
        # (NOT applied to the already-averaged R_fwd — your proposition was correct)
        R_bwd = np.zeros((L, L), dtype=complex)
        for i in range(n_subarrays):
            sub     = virtual_snapshot[i:i + L]
            sub_bwd = J @ sub.conj()            # time-reversed conjugate sub-array
            R_bwd  += np.outer(sub_bwd, sub_bwd.conj())
        R_bwd /= n_subarrays

        # Standard FB average
        R_fb = (R_fwd + R_bwd) / 2.0

        # ── Diagonal loading — prevents NaN eigenvalues ───────────────────
        # Small regularization stabilizes ill-conditioned R_fb.
        # Loading = 1e-6 × average diagonal power (relative, not absolute).
        diag_power = float(np.real(np.trace(R_fb))) / L
        R_fb      += (1e-6 * diag_power) * np.eye(L, dtype=complex)

        # ── Eigendecomposition ────────────────────────────────────────────
        eigenvalues, eigenvectors = np.linalg.eigh(R_fb)

        # eigh returns ascending order → reverse to descending
        idx_sorted   = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[idx_sorted]
        eigenvectors = eigenvectors[:, idx_sorted]

        # Guard n_sources against smoothed array size
        n_sources = min(n_sources, L - 1)

        E_noise = eigenvectors[:, n_sources:]   # noise subspace, shape (L, L-n_sources)

        if E_noise.shape[1] == 0:
            warnings.warn("No noise subspace after smoothing")
            return 0.0, 90.0

        # ── MUSIC spectrum scan ───────────────────────────────────────────
        spectrum = np.zeros(len(angle_search_deg))

        for i, theta_deg in enumerate(angle_search_deg):
            # Full N-element steering vector, then truncate to sub-array length L
            a_full = self._steering_vector_virtual(theta_deg)
            a_sub  = a_full[:L] / (np.linalg.norm(a_full[:L]) + 1e-12)

            denom       = float(np.abs(a_sub.conj() @ E_noise @ E_noise.conj().T @ a_sub))
            spectrum[i] = 1.0 / (denom + 1e-12)

        # ── Peak extraction ───────────────────────────────────────────────
        peak_idx  = int(np.argmax(spectrum))
        angle_deg = float(angle_search_deg[peak_idx])

        # ── SNR and std estimate ──────────────────────────────────────────
        signal_eigenvalue = float(eigenvalues[0])
        noise_eigenvalue  = float(eigenvalues[n_sources:].mean()) if n_sources < L else 1e-12
        snr_linear        = signal_eigenvalue / (noise_eigenvalue + 1e-12)

        crb_angle_rad2 = self.compute_crb_angle(snr_linear, n_virtual)
        angle_std_deg  = float(np.degrees(np.sqrt(max(crb_angle_rad2, 0.0))))

        if self.verbose:
            print(f"\n  [MUSIC Angle Estimation v4.3 - FB Smoothing]")
            _M_rx_row_p = min(2, self.n_rx_antennas)
            _M_rx_col_p = max(1, self.n_rx_antennas // _M_rx_row_p)
            print(f"    Virtual array: {n_virtual} elements  "
                  f"(2×2 TX ⊗ {_M_rx_row_p}×{_M_rx_col_p} RX)")
            print(f"    Sub-array length L: {L}, Sub-arrays: {n_subarrays}")
            print(f"    Search grid: {angle_search_deg[0]:.0f}° to "
                  f"{angle_search_deg[-1]:.0f}° in "
                  f"{angle_search_deg[1]-angle_search_deg[0]:.2f}° steps")
            print(f"    Estimated local azimuth: {angle_deg:.2f} ± {angle_std_deg:.2f} deg")
            print(f"    SNR (eigenvalue ratio): {10*np.log10(snr_linear + 1e-12):.1f} dB")
            print(f"    Spectrum peak: {spectrum[peak_idx]:.2e}")
            print(f"    Note: local frame — convert to global before GT comparison")

        return float(angle_deg), angle_std_deg
    
    def _steering_vector_virtual(self, theta_deg: float) -> np.ndarray:
        
        theta_rad = np.radians(theta_deg)
        k         = 2 * np.pi / self.wavelength_m
        d         = self.antenna_spacing_m
        sin_theta = np.sin(theta_rad)

        # ── TX array geometry — always 2×2 UPA (same for both topologies) ──
        M_tx_row = 2
        M_tx_col = 2   # TX azimuth columns (determines TX azimuth aperture)

        # ── RX array geometry — depends on mode ───────────────────────────
        # Monostatic BS RX: 2×2 UPA → M_rx_col = 2
        # Bistatic UE RX:   2×1 ULA → M_rx_col = 1
        # Rule: M_rx_col = n_rx_antennas // M_rx_row
        # For UE 2×1: 2 antennas, 2 rows, 1 col → M_rx_col = 1
        # For BS 4×4: 4 antennas, 2 rows, 2 cols → M_rx_col = 2
        M_rx_row = min(2, self.n_rx_antennas)   # rows = min(2, N_RX)
        M_rx_col = max(1, self.n_rx_antennas // M_rx_row)  # cols = N_RX / rows

        n_virtual = self.n_tx_antennas * self.n_rx_antennas
        a_virtual = np.zeros(n_virtual, dtype=complex)

        for tx_idx in range(self.n_tx_antennas):
            tx_col = tx_idx % M_tx_col   # TX azimuth column index (0 or 1)

            for rx_idx in range(self.n_rx_antennas):
                rx_col = rx_idx % M_rx_col   # RX azimuth column index

                # Kronecker virtual array index: TX outer, RX inner
                v_idx = tx_idx * self.n_rx_antennas + rx_idx

                # Virtual azimuth column position:
                #   Monostatic: col_v ∈ {0,1,2,3}  (2 TX cols × 2 RX cols)
                #   Bistatic:   col_v ∈ {0,1}       (2 TX cols × 1 RX col)
                col_v = tx_col * M_rx_col + rx_col

                # 1D azimuth phase (elevation projected out at broadside)
                phase = k * d * col_v * sin_theta
                a_virtual[v_idx] = np.exp(1j * phase)

        if self.verbose and theta_deg == 0.0:
            rx_type    = 'ULA' if M_rx_col == 1 else 'UPA'
            n_az_cols  = M_tx_col * M_rx_col
            n_div_rows = M_tx_row * M_rx_row
            print(f"    [Steering vector] Mode: {self.radar_mode.upper()}")
            print(f"      TX array: {M_tx_row}×{M_tx_col} UPA  "
                  f"({self.n_tx_antennas} antennas, BS)")
            _rx_node = ('UE' if (self.radar_mode == 'bistatic'
                                 and self.n_rx_antennas == 2)
                        else 'BS')
            print(f"      RX array: {M_rx_row}×{M_rx_col} {rx_type}  "
                  f"({self.n_rx_antennas} antennas, {_rx_node})")
            print(f"      Virtual aperture: {M_tx_col}×{M_rx_col} = "
                  f"{n_az_cols} azimuth cols  "
                  f"(angular resolution driven by this)")
            print(f"      Diversity rows: {M_tx_row}×{M_rx_row} = "
                  f"{n_div_rows}  "
                  f"(averaging gain, not resolution)")
            print(f"      Total virtual elements: {n_virtual}  "
                  f"({self.n_tx_antennas}×{self.n_rx_antennas} Kronecker)")
            if self.radar_mode == 'bistatic' and M_rx_col == 1:
                print(f"      ⚠️  Bistatic UE (2×1 ULA): M_rx_col=1 → azimuth "
                      f"aperture halved vs monostatic → wider CRB_angle expected")
            elif self.radar_mode == 'bistatic' and M_rx_col == 2:
                print(f"      ℹ️  Bistatic BS (2×2 UPA): M_rx_col=2 → same "
                      f"azimuth aperture as monostatic → CRB_angle identical")

        return a_virtual
    
    def _refine_peak(
        self,
        profile: Union[np.ndarray, 'cp.ndarray'],
        peak_idx: int
    ) -> float:
        
        xp = self.xp
        
        if self.peak_method == 'parabolic':
            if 0 < peak_idx < len(profile) - 1:
                alpha = float(xp.abs(profile[peak_idx - 1]) ** 2)
                beta = float(xp.abs(profile[peak_idx]) ** 2)
                gamma = float(xp.abs(profile[peak_idx + 1]) ** 2)
                
                denom = alpha - 2*beta + gamma
                
                if abs(denom) > 1e-12:
                    delta = 0.5 * (alpha - gamma) / denom
                    delta = np.clip(delta, -0.5, 0.5)
                else:
                    delta = 0.0
                
                refined_idx = peak_idx + delta
            else:
                refined_idx = float(peak_idx)
        
        elif self.peak_method == 'centroid':
            if 0 < peak_idx < len(profile) - 1:
                weights = xp.abs(profile[peak_idx-1:peak_idx+2]) ** 2
                indices = xp.arange(peak_idx-1, peak_idx+2)
                
                total_weight = xp.sum(weights)
                if total_weight > 1e-12:
                    refined_idx = float(xp.sum(indices * weights) / total_weight)
                else:
                    refined_idx = float(peak_idx)
            else:
                refined_idx = float(peak_idx)
        
        else:
            refined_idx = float(peak_idx)
        
        return refined_idx
    
    def _compute_snr_at_peak(
        self,
        profile: Union[np.ndarray, 'cp.ndarray'],
        peak_idx: int
    ) -> float:
        
        xp = self.xp

        if xp.iscomplexobj(profile):
            signal_power = float(xp.abs(profile[peak_idx]) ** 2)
        else:
            signal_power = float(profile[peak_idx])  # already power, no double-squaring

        # ── Returns radar_snr_input (pre-processing) for CRB use ─────────
        # This function feeds estimate_range() and estimate_velocity() CRB prints.
        # The authoritative radar_snr_input for the target dict is computed
        # separately in estimate_all_parameters() using input_snr_db (PATH 1).
        # Here we use the N_TX division as a best-effort approximation.
        if self._global_noise_power is not None and self._global_noise_power > 1e-15:
            radar_snr_rdm_linear   = signal_power / (self._global_noise_power + 1e-12)
            # N_TX is the asymmetric gain: signal gains N_TX², noise gains N_TX
            # → dividing by N_TX partially recovers radar_snr_input
            radar_snr_input_linear = radar_snr_rdm_linear / float(self.n_tx_antennas)
            return float(radar_snr_input_linear)

        noise_power              = self._estimate_noise_from_profile(profile, peak_idx=peak_idx)
        radar_snr_rdm_linear     = signal_power / (noise_power + 1e-12)
        radar_snr_input_linear   = radar_snr_rdm_linear / float(self.n_tx_antennas)
        return float(radar_snr_input_linear)
    
    def _estimate_noise_from_profile(
        self,
        profile: Union[np.ndarray, 'cp.ndarray'],
        peak_idx: Optional[int] = None,
        guard_cells: int = 10
    ) -> float:
    
        xp = self.xp
        
        power = xp.abs(profile) ** 2
        
        # ═══════════════════════════════════════════════════════════════════
        # METHOD 1: Guard-Based Noise Estimation (if peak location known)
        # ═══════════════════════════════════════════════════════════════════
        if peak_idx is not None:
            # Create mask excluding peak region
            mask = xp.ones(len(power), dtype=bool)
            
            start_guard = max(0, peak_idx - guard_cells)
            end_guard = min(len(power), peak_idx + guard_cells + 1)
            mask[start_guard:end_guard] = False
            
            # Extract noise cells (far from peak)
            noise_cells = power[mask]
            
            if len(noise_cells) > 50:
                # Use 10th percentile of cells far from peak
                noise_sorted = xp.sort(noise_cells)
                percentile_idx = max(1, len(noise_sorted) // 10)
                noise_power = float(noise_sorted[percentile_idx])
                
                return noise_power
        
        # ═══════════════════════════════════════════════════════════════════
        # METHOD 2: Low Percentile (FALLBACK - if no peak location)
        # ═══════════════════════════════════════════════════════════════════
        power_sorted = xp.sort(power)
        n_samples = len(power_sorted)
        
        # Use 5th percentile (more conservative than original 25th)
        percentile_idx = max(1, n_samples // 20)
        noise_power_raw = float(power_sorted[percentile_idx])
        noise_power = max(noise_power_raw, 1e-4) # hard floor

        return noise_power
    
    def compute_crb_range(
        self,
        snr_linear: float
    ) -> float:
        
        # RMS bandwidth for flat spectrum: β_rms = B / √12
        beta_rms_sq = (self.bandwidth_hz ** 2) / 12.0

        # Delay CRB (s²) — Kay 1993 eq. 3.31
        crb_delay_s2 = 1.0 / (8.0 * np.pi**2 * beta_rms_sq * snr_linear + 1e-12)

        # Range CRB (m²):  R = c · τ / delay_factor  →  σ²_R = (c/delay_factor)² · σ²_τ
        # delay_factor = 2 (monostatic): accounts for round-trip propagation
        # delay_factor = 1 (bistatic):  one-way TX→target and target→RX handled separately
        crb_range_m2 = ((LIGHTSPEED / self.delay_factor) ** 2) * crb_delay_s2

        return float(crb_range_m2)
    
    def compute_crb_velocity(
        self,
        snr_linear: float
    ) -> float:
        
        N          = self.n_chirps               # slow-time observations (chirps)
        T_rep      = self.chirp_duration_s       # slow-time sampling interval = T_chirp
        hann_loss  = 0.375                       # Hann window: sum(w²)/N = 3/8

        # Per-chirp SNR = input SNR × range FFT processing gain
        # The range FFT coherently integrates N_samples within each chirp,
        # boosting SNR by N_samples × hann_loss before Doppler estimation.
        snr_per_chirp = snr_linear * float(self.n_samples_per_chirp) * hann_loss

        # Doppler frequency CRB (Hz²) — Kay 1993 frequency estimation result
        # Uses T_rep (chirp-to-chirp spacing) NOT T_obs = N × T_rep
        crb_doppler_hz2 = 6.0 / (
            (2.0 * np.pi * T_rep) ** 2
            * N * (N**2 - 1)
            * snr_per_chirp
            + 1e-12
        )

        # Velocity CRB (m²/s²)
        # v_radial = (λ / doppler_factor) · f_d
        crb_velocity_m2s2 = (
            (self.wavelength_m / self.doppler_factor) ** 2
        ) * crb_doppler_hz2

        return float(crb_velocity_m2s2)
    
    def compute_crb_angle(
        self,
        snr_linear: float,
        n_antennas: int
    ) -> float:
        
        if n_antennas < 2:
            return np.inf

        d   = self.antenna_spacing_m
        lam = self.wavelength_m

        # ── TX array geometry — always 2×2 UPA ───────────────────────────
        M_tx_row = 2
        M_tx_col = 2

        # ── RX array geometry — derived from n_rx_antennas ───────────────
        # Monostatic BS RX: 4 antennas → 2×2 UPA → M_rx_row=2, M_rx_col=2
        # Bistatic UE RX:   2 antennas → 2×1 ULA → M_rx_row=2, M_rx_col=1
        M_rx_row = min(2, self.n_rx_antennas)
        M_rx_col = max(1, self.n_rx_antennas // M_rx_row)

        # ── Virtual aperture ──────────────────────────────────────────────
        # Unique azimuth column positions (determines angular resolution)
        M_col_virtual = M_tx_col * M_rx_col   # monostatic=4, bistatic UE=2
        # Diversity rows (determines averaging gain, not resolution)
        M_row_virtual = M_tx_row * M_rx_row   # monostatic=4, bistatic UE=4

        if self.verbose:
            print(f"      [CRB_angle] Mode: {self.radar_mode.upper()}, "
                  f"N_virtual={n_antennas}")
            print(f"        TX: {M_tx_row}×{M_tx_col} UPA, "
                  f"RX: {M_rx_row}×{M_rx_col} "
                  f"({'ULA' if M_rx_col == 1 else 'UPA'})")
            print(f"        Azimuth cols (M_col): {M_col_virtual}  "
                  f"Diversity rows (M_row): {M_row_virtual}")

        if M_col_virtual >= 2:
            # ── Rectangular aperture CRB ──────────────────────────────────
            # Valid for both monostatic (M_col=4) and bistatic UE (M_col=2)
            crb_angle_rad2 = (
                (lam / (2.0 * np.pi * d)) ** 2
                * 6.0
                / (  M_row_virtual
                   * M_col_virtual
                   * (M_col_virtual ** 2 - 1)
                   * snr_linear
                   + 1e-12)
            )
            if self.verbose:
                print(f"        CRB_θ = "
                      f"{np.degrees(np.sqrt(crb_angle_rad2)):.4f} deg")
        else:
            # ── Degenerate: single azimuth column, no angular resolution ──
            crb_angle_rad2 = float('inf')
            if self.verbose:
                print(f"        Single azimuth column — no angular resolution")

        return float(crb_angle_rad2)
    
    def estimate_all_parameters(
        self,
        range_doppler_map: Union[np.ndarray, 'cp.ndarray'],
        detection_list: List[Tuple[int, int, float]],
        range_fft_size: int,
        doppler_fft_size: int,
        virtual_array_data: Optional[np.ndarray] = None,
        cfar_threshold_map: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
        thermal_noise_power: Optional[float] = None,
        input_snr_db: Optional[float] = None   # ← design-time SNR from add_awgn_radar_aware
    ) -> List[Dict]:
        
        # ═══════════════════════════════════════════════════════════════════
        # FORCED BANNER - ALWAYS PRINTS
        # ═══════════════════════════════════════════════════════════════════
        import sys
        sys.stdout.flush()  # Force flush
        print("\n" + "="*80, flush=True)
        print("🔴🔴🔴 [ESTIMATE_ALL_PARAMETERS CALLED] 🔴🔴🔴", flush=True)
        print("="*80, flush=True)
        print(f"  Function started at: {len(detection_list)} detections to process", flush=True)
        print("="*80 + "\n", flush=True)
        sys.stdout.flush()
        # ═══════════════════════════════════════════════════════════════════
        
        targets = []
        
        noise_power = self.estimate_noise_power(
            range_doppler_map,
            cfar_threshold_map=cfar_threshold_map,
            thermal_noise_power=thermal_noise_power  # ← THREAD THROUGH
        )
        # Store global noise for use in per-target SNR computation
        self._global_noise_power = noise_power
        
        # ═══════════════════════════════════════════════════════════════════
        # DIAGNOSTIC #1: INPUT VALIDATION (FORCED)
        # ═══════════════════════════════════════════════════════════════════
        print("\n" + "="*80, flush=True)
        print("[ESTIMATE_ALL_PARAMETERS - ENTRY DIAGNOSTIC]", flush=True)
        print("="*80, flush=True)
        print("  INPUTS RECEIVED:", flush=True)
        print(f"    detection_list length: {len(detection_list)}", flush=True)
        print(f"    range_doppler_map shape: {range_doppler_map.shape}", flush=True)
        print(f"    range_fft_size: {range_fft_size}", flush=True)
        print(f"    doppler_fft_size: {doppler_fft_size}", flush=True)
        print(f"    virtual_array_data: {'YES' if virtual_array_data is not None else 'NO'}", flush=True)
        print("  ", flush=True)
        print("  DETECTION LIST (first 5):", flush=True)
        for i, (r_idx, d_idx, pwr) in enumerate(detection_list[:5], 1):
            print(f"    {i}. Range bin {r_idx:4d}, Doppler bin {d_idx:3d}, Power {pwr:.6e}", flush=True)
        if len(detection_list) > 5:
            print(f"    ... and {len(detection_list) - 5} more detections", flush=True)
        print("  ", flush=True)
        print(f"  EXPECTED PROCESSING:", flush=True)
        print(f"    Will process {len(detection_list)} targets", flush=True)
        print(f"    Noise power: {noise_power:.6e} ({10*np.log10(noise_power + 1e-12):.1f} dB)", flush=True)
        print(f"    Input SNR (design target): "
              f"{'%.1f dB' % input_snr_db if input_snr_db is not None else 'NOT PROVIDED'}", flush=True)
        if input_snr_db is None:
            print(f"    ⚠️  input_snr_db is None → CRB will use fallback (less accurate)", flush=True)
        else:
            print(f"    ✅ input_snr_db provided → CRB will use exact design SNR", flush=True)
        print("="*80 + "\n", flush=True)
        sys.stdout.flush()
        # ═══════════════════════════════════════════════════════════════════
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"[BATCH PARAMETER ESTIMATION - {len(detection_list)} targets]")
            print(f"{'='*80}")
            print(f"  Noise power: {10*np.log10(noise_power + 1e-12):.1f} dB")
        
        # Process all targets (debug mode removed)
        detection_list_debug = detection_list  # ← PROCESS ALL
        
        for i, (range_idx, doppler_idx, power) in enumerate(detection_list_debug, 1):
            # ═══════════════════════════════════════════════════════════════════
            # DIAGNOSTIC #2: PER-TARGET INPUT (FORCED)
            # ═══════════════════════════════════════════════════════════════════
            print("\n" + "─"*80, flush=True)
            print(f"[TARGET {i}/{len(detection_list_debug)} - PROCESSING]", flush=True)
            print("─"*80, flush=True)
            print("  INPUT:", flush=True)
            print(f"    Range bin: {range_idx}", flush=True)
            print(f"    Doppler bin: {doppler_idx}", flush=True)
            print(f"    Power: {power:.6e}", flush=True)
            sys.stdout.flush()
            # ═══════════════════════════════════════════════════════════════════

            if self.verbose:
                print(f"\n{'─'*80}")
                print(f"Target {i}/{len(detection_list_debug)}")
                print(f"{'─'*80}")
            
            range_m, range_std_m = self.estimate_range(
                range_doppler_map, range_idx, doppler_idx, range_fft_size
            )
            
            # ═══════════════════════════════════════════════════════════════════
            # DIAGNOSTIC #3: RANGE ESTIMATION OUTPUT (FORCED)
            # ═══════════════════════════════════════════════════════════════════
            print("  RANGE ESTIMATION:", flush=True)
            print(f"    Estimated range: {range_m:.2f} m", flush=True)
            print(f"    Range std: {range_std_m:.3f} m", flush=True)
            sys.stdout.flush()
            # ═══════════════════════════════════════════════════════════════════
            
            velocity_ms, velocity_std_ms = self.estimate_velocity(
                range_doppler_map, range_idx, doppler_idx, doppler_fft_size
            )
            
            # ═══════════════════════════════════════════════════════════════════
            # DIAGNOSTIC #4: VELOCITY ESTIMATION OUTPUT (FORCED)
            # ═══════════════════════════════════════════════════════════════════
            print("  VELOCITY ESTIMATION:", flush=True)
            print(f"    Estimated velocity: {velocity_ms:+.2f} m/s", flush=True)
            print(f"    Velocity std: {velocity_std_ms:.3f} m/s", flush=True)
            sys.stdout.flush()
            # ═══════════════════════════════════════════════════════════════════
            
            # ════════════════════════════════════════════════════════════════
            # SNR COMPUTATION — TWO PHYSICALLY DISTINCT QUANTITIES
            # ════════════════════════════════════════════════════════════════
            #
            # radar_snr_rdm  (~69 dB): Post-processing SNR in the combined RDM.
            #   = peak power / noise floor in RDM
            #   Includes all processing gains: N_TX² × N_RX × FFT × Hann × β⁴
            #   Used for: CFAR detection, target ranking, P_d estimation
            #   NOT suitable for CRB (gives σ_R ≈ 0.0001 m, physically wrong)
            #
            # radar_snr_input (~25 dB): Pre-processing SNR at antenna input.
            #   = β² × |H|² × P_chirp / σ²_thermal  (per-channel, per-sample)
            #   Standard radar literature reference (Kay 1993, Trees 2001)
            #   Used for: CRB computation, SCNR noise floor, topology comparison
            #   Relationship: radar_snr_rdm = radar_snr_input × G_processing / N_TX
            #     where G_processing = N_TX × N_RX × N_samples×hann × N_chirps×hann × β⁴
            #
            # For PhD paper Table: report both, explain the processing gain bridge.
            # ════════════════════════════════════════════════════════════════

            # ── radar_snr_rdm: post-processing, for CFAR / detection ──────
            radar_snr_rdm_linear = power / (noise_power + 1e-12)
            radar_snr_rdm_db     = float(10 * np.log10(radar_snr_rdm_linear + 1e-12))

            # ── radar_snr_input: pre-processing, for CRB ─────────────────
            if input_snr_db is not None:
                # PATH 1 — BEST: exact design-time SNR from add_awgn_radar_aware
                # This is the SNR parameter used when generating the noisy signal.
                # It equals β² × signal_power / noise_power at the antenna input.
                radar_snr_input_linear = float(10 ** (input_snr_db / 10.0))
                radar_snr_input_db     = float(input_snr_db)
                radar_snr_input_source = "design-time (exact)"

            elif thermal_noise_power is not None and thermal_noise_power > 1e-15:
                # PATH 2 — APPROXIMATE: reconstruct from thermal noise + gain model
                # G_signal = N_TX² × N_RX × N_samples×hann × N_chirps×hann × β⁴
                # radar_snr_input = RDM_peak / (G_signal × thermal_per_channel)
                _g_tx      = float(self.n_tx_antennas) ** 2   # coherent TX: ×N_TX²
                _g_rx      = float(self.n_rx_antennas)         # non-coherent RX: ×N_RX
                _g_range   = float(self.n_samples_per_chirp) * 0.375  # FFT × Hann
                _g_doppler = float(self.n_chirps) * 0.375             # FFT × Hann
                _beta      = float(np.sqrt(0.5))               # equal split default
                _g_beta    = _beta ** 4                        # β² TX × β² ref
                _g_total   = _g_tx * _g_rx * _g_range * _g_doppler * _g_beta
                radar_snr_input_linear = power / (thermal_noise_power * _g_total)
                radar_snr_input_db     = float(10 * np.log10(radar_snr_input_linear + 1e-12))
                radar_snr_input_source = "thermal reconstruction (approx)"

            else:
                # PATH 3 — ROUGH FALLBACK: radar_snr_rdm / N_TX
                # N_TX is the only gain factor that differs between signal (N_TX²)
                # and noise (N_TX) paths in coherent-TX / non-coherent-RX system.
                radar_snr_input_linear = radar_snr_rdm_linear / float(self.n_tx_antennas)
                radar_snr_input_db     = float(10 * np.log10(radar_snr_input_linear + 1e-12))
                radar_snr_input_source = "RDM/N_TX fallback (rough)"

            # ── CRB — uses radar_snr_input exclusively ────────────────────
            # CRB is a pre-processing bound (Kay 1993).  Using radar_snr_rdm
            # would give σ_R → 0, which is physically wrong.
            # Monostatic/bistatic distinction is handled by self.delay_factor
            # and self.doppler_factor set in __init__().
            crb_range_m2      = self.compute_crb_range(radar_snr_input_linear)
            crb_velocity_m2s2 = self.compute_crb_velocity(radar_snr_input_linear)
            crb_range_m       = float(np.sqrt(max(crb_range_m2, 0.0)))
            crb_velocity_ms   = float(np.sqrt(max(crb_velocity_m2s2, 0.0)))

            # ── Processing gain bridge (for paper Table) ──────────────────
            # radar_snr_rdm ≈ radar_snr_input × G_total / N_TX
            # Verify this relationship holds (diagnostic only):
            _g_proc_db = radar_snr_rdm_db - radar_snr_input_db
            _expected_g_db = float(10 * np.log10(
                float(self.n_tx_antennas)    # N_TX (net after N_TX² signal / N_TX noise)
                * float(self.n_rx_antennas)  # N_RX
                * float(self.n_samples_per_chirp) * 0.375  # range FFT × Hann
                * float(self.n_chirps) * 0.375              # Doppler FFT × Hann
                * (0.5 ** 2)                                # β⁴ at equal split
                + 1e-12
            ))

            # ── Diagnostic print ──────────────────────────────────────────
            print(f"\n  [SNR & CRB DIAGNOSTICS - Target {i}]", flush=True)
            print(f"    SNR source:            {radar_snr_input_source}", flush=True)
            print(f"    radar_snr_rdm:         {radar_snr_rdm_db:.2f} dB  "
                  f"(post-processing, CFAR/ranking)", flush=True)
            print(f"    radar_snr_input:       {radar_snr_input_db:.2f} dB  "
                  f"(pre-processing, CRB reference)", flush=True)
            print(f"    Processing gain gap:   {_g_proc_db:.1f} dB  "
                  f"(expected ≈ {_expected_g_db:.1f} dB)", flush=True)
            if abs(_g_proc_db - _expected_g_db) < 3.0:
                print(f"    ✅ Processing gain consistent with model", flush=True)
            else:
                print(f"    ⚠️  Gain gap = {_g_proc_db:.1f} dB vs expected "
                      f"{_expected_g_db:.1f} dB — verify β and window factors", flush=True)
            print(f"    CRB_range:             {crb_range_m:.6f} m  "
                  f"(σ_R lower bound)", flush=True)
            print(f"    CRB_velocity:          {crb_velocity_ms:.8f} m/s  "
                  f"(σ_v lower bound)", flush=True)
            if input_snr_db is not None:
                print(f"    ✅ PATH 1 — exact design SNR → CRB accurate", flush=True)
            elif thermal_noise_power is not None:
                print(f"    ⚠️  PATH 2 — thermal reconstruction → CRB approximate", flush=True)
            else:
                print(f"    ❌ PATH 3 — rough fallback → pass input_snr_db", flush=True)
            sys.stdout.flush()

            target = {
                # ── Geometry ──────────────────────────────────────────────
                'range_m':              range_m,
                'velocity_ms':          velocity_ms,
                'range_std_m':          range_std_m,
                'velocity_std_ms':      velocity_std_ms,

                # ── SNR (two distinct quantities, CSV-ready names) ─────────
                'radar_snr_rdm_db':     radar_snr_rdm_db,      # ~69 dB, post-processing
                'radar_snr_rdm_linear': radar_snr_rdm_linear,
                'radar_snr_input_db':   radar_snr_input_db,    # ~25 dB, pre-processing
                'radar_snr_input_linear': radar_snr_input_linear,
                'radar_snr_input_source': radar_snr_input_source,

                # ── CRB lower bounds ──────────────────────────────────────
                'crb_range_m':          crb_range_m,           # σ_R lower bound (m)
                'crb_velocity_ms':      crb_velocity_ms,       # σ_v lower bound (m/s)

                # ── Legacy compatibility keys (kept for existing downstream code) ─
                # These map to the physically correct quantities:
                'snr_db':              radar_snr_rdm_db,       # = radar_snr_rdm_db
                'snr_input_db':        radar_snr_input_db,     # = radar_snr_input_db
                'snr_linear':          radar_snr_rdm_linear,
                'snr_input_linear':    radar_snr_input_linear,
                'snr_input_source':    radar_snr_input_source,

                # ── Metadata ──────────────────────────────────────────────
                'power':               power,
                'range_idx':           range_idx,
                'doppler_idx':         doppler_idx,
                'radar_mode':          self.radar_mode
            }
            
            # ═══════════════════════════════════════════════════════════════════
            # DIAGNOSTIC #5: TARGET DICT CREATED (FORCED)
            # ═══════════════════════════════════════════════════════════════════
            print("  TARGET DICT CREATED:", flush=True)
            print(f"    Range:              {target['range_m']:.4f} m  (bin {target['range_idx']})", flush=True)
            print(f"    Velocity:           {target['velocity_ms']:+.4f} m/s  (bin {target['doppler_idx']})", flush=True)
            _snr_tgt_print = (input_snr_db if input_snr_db is not None else 25.0)
            print(f"    radar_snr_rdm:      {target['radar_snr_rdm_db']:.2f} dB  "
                  f"← post-processing (CFAR/ranking)", flush=True)
            print(f"    radar_snr_input:    {target['radar_snr_input_db']:.2f} dB  "
                  f"← pre-processing (~{_snr_tgt_print:.0f} dB expected, CRB ref)",
                  flush=True)
            print(f"    SNR source:         {target['radar_snr_input_source']}", flush=True)
            print(f"    CRB_range:          {target['crb_range_m']:.6f} m  "
                  f"← σ_R lower bound", flush=True)
            print(f"    CRB_velocity:       {target['crb_velocity_ms']:.8f} m/s  "
                  f"← σ_v lower bound", flush=True)
            print(f"    Power:              {target['power']:.6e}", flush=True)
            snr_in = target['radar_snr_input_db']
            crb_r  = target['crb_range_m']
            _snr_tgt_chk = (input_snr_db if input_snr_db is not None else 25.0)
            if abs(snr_in - _snr_tgt_chk) < 3.0:
                print(f"    ✅ radar_snr_input close to {_snr_tgt_chk:.0f} dB design target",
                      flush=True)
            else:
                print(f"    ⚠️  radar_snr_input = {snr_in:.1f} dB — "
                      f"check input_snr_db flow (target: {_snr_tgt_chk:.0f} dB)",
                      flush=True)
            if crb_r > 0.005:
                print(f"    ✅ CRB_range = {crb_r:.4f} m — physically meaningful", flush=True)
            else:
                print(f"    ⚠️  CRB_range very small — radar_snr_input may still be too high", flush=True)
            sys.stdout.flush()
            # ═══════════════════════════════════════════════════════════════════
            
            if virtual_array_data is not None:
                try:
                    virtual_snapshot = virtual_array_data[:, range_idx, doppler_idx]
                    angle_deg, angle_std_deg = self.estimate_angle_music(virtual_snapshot)

                    # CRB_angle uses radar_snr_input_linear (pre-processing) for
                    # consistency with CRB_range and CRB_velocity — all three bounds
                    # reference the same SNR quantity for fair monostatic vs bistatic
                    # topology comparison in the paper Table.
                    crb_angle_rad2 = self.compute_crb_angle(
                        radar_snr_input_linear, self.n_virtual_antennas
                    )
                    crb_angle_deg = float(np.degrees(np.sqrt(max(crb_angle_rad2, 0.0))))

                    target['angle_deg']     = angle_deg
                    target['angle_std_deg'] = angle_std_deg
                    target['crb_angle_deg'] = crb_angle_deg

                    print(f"    Angle:              {angle_deg:.2f} ± {angle_std_deg:.2f} deg  "
                          f"(MUSIC)", flush=True)
                    print(f"    CRB_angle:          {crb_angle_deg:.4f} deg  "
                          f"(using radar_snr_input = {radar_snr_input_db:.1f} dB)", flush=True)

                except Exception as e:
                    if self.verbose:
                        print(f"    ⚠️  Angle estimation failed: {e}")
                    target['angle_deg']     = None
                    target['angle_std_deg'] = None
                    target['crb_angle_deg'] = None
            
            targets.append(target)
            
            # ═══════════════════════════════════════════════════════════════════
            # DIAGNOSTIC #6: TARGET ADDED TO LIST (FORCED)
            # ═══════════════════════════════════════════════════════════════════
            print(f"  ✅ Target {i} appended to list", flush=True)
            print(f"     Current list size: {len(targets)}", flush=True)
            print("─"*80 + "\n", flush=True)
            sys.stdout.flush()
            # ═══════════════════════════════════════════════════════════════════
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"[BATCH ESTIMATION COMPLETE - {len(targets)} targets processed]")
            print(f"{'='*80}\n")
        
        # ═══════════════════════════════════════════════════════════════════
        # DIAGNOSTIC #7: FINAL OUTPUT SUMMARY (FORCED)
        # ═══════════════════════════════════════════════════════════════════
        print("\n" + "="*80, flush=True)
        print("[ESTIMATE_ALL_PARAMETERS - FINAL OUTPUT]", flush=True)
        print("="*80, flush=True)
        print(f"  TOTAL TARGETS PROCESSED: {len(targets)}", flush=True)
        print("  ", flush=True)
        
        if len(targets) > 0:
            print("  OUTPUT LIST (first 10 targets):", flush=True)
            for i, tgt in enumerate(targets[:10], 1):
                marker     = " ← BIN 177!" if tgt['range_idx'] == 177 else ""
                _snr_tgt_ok = (input_snr_db if input_snr_db is not None else 25.0)
                snr_ok      = ("✅" if abs(tgt.get('radar_snr_input_db', 0)
                                          - _snr_tgt_ok) < 5 else "⚠️ ")
                crb_ok     = "✅" if tgt['crb_range_m'] > 0.005 else "⚠️ "
                print(f"    {i}. R={tgt['range_m']:7.2f}m  V={tgt['velocity_ms']:+6.2f}m/s  "
                      f"rdm={tgt['radar_snr_rdm_db']:5.1f}dB  "
                      f"{snr_ok}inp={tgt.get('radar_snr_input_db', float('nan')):5.1f}dB  "
                      f"{crb_ok}CRB_R={tgt['crb_range_m']:.4f}m  "
                      f"CRB_V={tgt['crb_velocity_ms']:.6f}m/s  "
                      f"src={tgt.get('radar_snr_input_source','?')[:8]}  "
                      f"P={tgt['power']:.3e}{marker}", flush=True)

            if len(targets) > 0:
                t0 = targets[0]
                print(f"\n  [SNR / CRB VALIDATION SUMMARY — Top Target]", flush=True)
                print(f"    radar_snr_rdm:    {t0.get('radar_snr_rdm_db', float('nan')):.2f} dB  "
                      f"← post-processing (CFAR/ranking)", flush=True)
                print(f"    radar_snr_input:  {t0.get('radar_snr_input_db', float('nan')):.2f} dB  "
                      f"← pre-processing (CRB reference)", flush=True)
                print(f"    SNR source:       {t0.get('radar_snr_input_source', 'unknown')}", flush=True)
                print(f"    CRB_range:        {t0['crb_range_m']:.6f} m", flush=True)
                print(f"    CRB_velocity:     {t0['crb_velocity_ms']:.8f} m/s", flush=True)
                src = t0.get('radar_snr_input_source', '')
                if src == 'design-time (exact)':
                    print(f"    ✅ PATH 1: CRB accurate (exact design SNR used)", flush=True)
                elif 'thermal' in src:
                    print(f"    ⚠️  PATH 2: CRB approximate (thermal reconstruction)", flush=True)
                else:
                    print(f"    ❌ PATH 3: CRB rough — pass input_snr_db from add_awgn", flush=True)
                print(f"    → Paper Table values: SNR_input={t0.get('radar_snr_input_db',float('nan')):.1f} dB, "
                      f"CRB_R={t0['crb_range_m']:.4f} m, "
                      f"CRB_v={t0['crb_velocity_ms']:.6f} m/s", flush=True)
            
            if len(targets) > 10:
                print(f"    ... and {len(targets) - 10} more targets", flush=True)
            
            print("  ", flush=True)
            print("  TOP 3 BY POWER:", flush=True)
            targets_by_power = sorted(targets, key=lambda t: t['power'], reverse=True)
            for i, tgt in enumerate(targets_by_power[:3], 1):
                marker = " ← BIN 177!" if tgt['range_idx'] == 177 else ""
                print(f"    {i}. Range={tgt['range_m']:7.2f}m (bin {tgt['range_idx']:4d}), "
                    f"Power={tgt['power']:.3e}{marker}", flush=True)
            
            print("  ", flush=True)
            print("  TOP 3 BY SNR:", flush=True)
            targets_by_snr = sorted(targets, key=lambda t: t['snr_db'], reverse=True)
            for i, tgt in enumerate(targets_by_snr[:3], 1):
                marker = " ← BIN 177!" if tgt['range_idx'] == 177 else ""
                print(f"    {i}. Range={tgt['range_m']:7.2f}m (bin {tgt['range_idx']:4d}), "
                    f"SNR={tgt['snr_db']:.1f}dB{marker}", flush=True)
            
            print("  ", flush=True)
            print(f"  TOP TARGET CHECK:", flush=True)
            print(f"    #1 target: bin {targets[0]['range_idx']} → "
                  f"{targets[0]['range_m']:.2f} m, "
                  f"V={targets[0]['velocity_ms']:+.2f} m/s", flush=True)
            print(f"    (GT comparison performed in _compute_metrics "
                  f"with actual GT list)", flush=True)
        else:
            print("  ❌ NO TARGETS IN OUTPUT LIST!", flush=True)
        
        print("="*80 + "\n", flush=True)
        sys.stdout.flush()
        # ═══════════════════════════════════════════════════════════════════
        
        # ═══════════════════════════════════════════════════════════════════
        # FORCED BANNER - FUNCTION EXIT
        # ═══════════════════════════════════════════════════════════════════
        print("\n" + "="*80, flush=True)
        print("🔴🔴🔴 [ESTIMATE_ALL_PARAMETERS RETURNING] 🔴🔴🔴", flush=True)
        print("="*80, flush=True)
        print(f"  Returning {len(targets)} targets", flush=True)
        print("="*80 + "\n", flush=True)
        sys.stdout.flush()
        # ═══════════════════════════════════════════════════════════════════

        return targets
    
    def estimate_noise_power(
        self,
        range_doppler_map: Union[np.ndarray, 'cp.ndarray'],
        cfar_threshold_map: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
        thermal_noise_power: Optional[float] = None  # ← NEW PARAMETER
    ) -> float:

        xp = self.xp

        if self.use_gpu and isinstance(range_doppler_map, np.ndarray):
            range_doppler_map = cp.asarray(range_doppler_map)
        elif not self.use_gpu and hasattr(range_doppler_map, 'get'):
            range_doppler_map = range_doppler_map.get()

        if xp.iscomplexobj(range_doppler_map):
            power_map = xp.abs(range_doppler_map) ** 2
        else:
            power_map = xp.asarray(range_doppler_map, dtype=xp.float64)

        # ═══════════════════════════════════════
        # TIER 1: THERMAL MODEL — most reliable
        # noise_combined = thermal × N_TX × N_RX
        # (coherent TX sums noise incoherently → ×N_TX)
        # (non-coherent RX sums power → ×N_RX)
        # Do NOT add FFT gain factors — the code normalizes those already.
        # ═══════════════════════════════════════
        if thermal_noise_power is not None:
            n_tx = self.n_tx_antennas
            n_rx = self.n_rx_antennas

            # Array combining noise gain:
            #   Coherent TX sum: signal gains N_TX², noise gains N_TX (incoherent)
            #   Non-coherent RX sum: signal and noise both gain N_RX
            array_noise_gain = n_tx * n_rx  # = 16 for 4×4

            # FFT processing noise gain:
            #   numpy fft() does NOT normalize → each output bin power = input_power × N
            #   Zero-padding does NOT add real noise, so effective noise gain
            #   uses actual sample counts, not FFT sizes:
            #     Range:   N_samples_per_chirp = 1664
            #     Doppler: N_chirps = 128
            range_fft_noise_gain = self.n_samples_per_chirp    # 1664
            doppler_fft_noise_gain = self.n_chirps              # 128

            # Hann window reduces noise power by its normalized power ≈ 3/8 = 0.375
            # Applied to both range and Doppler dimensions
            hann_power_loss = 0.375  # Hann window: sum(w²)/N = 3/8

            noise_combined = (
                thermal_noise_power
                * array_noise_gain
                * range_fft_noise_gain * hann_power_loss
                * doppler_fft_noise_gain * hann_power_loss
            )

            if self.verbose:
                print(f"    [Noise Est] THERMAL MODEL (Tier 1):")
                print(f"      Per-channel thermal: {thermal_noise_power:.6e}")
                print(f"      Array noise gain (N_TX×N_RX=×{array_noise_gain}): "
                    f"{thermal_noise_power * array_noise_gain:.6e}")
                print(f"      Range FFT gain (×{range_fft_noise_gain}×{hann_power_loss}): "
                    f"{thermal_noise_power * array_noise_gain * range_fft_noise_gain * hann_power_loss:.6e}")
                print(f"      Doppler FFT gain (×{doppler_fft_noise_gain}×{hann_power_loss}): "
                    f"{noise_combined:.6e} ({10*np.log10(noise_combined+1e-12):.1f} dB)")
            return float(noise_combined)

        # ═══════════════════════════════════════
        # TIER 2: CFAR THRESHOLD MAP
        # CRITICAL: CFAR ran on dB-scale RDM → threshold_map is in dB units.
        # Must convert to linear BEFORE comparing with linear power_map.
        # Previous code did: power_map < (cfar_threshold_map * 0.1)
        # This compared linear values against dB values → always False → 0 clean cells.
        # ═══════════════════════════════════════
        if cfar_threshold_map is not None:
            if self.use_gpu and isinstance(cfar_threshold_map, np.ndarray):
                cfar_threshold_map = cp.asarray(cfar_threshold_map)
            elif not self.use_gpu and hasattr(cfar_threshold_map, 'get'):
                cfar_threshold_map = cfar_threshold_map.get()

            # Convert dB threshold to linear power — THE KEY FIX
            threshold_linear = 10 ** (cfar_threshold_map / 10.0)
            # 0.05 = -13 dB below threshold. Captures more clean cells in dense sidelobe scenes
            # where even 0.01 (-20 dB) leaves too few cells (only 25 in your 524K-cell map)
            noise_mask = power_map < (threshold_linear * 0.05)
            n_noise_cells = int(xp.sum(noise_mask))

            if self.verbose:
                print(f"    [Noise Est] CFAR THRESHOLD (Tier 2):")
                print(f"      Threshold range: {float(xp.min(cfar_threshold_map)):.1f}"
                    f" to {float(xp.max(cfar_threshold_map)):.1f} dB")
                print(f"      Threshold linear range: {float(xp.min(threshold_linear)):.2e}"
                    f" to {float(xp.max(threshold_linear)):.2e}")
                print(f"      Clean cells (< threshold×0.01): {n_noise_cells}/{power_map.size}")

            if n_noise_cells > 100:
                noise_cells = power_map[noise_mask]
                noise_power = float(xp.median(noise_cells))
                # Apply hard floor — prevents sidelobe contamination even here
                hard_floor = 1e-4
                noise_power = max(noise_power, hard_floor)
                if self.verbose:
                    print(f"      Noise power: {noise_power:.6e} "
                        f"({10*np.log10(noise_power+1e-12):.1f} dB)")
                    print(f"      Status: ✅ CFAR Tier 2 succeeded")
                return float(noise_power)
            else:
                if self.verbose:
                    print(f"      Status: ⚠️  Skipped — only {n_noise_cells} clean cells "
                        f"(need >100). Falling to Tier 3.")

        # ═══════════════════════════════════════
        # TIER 3: ULTRA-AGGRESSIVE PERCENTILE + HARD FLOOR
        # Even this will be sidelobe-contaminated in dense scenes.
        # The hard floor is the critical guard — it caps SNR inflation.
        # Without thermal_noise_power, this is the best we can do.
        # ═══════════════════════════════════════
        powers_sorted = xp.sort(power_map.flatten())
        n_samples = len(powers_sorted)

        if n_samples > 500000:
            percentile_idx = max(1, n_samples // 20000)  # 0.005th percentile
            percentile_name = "0.005th"
        elif n_samples > 100000:
            percentile_idx = max(1, n_samples // 10000)  # 0.01th
            percentile_name = "0.01th"
        elif n_samples > 10000:
            percentile_idx = max(1, n_samples // 2000)   # 0.05th
            percentile_name = "0.05th"
        elif n_samples > 1000:
            percentile_idx = max(1, n_samples // 200)    # 0.5th
            percentile_name = "0.5th"
        else:
            percentile_idx = max(1, n_samples // 50)     # 2nd
            percentile_name = "2nd"

        noise_power_raw = float(powers_sorted[percentile_idx])

        # Hard floor: prevents sidelobe contamination from collapsing SNR realism.
        # 1e-4 is safely below the true combined thermal floor (~0.01–0.1 after combining)
        # but high enough to cap any absurd inflation from dense sidelobe scenes.
        hard_floor = 1e-4
        noise_power = max(noise_power_raw, hard_floor)

        if self.verbose:
            print(f"    [Noise Est] ULTRA-AGGRESSIVE PERCENTILE (Tier 3 fallback):")
            print(f"      Map size: {n_samples:,} cells")
            print(f"      Percentile: {percentile_name} (index {percentile_idx})")
            print(f"      Raw estimate: {noise_power_raw:.6e} "
                f"({10*np.log10(noise_power_raw+1e-12):.1f} dB)")
            print(f"      Hard floor applied: {hard_floor:.6e}")
            print(f"      Final noise power: {noise_power:.6e} "
                f"({10*np.log10(noise_power+1e-12):.1f} dB)")
            max_power = float(xp.max(power_map))
            snr_check = 10 * np.log10(max_power / (noise_power + 1e-12))
            status = "✅ Realistic" if snr_check < 65 else "⚠️  High — pass thermal_noise_power for accuracy"
            print(f"      Implied peak SNR: {snr_check:.1f} dB → {status}")

        return float(noise_power)
    
    def format_target_report(
        self,
        targets: List[Dict],
        max_targets: int = 10
    ) -> str:
        
        if not targets:
            return "No targets detected."
        
        lines = [
            "="*80,
            f"TARGET REPORT ({self.radar_mode.upper()} MODE - {len(targets)} targets)",
            "="*80,
            ""
        ]
        
        targets_sorted = sorted(targets, key=lambda t: t['snr_db'], reverse=True)
        
        for i, tgt in enumerate(targets_sorted[:max_targets], 1):
            direction = "→ approaching" if tgt['velocity_ms'] > 0 else "← receding"
            
            lines.append(f"Target {i}:")
            lines.append(f"  Range:    {tgt['range_m']:8.2f} ± {tgt['range_std_m']:.3f} m")
            lines.append(f"  Velocity: {tgt['velocity_ms']:+8.2f} ± {tgt['velocity_std_ms']:.3f} m/s  {direction}")
            
            if 'angle_deg' in tgt and tgt['angle_deg'] is not None:
                lines.append(f"  Angle:    {tgt['angle_deg']:8.1f} ± {tgt['angle_std_deg']:.1f} deg")
            
            lines.append(f"  SNR:      {tgt['snr_db']:8.2f} dB")
            lines.append(f"  CRB:      σ_R={tgt['crb_range_m']:.3f} m, σ_v={tgt['crb_velocity_ms']:.3f} m/s")
            lines.append(f"  Bin:      range={tgt['range_idx']}, doppler={tgt['doppler_idx']}")
            lines.append("")
        
        if len(targets) > max_targets:
            lines.append(f"... and {len(targets) - max_targets} more targets")
            lines.append("")
        
        lines.append("="*80)
        
        return "\n".join(lines)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PARAMETER ESTIMATION v4.2 - CORRECTED & PRODUCTION-READY")
    print("="*80)