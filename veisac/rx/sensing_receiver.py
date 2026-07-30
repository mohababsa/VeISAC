# sensing_receiver.py
"""
VeISAC — FMCW Sensing Receiver

Full Sen-RX chain: de-chirping, range/Doppler FFT, clutter removal, NLMS OFDM interference mitigation, CFAR detection, and MUSIC angle estimation.

Paper: "VeISAC: An End-to-End MIMO-OFDM-FMCW Framework for ISAC
        in 6G Vehicular Networks"
Authors: M. Ababsa, S. Ribouh, Y. El Hillali, A. Rivenq
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Union
import warnings

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np

LIGHTSPEED = 299792458.0


class SensingReceiver:
    
    def __init__(
        self,
        config,
        use_gpu: bool = True,
        verbose: bool = True
    ):
        self.config = config
        self.verbose = verbose
        
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        try:
            from veisac.rx.detection import CFARDetector
        except ImportError:
            raise ImportError("CFARDetector not found. Ensure detection.py v4.3 is available.")
        
        self.cfar_detector = CFARDetector(
            method=config.cfar_method,
            guard_cells=config.cfar_guard_cells,
            training_cells=config.cfar_training_cells,
            pfa=config.cfar_pfa,
            enable_2d=config.use_2d_cfar,
            use_gpu=use_gpu,
            verbose=True
        )
        
        try:
            from veisac.rx.estimation import ParameterEstimator
        except ImportError:
            raise ImportError("ParameterEstimator not found. Ensure estimation.py v4.2 is available.")
        
        self.param_estimator = ParameterEstimator(
            carrier_freq_hz=config.carrier_freq_hz,
            bandwidth_hz=config.fmcw_bandwidth_hz,
            chirp_duration_s=config.fmcw_t_chirp_s,
            n_chirps=config.fmcw_n_chirps,
            n_samples_per_chirp=config.fmcw_n_samples_per_chirp,
            sampling_rate_hz=config.fmcw_sampling_rate_hz,
            antenna_spacing_m=config.wavelength_m / 2,
            n_tx_antennas=config.n_radar_tx_antennas,
            n_rx_antennas=config.n_radar_rx_antennas,
            radar_mode=config.radar_mode,
            peak_method=config.peak_search_method,
            use_gpu=use_gpu,
            verbose=verbose
        )
        
        self._precompute_reference_chirp()

        # ========== OFDM INTERFERENCE MITIGATION CONFIGURATION ==========
        # Theory (Section 3.4 - OFDM Interference in Additive ISAC):
        #   In additive ISAC, the received sensing signal contains:
        #   r_sens(t) = β·H_radar ⊗ c(t) + α·H_radar ⊗ x_ofdm(t) + n(t)
        #                ↑ Desired echo      ↑ OFDM interference
        #
        # Mitigation Strategy: NLMS Adaptive Filtering
        #   - Reference: Known transmitted OFDM waveform x_ofdm(t)
        #   - Algorithm: Normalized Least-Mean-Squares (NLMS)
        #   - Operates: Before de-chirping (time-domain)
        #   - Estimates: Effective channel H_radar through which OFDM propagates
        #   - Subtracts: α·H_radar ⊗ x_ofdm(t) while preserving β·H_radar ⊗ c(t)
        #
        # Key Property: FMCW chirp c(t) is uncorrelated with OFDM x_ofdm(t)
        #   → Adaptive filter converges to OFDM interference only
        #   → Radar echo remains intact
        #
        # Literature:
        #   - Passive radar DPI/MPI cancellation (standard technique)
        #   - arXiv:2509.25750 (coordinated FMCW-OFDM)
        #   - arXiv:2507.20942 (multistatic OFDM-ISAC SIC)

        self.enable_ofdm_mitigation = getattr(config, 'enable_ofdm_mitigation', False)

        # NLMS filter length (number of taps)
        # Guideline: Should cover expected multipath delay spread
        #   - Urban: 32-64 taps (typical)
        #   - Indoor: 16-32 taps (shorter delay spread)
        #   - Rural: 64-128 taps (longer delay spread)
        self.nlms_filter_length = getattr(config, 'nlms_filter_length', 64)

        # NLMS step size (learning rate)
        # Guideline: Trade-off between convergence speed and stability
        #   - 0.001-0.01: Stable, slower convergence (recommended)
        #   - 0.01-0.05: Faster convergence, less stable
        #   - 0.05+: Fast but may diverge
        self.nlms_step_size = getattr(config, 'nlms_step_size', 0.01)

        # ========== MTI (Moving Target Indication) ==========
        self.enable_mti = getattr(config, 'enable_mti', True)  # Default: ON
        self.mti_velocity_threshold = getattr(config, 'mti_velocity_threshold', 2.0)  # m/s
        # ====================================================
        
        # Stores thermal noise passed from main script for accurate SNR computation
        self._thermal_noise_power: Optional[float] = None
        self._input_snr_db: Optional[float] = None   # ← design-time SNR from add_awgn
        
        if self.verbose:
            self._print_initialization()
    
    def _precompute_reference_chirp(self):
    
        xp = self.xp
        
        n_samples = self.config.fmcw_n_samples_per_chirp
        t_fast = np.arange(n_samples) / self.config.fmcw_sampling_rate_hz
        
        chirp_slope = self.config.fmcw_bandwidth_hz / self.config.fmcw_t_chirp_s
        
        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL FIX: β-SCALED REFERENCE CHIRP
        # ═══════════════════════════════════════════════════════════════════
        # Theory: In ISAC additive mode, the transmitted radar signal is:
        #   s_tx(t) = β · chirp(t)
        # where β = radar_power_factor (typically 0.707 for 50% power split)
        #
        # The received signal is:
        #   r(t) = β · H_radar ⊗ chirp(t) + noise
        #
        # For proper de-chirping, the reference must match the TX scaling:
        #   beat(t) = r(t) × conj(β · chirp(t))
        #           = [β · H ⊗ chirp] × conj(β · chirp)
        #           = β² · [H ⊗ chirp] × conj(chirp)  ← Correct amplitude!
        #
        # Without this scaling:
        #   beat(t) = [β · H ⊗ chirp] × conj(1.0 · chirp)  ← WRONG amplitude!
        #   → SNR over-estimated by factor of 1/β
        #   → CFAR threshold computed incorrectly
        #   → 300 false alarms, no real target detected
        # ═══════════════════════════════════════════════════════════════════
        
        # Get β from config (same factor used in TX path)
        beta = getattr(self.config, 'radar_power_factor', 1.0)
        
        phase = np.pi * chirp_slope * t_fast**2
        s_ref = beta * np.exp(1j * phase)  # ← FIXED: Now β-scaled
        
        self.reference_chirp_conj = xp.conj(xp.asarray(s_ref)) if self.use_gpu else np.conj(s_ref)
        
        if self.verbose:
            ref_power = float(np.mean(np.abs(s_ref)**2))
            print(f"\n[Reference Chirp Precomputed - β-SCALED]")
            print(f"  Samples: {n_samples}")
            print(f"  Chirp slope: {chirp_slope/1e12:.2f} THz/s")
            print(f"  Bandwidth: {self.config.fmcw_bandwidth_hz/1e6:.2f} MHz")
            print(f"  β (radar power factor): {beta:.6f}")
            print(f"  Reference chirp power: {ref_power:.6e} (should be β² = {beta**2:.6f})")
            if abs(ref_power - beta**2) > 0.01:
                print(f"  ⚠️  WARNING: Power mismatch! Expected {beta**2:.6f}, got {ref_power:.6e}")
            else:
                print(f"  ✅ Reference chirp correctly β-scaled")
    
    def _print_initialization(self):
        
        gpu_status = "GPU (CuPy)" if self.use_gpu else "CPU (NumPy)"
        
        print(f"\n{'='*80}")
        print(f"[SENSING RECEIVER v4.2 - CORRECTED & PRODUCTION-READY]")
        print(f"{'='*80}")
        print(f"  Mode: {self.config.radar_mode.upper()}")
        print(f"  Device: {gpu_status}")
        
        _mode_label = ('MONOSTATIC' if self.config.radar_mode == 'monostatic'
                       else 'BISTATIC')
        _delay_f    = getattr(self.config, 'delay_factor',
                              2.0 if self.config.radar_mode == 'monostatic' else 1.0)
        _doppler_f  = getattr(self.config, 'doppler_factor',
                              2.0 if self.config.radar_mode == 'monostatic' else 1.0)
        _delay_lbl  = 'round-trip' if _delay_f == 2.0 else 'one-way TX→target→RX'
        _dopp_lbl   = 'two-way'    if _doppler_f == 2.0 else 'one-way'
        if self.config.n_radar_rx_antennas == 4:
            _rx_label = ('BS 2×2 UPA (bistatic BS)'
                         if self.config.radar_mode == 'bistatic'
                         else 'BS 2×2 UPA (monostatic)')
        else:
            _rx_label = f'UE 2×{self.config.n_radar_rx_antennas // 2} ULA'

        print(f"\n[{_mode_label} CONFIGURATION]")
        print(f"  Mode: {self.config.radar_mode.upper()}")
        print(f"  RX antennas: {self.config.n_radar_rx_antennas}  ({_rx_label})")
        print(f"  TX antennas: {self.config.n_radar_tx_antennas}  (BS 2×2 UPA)")
        print(f"  Virtual array: {self.config.virtual_array_size} elements  "
              f"({self.config.n_radar_tx_antennas}×{self.config.n_radar_rx_antennas} Kronecker)")
        print(f"  Delay factor: {_delay_f:.0f}×  ({_delay_lbl})")
        print(f"  Doppler factor: {_doppler_f:.0f}×  ({_dopp_lbl})")
        
        print(f"\n[SIGNAL PROCESSING]")
        print(f"  RADAR Fs: {self.config.fmcw_sampling_rate_hz/1e6:.0f} MHz")
        print(f"  Chirp duration: {self.config.fmcw_t_chirp_s*1e6:.2f} µs")
        print(f"  Bandwidth: {self.config.fmcw_bandwidth_hz/1e6:.2f} MHz")
        print(f"  Chirps: {self.config.fmcw_n_chirps}")
        print(f"  Samples/chirp: {self.config.fmcw_n_samples_per_chirp}")

        # OFDM interference mitigation status
        if self.enable_ofdm_mitigation:
            print(f"  OFDM mitigation: NLMS adaptive filter")
            print(f"    Filter length: {self.nlms_filter_length} taps")
            print(f"    Step size: {self.nlms_step_size}")
        else:
            print(f"  OFDM mitigation: OFF")

        print(f"\n[PROCESSING CHAIN]")
        print(f"  Range FFT: {self.config.range_fft_size} points")
        print(f"  Doppler FFT: {self.config.doppler_fft_size} points")
        print(f"  Range window: {self.config.range_window}")
        print(f"  Doppler window: {self.config.doppler_window}")
        print(f"  Coherent TX combining: {self.config.coherent_combining.upper()}")
        print(f"  Non-coherent RX integration: {'ON' if self.config.use_non_coherent_integration else 'OFF'}")
        print(f"  Static clutter removal: {'ON' if self.config.remove_static_clutter else 'OFF'}")
        
        print(f"\n[DETECTION]")
        print(f"  CFAR method: {self.config.cfar_method.upper()}")
        print(f"  Guard cells: {self.config.cfar_guard_cells}")
        print(f"  Training cells: {self.config.cfar_training_cells}")
        print(f"  P_fa: {self.config.cfar_pfa:.1e}")
        print(f"  2D CFAR: {'ON' if self.config.use_2d_cfar else 'OFF'}")
        
        print(f"\n[RESOLUTION]")
        print(f"  Range:    {self.config.range_resolution_m:.3f} m  "
              f"(c / ({_delay_f:.0f}×B))")
        print(f"  Velocity: {self.config.doppler_resolution_ms:.3f} m/s  "
              f"(λ / ({_doppler_f:.0f}×N×T))")
        print(f"  Velocity sign: {self.config.velocity_sign_factor:+.1f}")
        
        print(f"{'='*80}\n")
    
    def process_radar_frame(
        self,
        received_signal: Union[np.ndarray, 'cp.ndarray'],
        ground_truth: Optional[Union[Dict, List[Dict]]] = None,
        detection_match_threshold_range_m: float = 6.3,
        detection_match_threshold_velocity_ms: float = 0.70,
        extract_virtual_array: bool = False,
        tx_comm_signal: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
        enable_ofdm_mitigation: Optional[bool] = None,
        thermal_noise_power: Optional[float] = None,
        input_snr_db: Optional[float] = None   # ← design-time SNR (e.g. 25.0 dB)
    ) -> Dict:

        # Store for downstream use in CRB computation and diagnostics
        self._thermal_noise_power = thermal_noise_power
        self._input_snr_db = input_snr_db

        print(f"\n  [process_radar_frame - SNR/CRB INPUTS]")
        print(f"    thermal_noise_power: "
              f"{'%.6e' % thermal_noise_power if thermal_noise_power is not None else 'NOT PROVIDED ⚠️'}")
        print(f"    input_snr_db:        "
              f"{'%.2f dB' % input_snr_db if input_snr_db is not None else 'NOT PROVIDED ⚠️'}")
        if input_snr_db is None:
            print(f"    ❌ CRB will use fallback — pass input_snr_db=25.0 from add_awgn_radar_aware")
        else:
            print(f"    ✅ CRB will use exact design SNR → accurate lower bounds")

        xp = self.xp
        
        if self.use_gpu and isinstance(received_signal, np.ndarray):
            received_signal = cp.asarray(received_signal)
        elif not self.use_gpu and hasattr(received_signal, 'get'):
            received_signal = received_signal.get()
        
        expected_shape = (
            self.config.n_radar_rx_antennas,
            self.config.n_radar_tx_antennas,
            self.config.fmcw_n_samples_per_chirp,
            self.config.fmcw_n_chirps
        )
        
        if received_signal.shape != expected_shape:
            raise ValueError(
                f"Received signal shape mismatch: got {received_signal.shape}, "
                f"expected {expected_shape} for {self.config.radar_mode} mode"
            )

        if ground_truth is not None:
            if isinstance(ground_truth, dict):
                gt_list = [ground_truth]
            elif isinstance(ground_truth, list):
                gt_list = ground_truth
            else:
                gt_list = []
        else:
            gt_list = []

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"[SENSING RECEIVER PROCESSING - 9-STEP PIPELINE]")
            print(f"{'='*80}")
            print(f"  Input shape: {received_signal.shape}")

        # Runtime override for OFDM mitigation (optional)
        if enable_ofdm_mitigation is not None:
            self.enable_ofdm_mitigation = enable_ofdm_mitigation

        # Pass tx_comm_signal for OFDM mitigation
        range_doppler_map, virtual_array_4d = self._compute_range_doppler(
            received_signal, 
            extract_virtual_array=extract_virtual_array,
            ground_truth=gt_list,
            tx_comm_signal=tx_comm_signal  # ← NEW PARAMETER
        )
        
        if self.verbose:
            print(f"\n{'─'*80}")
            print(f"[STEP 7] CFAR Detection")
            print(f"{'─'*80}")
        
        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL FIX: dB-Scale CFAR
        # ═══════════════════════════════════════════════════════════════════
        xp = self.xp
        rdm_linear = range_doppler_map.copy()

        if xp.iscomplexobj(rdm_linear):
            rdm_power = xp.abs(rdm_linear) ** 2
        else:
            rdm_power = rdm_linear.astype(xp.float64)

        epsilon = 1e-12
        rdm_db = 10 * xp.log10(rdm_power + epsilon)

        if self.verbose:
            print(f"  Converting RDM to dB scale for CFAR:")
            print(f"    Linear: min={float(xp.min(rdm_power)):.2e}, max={float(xp.max(rdm_power)):.2e}")
            print(f"    dB: min={float(xp.min(rdm_db)):.1f}, max={float(xp.max(rdm_db)):.1f} dB")
            print(f"    Dynamic range: {float(xp.max(rdm_db) - xp.min(rdm_db)):.1f} dB")
            print(f"  Using dB-scale RDM for CFAR detection")

        detections, threshold_map = self.cfar_detector.detect(
            rdm_db,  # ← Use dB scale!
            axis=None
        )
        # ═══════════════════════════════════════════════════════════════════
        # OPTIONAL: Store threshold map for metrics computation
        # ═══════════════════════════════════════════════════════════════════
        # Store the threshold map so _compute_metrics() can also use it
        # for consistent SNR estimation across all locations
        # ═══════════════════════════════════════════════════════════════════
        self._last_threshold_map = threshold_map
        
        # ═══════════════════════════════════════════════════════════════════
        # DIAGNOSTIC: CFAR INPUT/OUTPUT CONSISTENCY CHECK
        # ═══════════════════════════════════════════════════════════════════
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"[CFAR DETECTION OUTPUT - sensing_receiver.py]")
            print(f"{'='*80}")
            
            # Convert to CPU/NumPy for inspection
            xp = self.xp
            detections_cpu = cp.asnumpy(detections) if self.use_gpu else detections
            rdm_for_check = cp.asnumpy(rdm_db) if self.use_gpu else rdm_db
            
            # Total detections
            n_det = int(np.sum(detections_cpu))
            print(f"  Total CFAR detections: {n_det}")
            print(f"  Detection rate: {n_det / detections_cpu.size * 100:.2f}%")
            
            # Find strongest detected bin
            if n_det > 0:
                detected_powers = rdm_for_check[detections_cpu]
                strongest_idx = np.argmax(detected_powers)
                
                # Get location of strongest detection
                detected_locs = np.argwhere(detections_cpu)
                strongest_range_bin = detected_locs[strongest_idx, 0]
                strongest_doppler_bin = detected_locs[strongest_idx, 1]
                strongest_power_db = detected_powers[strongest_idx]
                
                print(f"  ")
                print(f"  STRONGEST CFAR DETECTION:")
                print(f"    Range bin: {strongest_range_bin}")
                print(f"    Doppler bin: {strongest_doppler_bin}")
                print(f"    Power: {strongest_power_db:.2f} dB")
                
            
            print(f"{'='*80}\n")
        # ═══════════════════════════════════════════════════════════════════
        
        if self.verbose:
            print(f"\n{'─'*80}")
            print(f"[STEP 8] Peak Extraction")
            print(f"{'─'*80}")

        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL: Use LINEAR power map for extraction (NOT dB!)
        # ═══════════════════════════════════════════════════════════════════
        # CFAR was run on dB-scale, but extract_detections() needs LINEAR power
        # to compute accurate SNR values. Use the saved rdm_linear.
        # ═══════════════════════════════════════════════════════════════════

        rd_map_cpu = cp.asnumpy(rdm_linear) if self.use_gpu else rdm_linear

        if np.iscomplexobj(rd_map_cpu) or rd_map_cpu.dtype == np.complex128:
            power_map = np.abs(rd_map_cpu) ** 2
            if self.verbose:
                print("  [EXTRACT] Using LINEAR power map (from rdm_linear)")
        else:
            power_map = rd_map_cpu.astype(np.float64, copy=False)
            if self.verbose:
                print("  [EXTRACT] Using LINEAR power map (already power)")
        # ═══════════════════════════════════════════════════════════════════════════
        
        # ═══════════════════════════════════════════════════════════════════
        # DIAGNOSTIC: POWER MAP CONSISTENCY CHECK BEFORE EXTRACTION
        # ═══════════════════════════════════════════════════════════════════
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"[POWER MAP CONSISTENCY - BEFORE extract_detections()]")
            print(f"{'='*80}")
            
            # Check if rd_map_cpu is complex or real
            if np.iscomplexobj(rd_map_cpu):
                print(f"  rd_map_cpu: COMPLEX → computing power")
                computed_power_map = np.abs(rd_map_cpu) ** 2
            else:
                print(f"  rd_map_cpu: REAL (already power)")
                print(f"  ⚠️  WARNING: If use_non_coherent_integration=True, rd_map is ALREADY power!")
                print(f"  ⚠️  Squaring it again will give POWER² (WRONG!)")
                computed_power_map = rd_map_cpu  # Don't square again!
            
            # Find strongest bin in power_map being passed to extraction
            max_power_extraction = float(np.max(power_map))
            max_loc_extraction = np.unravel_index(np.argmax(power_map), power_map.shape)
            extr_range_bin, extr_doppler_bin = max_loc_extraction
            
            print(f"  ")
            print(f"  POWER MAP BEING PASSED TO extract_detections():")
            print(f"    Strongest bin: ({extr_range_bin}, {extr_doppler_bin})")
            print(f"    Power: {max_power_extraction:.6e}")
            
            # Convert to range
            _cs_tmp  = self.config.fmcw_bandwidth_hz / self.config.fmcw_t_chirp_s
            _df_tmp  = float(self.param_estimator.delay_factor)
            extr_beat_freq = extr_range_bin * (self.config.fmcw_sampling_rate_hz / self.config.range_fft_size)
            extr_range_m   = extr_beat_freq * LIGHTSPEED / (_df_tmp * _cs_tmp)
            print(f"    Range: {extr_range_m:.2f} m")
            
            # CRITICAL CHECK: Is rd_map_cpu already power?
            if not np.iscomplexobj(rd_map_cpu):
                print(f"  ")
                print(f"  🔴 CRITICAL ISSUE DETECTED:")
                print(f"    rd_map_cpu is REAL-valued (dtype: {rd_map_cpu.dtype})")
                print(f"    This means use_non_coherent_integration=True")
                print(f"    → rd_map_cpu is ALREADY POWER (not complex)")
                print(f"    → Computing power_map = |rd_map_cpu|² is WRONG!")
                print(f"    → This squares power → power² → inflates values!")
                print(f"    ")
                print(f"    FIX NEEDED:")
                print(f"    if np.iscomplexobj(rd_map_cpu):")
                print(f"        power_map = np.abs(rd_map_cpu) ** 2")
                print(f"    else:")
                print(f"        power_map = rd_map_cpu  # Already power!")
            
            print(f"{'='*80}\n")
        # ═══════════════════════════════════════════════════════════════════
        
        # ═══════════════════════════════════════════════════════════════════
        # DIAGNOSTIC: PRE-EXTRACTION POWER MAP CHECK
        # ═══════════════════════════════════════════════════════════════════
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"[PRE-EXTRACTION DIAGNOSTIC - sensing_receiver.py]")
            print(f"{'='*80}")
            
            # Check power_map being passed to extraction
            print(f"  POWER MAP BEING PASSED TO extract_detections():")
            print(f"    Shape: {power_map.shape}")
            print(f"    Dtype: {power_map.dtype}")
            print(f"    Is complex: {np.iscomplexobj(power_map)}")
            
            # Find strongest bin in power_map
            max_power_val = float(np.max(power_map))
            max_loc = np.unravel_index(np.argmax(power_map), power_map.shape)
            max_range_bin, max_doppler_bin = max_loc
            
            print(f"  ")
            print(f"  STRONGEST BIN IN power_map:")
            print(f"    Range bin: {max_range_bin}")
            print(f"    Doppler bin: {max_doppler_bin}")
            print(f"    Power: {max_power_val:.6e}")
            
            # Cross-check with CFAR detections
            detections_cpu = cp.asnumpy(detections) if self.use_gpu else detections
            if detections_cpu.shape[0] > max_range_bin and detections_cpu.shape[1] > max_doppler_bin:
                strongest_detected = bool(detections_cpu[max_range_bin, max_doppler_bin])
                print(f"  ")
                print(f"  CFAR CROSS-CHECK:")
                print(f"    Strongest bin ({max_range_bin}, {max_doppler_bin}) detected by CFAR: {'✅ YES' if strongest_detected else '❌ NO'}")
                
                if not strongest_detected:
                    print(f"    Status: ❌ CRITICAL - Strongest bin NOT detected by CFAR!")
                    print(f"    → power_map and CFAR detection mask are inconsistent!")
            
            print(f"{'='*80}\n")
        # ═══════════════════════════════════════════════════════════════════

        # Phase 1 Fix #1: Raise limit to support dense multi-target scenes
        detection_list = self.cfar_detector.extract_detections(
            detections,
            power_map,
            max_detections=40  # ← CHANGED: was 100, now 300 (matches detection.py default)
        )

        # ═══════════════════════════════════════════════════════════════════
        # DIAGNOSTIC: POST-EXTRACTION OUTPUT CHECK
        # ═══════════════════════════════════════════════════════════════════
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"[POST-EXTRACTION DIAGNOSTIC - sensing_receiver.py]")
            print(f"{'='*80}")
            
            print(f"  EXTRACTION OUTPUT:")
            print(f"    Total detections extracted: {len(detection_list)}")
            
            if len(detection_list) > 0:
                # Show first detection
                first_det = detection_list[0]
                first_range_idx = first_det[0]
                first_doppler_idx = first_det[1]
                first_power = first_det[2]
                
                print(f"  ")
                print(f"  FIRST DETECTION (should be strongest):")
                print(f"    Range bin: {first_range_idx}")
                print(f"    Doppler bin: {first_doppler_idx}")
                print(f"    Power: {first_power:.6e}")
            
            print(f"{'='*80}\n")
        # ═══════════════════════════════════════════════════════════════════

        if self.verbose:
            print(f"  Total detections extracted: {len(detection_list)}")
            if len(detection_list) >= 100:
                print(f"    ⚠️  Reached extraction limit (100). Very dense scene!")
        
        if self.verbose:
            print(f"\n{'─'*80}")
            print(f"[STEP 9] Parameter Estimation")
            print(f"{'─'*80}")
        
        virtual_array_data = None
        if extract_virtual_array and virtual_array_4d is not None:
            if self.use_gpu and isinstance(virtual_array_4d, cp.ndarray):
                virtual_array_4d_cpu = cp.asnumpy(virtual_array_4d)
            else:
                virtual_array_4d_cpu = virtual_array_4d
    
            # Virtual array formation — valid for both monostatic and bistatic:
            #   Monostatic: (4,4,N_r,N_d) → 16 virtual elements
            #   Bistatic UE: (2,4,N_r,N_d) → 8 virtual elements
            # Kronecker order: a_v = a_tx ⊗ a_rx → TX-major reshape required
            # Transpose (n_rx,n_tx,...) → (n_tx,n_rx,...) BEFORE reshape
    
            n_virtual = self.config.n_radar_tx_antennas * self.config.n_radar_rx_antennas
            n_range, n_doppler = rd_map_cpu.shape
    
            # Reshape: (n_rx, n_tx, n_range, n_doppler) → (n_virtual, n_range, n_doppler)
            # CRITICAL: steering vector uses TX-outer, RX-inner (Kronecker: kron(a_tx, a_rx))
            # → must transpose to (n_tx, n_rx, n_range, n_doppler) BEFORE reshaping
            virtual_array_4d_txfirst = virtual_array_4d_cpu.transpose(1, 0, 2, 3)
            virtual_array_data = virtual_array_4d_txfirst.reshape(n_virtual, n_range, n_doppler)

            if self.verbose:
                print(f"  Virtual array extracted: {virtual_array_data.shape}")
                print(f"  Virtual array ordering: TX-major (matches kron(a_tx, a_rx))")
        
        # ═══════════════════════════════════════════════════════════════════
        # DIAGNOSTIC: PARAMETER ESTIMATOR INPUT
        # ═══════════════════════════════════════════════════════════════════
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"[PARAMETER ESTIMATOR INPUT - sensing_receiver.py]")
            print(f"{'='*80}")
            
            print(f"  INPUTS TO estimate_all_parameters():")
            print(f"    detection_list length: {len(detection_list)}")
            print(f"    range_fft_size: {self.config.range_fft_size}")
            print(f"    doppler_fft_size: {self.config.doppler_fft_size}")
            print(f"    rd_map_cpu shape: {rd_map_cpu.shape}")
            print(f"    rd_map_cpu dtype: {rd_map_cpu.dtype}")
            
            if len(detection_list) > 0:
                print(f"  ")
                print(f"  DETECTION LIST (first 5 being passed to estimator):")
                for i, det in enumerate(detection_list[:5], 1):
                    r_idx, d_idx, pwr = det
                    marker = " ← BIN 177!" if r_idx == 177 else ""
                    print(f"    {i}. Range bin {r_idx:4d}, Doppler bin {d_idx:3d}, Power {pwr:.6e}{marker}")
            print(f"  ")
            print(f"  ⏩ Calling param_estimator.estimate_all_parameters()...")
            print(f"{'='*80}\n")
        # ═══════════════════════════════════════════════════════════════════

        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL FIX: Pass CFAR threshold map for accurate noise estimation
        # ═══════════════════════════════════════════════════════════════════
        # The CFAR detector has already computed a per-cell threshold based on
        # local noise estimation. Cells below this threshold are guaranteed to
        # contain ONLY noise (no signal). This is the BEST noise estimator!
        #
        # By passing threshold_map to the parameter estimator, we enable:
        # 1. Accurate global noise floor estimation (not contaminated by sidelobes)
        # 2. Correct per-target SNR calculation
        # 3. Realistic SNR values (~25 dB instead of ~72 dB)
        # ═══════════════════════════════════════════════════════════════════

        # Convert threshold_map to CPU/NumPy if needed
        if self.use_gpu and hasattr(threshold_map, 'get'):
            threshold_map_cpu = threshold_map.get()
        else:
            threshold_map_cpu = threshold_map

        if self.verbose:
            print(f"  ")
            print(f"  CFAR THRESHOLD MAP:")
            print(f"    Shape: {threshold_map_cpu.shape}")
            print(f"    Dtype: {threshold_map_cpu.dtype}")
            print(f"    Min threshold: {float(np.min(threshold_map_cpu)):.6e}")
            print(f"    Max threshold: {float(np.max(threshold_map_cpu)):.6e}")
            print(f"    Mean threshold: {float(np.mean(threshold_map_cpu)):.6e}")
            print(f"  ✅ Passing CFAR threshold map to parameter estimator")

        targets = self.param_estimator.estimate_all_parameters(
            rd_map_cpu,
            detection_list,
            self.config.range_fft_size,
            self.config.doppler_fft_size,
            virtual_array_data=virtual_array_data,
            cfar_threshold_map=threshold_map_cpu,
            thermal_noise_power=thermal_noise_power,
            input_snr_db=self._input_snr_db          # ← thread exact design SNR for CRB
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # DIAGNOSTIC: PARAMETER ESTIMATOR OUTPUT
        # ═══════════════════════════════════════════════════════════════════
        if self.verbose and len(targets) > 0:
            print(f"\n{'='*80}")
            print(f"[PARAMETER ESTIMATOR OUTPUT - sensing_receiver.py]")
            print(f"{'='*80}")
            
            top_target = targets[0]
            
            print(f"  ESTIMATOR RETURNED {len(targets)} TARGETS")
            print(f"  ")
            print(f"  TOP TARGET (what will be reported):")
            print(f"    Range bin: {top_target.get('range_idx', 'N/A')}")
            print(f"    Doppler bin: {top_target.get('doppler_idx', 'N/A')}")
            print(f"    Range: {top_target['range_m']:.2f} m")
            print(f"    Velocity: {top_target['velocity_ms']:+.2f} m/s")
            print(f"    SNR: {top_target['snr_db']:.1f} dB")
            print(f"    Power: {top_target['power']:.6e}")
            
            # Check if it's bin 177
            top_range_idx = top_target.get('range_idx', -1)
            
            print(f"{'='*80}\n")
        elif self.verbose and len(targets) == 0:
            print(f"\n{'='*80}")
            print(f"[PARAMETER ESTIMATOR OUTPUT - sensing_receiver.py]")
            print(f"{'='*80}")
            print(f"  ❌ NO TARGETS RETURNED BY ESTIMATOR!")
            print(f"  Input had {len(detection_list)} detections")
            print(f"{'='*80}\n")
        # ═══════════════════════════════════════════════════════════════════
        
        # ── CRB / SNR_input diagnostic ────────────────────────────────────
        if self.verbose and len(targets) > 0:
            t0     = targets[0]
            rdm_db   = t0.get('radar_snr_rdm_db',   t0.get('snr_db',       float('nan')))
            input_db = t0.get('radar_snr_input_db', t0.get('snr_input_db', float('nan')))
            src      = t0.get('radar_snr_input_source', t0.get('snr_input_source', 'unknown'))
            crb_r    = t0['crb_range_m']
            crb_v    = t0['crb_velocity_ms']
            print(f"\n{'='*80}")
            print(f"[CRB & SNR DIAGNOSTIC - TOP TARGET]")
            print(f"{'='*80}")
            print(f"  radar_snr_rdm   (post-processing): {rdm_db:.2f} dB  "
                  f"← CFAR/ranking")
            _snr_tgt_diag = (input_snr_db if input_snr_db is not None else 25.0)
            print(f"  radar_snr_input (pre-processing):  {input_db:.2f} dB  "
                  f"← CRB reference (target: {_snr_tgt_diag:.0f} dB)")
            print(f"  SNR source:  {src}")
            _n_tx = self.config.n_radar_tx_antennas
            _n_rx = self.config.n_radar_rx_antennas
            print(f"  Processing gain (rdm - input): {rdm_db - input_db:.1f} dB  "
                  f"(mode: {self.config.radar_mode}, "
                  f"{_n_tx}×{_n_rx} MIMO)")
            print(f"  CRB_range:    {crb_r:.6f} m")
            print(f"  CRB_velocity: {crb_v:.8f} m/s")
            _snr_tgt_chk = (self._input_snr_db if self._input_snr_db is not None else 25.0)
            if abs(input_db - _snr_tgt_chk) < 3.0:
                print(f"  ✅ radar_snr_input = {input_db:.1f} dB — "
                      f"matches {_snr_tgt_chk:.0f} dB design target")
            else:
                print(f"  ❌ radar_snr_input = {input_db:.1f} dB — "
                      f"mismatch vs {_snr_tgt_chk:.0f} dB target, check input_snr_db flow")
            if crb_r > 0.005:
                print(f"  ✅ CRB_range = {crb_r:.4f} m — physically meaningful")
            else:
                print(f"  ❌ CRB_range nearly zero — radar_snr_input too high")
            print(f"{'='*80}\n")
        
        if self.verbose and len(targets) > 0:
            print(f"\n  Top target:")
            print(f"    Range: {targets[0]['range_m']:.2f} m")
            print(f"    Velocity: {targets[0]['velocity_ms']:+.2f} m/s")
            print(f"    SNR: {targets[0]['snr_db']:.1f} dB")
            if 'angle_deg' in targets[0] and targets[0]['angle_deg'] is not None:
                print(f"    Angle: {targets[0]['angle_deg']:.1f} deg")
        
        if self.verbose:
            print(f"\n{'─'*80}")
            print(f"[METRICS COMPUTATION]")
            print(f"{'─'*80}")
        
        metrics = self._compute_metrics(
            rd_map_cpu,
            targets,
            gt_list,
            detection_match_threshold_range_m,
            detection_match_threshold_velocity_ms
        )
        
        if ground_truth is not None:
            if self.verbose:
                print(f"\n  Computing P_fa...")
            
            p_fa, n_false_alarms = self.compute_false_alarm_probability(
                detections=detections,
                targets=targets,
                ground_truth=gt_list,
                range_threshold_m=detection_match_threshold_range_m,
                velocity_threshold_ms=detection_match_threshold_velocity_ms
            )
            
            metrics['false_alarm_probability'] = p_fa
            metrics['n_false_alarms'] = n_false_alarms
            
            if self.verbose:
                print(f"    P_fa: {p_fa:.6f} ({p_fa*100:.4f}%)")
                print(f"    False alarms: {n_false_alarms}")
        else:
            metrics['false_alarm_probability'] = None
            metrics['n_false_alarms'] = None
        
        if len(targets) > 0:
            if self.verbose:
                print(f"\n  Computing SCNR...")

            scnr_db, scnr_linear = self.compute_scnr(
                range_doppler_map=rd_map_cpu,
                targets=targets,
                clutter_guard_cells=5,
                thermal_noise_power=thermal_noise_power   # ← pass through
            )
            
            metrics['scnr_db'] = scnr_db
            metrics['scnr_linear'] = scnr_linear
            
            if self.verbose:
                print(f"    SCNR: {scnr_db:.2f} dB")
        else:
            metrics['scnr_db'] = None
            metrics['scnr_linear'] = None
        
        if self.verbose:
            self._print_summary(metrics, targets, gt_list)

        # ── RD MAP VISUALIZATION ──────────────────────────────────────────
        # Always pass the full gt_list and the same thresholds used in
        # _compute_metrics so the plot errors match the CSV exactly.
        self._visualize_range_doppler(
            rd_map_cpu,
            targets,
            gt_list,
            max_gt_display=100,
            range_threshold_m=detection_match_threshold_range_m,
            velocity_threshold_ms=detection_match_threshold_velocity_ms
        )
        # ─────────────────────────────────────────────────────────────────

        radar_dict = {
            'range_doppler_map': rd_map_cpu,
            'detections': detections,
            'threshold_map': threshold_map,
            'detection_list': detection_list,
            'targets': targets,
            **metrics
        }
        
        return radar_dict
    
    def _compute_range_doppler(
        self,
        received_signal: Union[np.ndarray, 'cp.ndarray'],
        extract_virtual_array: bool = False,
        ground_truth: Optional[Union[Dict, List[Dict]]] = None,
        tx_comm_signal: Optional[Union[np.ndarray, 'cp.ndarray']] = None
    ) -> Tuple[Union[np.ndarray, 'cp.ndarray'], Optional[Union[np.ndarray, 'cp.ndarray']]]:
        
        xp = self.xp
        n_rx, n_tx, n_samples, n_chirps = received_signal.shape

        # ── Mode-aware propagation factors ────────────────────────────────
        # delay_factor:   2 monostatic (round-trip),  1 bistatic (one-way)
        # doppler_factor: 2 monostatic (two-way),     1 bistatic (one-way)
        # These drive ALL range and velocity conversions in this method.
        # Reading from param_estimator ensures consistency with CRB formulas.
        _delay_factor   = float(self.param_estimator.delay_factor)
        _doppler_factor = float(self.param_estimator.doppler_factor)
        chirp_slope     = self.config.fmcw_bandwidth_hz / self.config.fmcw_t_chirp_s
        prf             = 1.0 / self.config.fmcw_t_chirp_s
        dc_bin_center   = self.config.doppler_fft_size // 2

        # ========== OFDM MITIGATION ==========
        if self.enable_ofdm_mitigation and tx_comm_signal is not None:
            print(f"\n{'─'*80}")
            print(f"[STEP 0] OFDM Interference Mitigation (NLMS)")
            print(f"{'─'*80}")
            received_signal = self._mitigate_ofdm_interference_nlms(received_signal, tx_comm_signal)
            print(f"  Signal after OFDM mitigation ready for de-chirping")
        elif self.enable_ofdm_mitigation and tx_comm_signal is None:
            warnings.warn("OFDM mitigation enabled but no tx_comm_signal provided. Skipping mitigation.")

        # ========== DE-CHIRPING ==========
        print(f"\n{'─'*80}")
        print(f"[STEP 1] De-chirping (Beat Signal Generation)")
        print(f"{'─'*80}")

        beat_signal = received_signal * self.reference_chirp_conj[None, None, :, None]

        # FORCE DIAGNOSTICS - ALWAYS PRINT
        rx_power_pre = float(xp.mean(xp.abs(received_signal)**2))
        beat_power_post = float(xp.mean(xp.abs(beat_signal)**2))
        ref_chirp_power = float(xp.mean(xp.abs(self.reference_chirp_conj)**2))
        beta = getattr(self.config, 'radar_power_factor', 1.0)
        
        print(f"\n{'='*80}")
        print(f"[DE-CHIRPING DIAGNOSTICS - FORCED PRINT]")
        print(f"{'='*80}")
        print(f"  RX signal power (before de-chirp): {rx_power_pre:.6e}")
        print(f"  Reference chirp power (|s_ref|²): {ref_chirp_power:.6e}")
        print(f"  Beat signal power (after de-chirp): {beat_power_post:.6e}")
        print(f"  ")
        print(f"  β from config: {beta:.6f}")
        print(f"  β² (expected ref power): {beta**2:.6f}")
        print(f"  ")
        print(f"  POWER RATIO ANALYSIS:")
        print(f"    Beat/RX ratio: {beat_power_post/rx_power_pre:.6f}")
        print(f"    Expected ratio: ~1.0 (if ref=1.0) or ~{beta**2:.6f} (if ref=β²)")
        print(f"  ")
        print(f"  REFERENCE CHIRP CHECK:")
        if abs(ref_chirp_power - 1.0) < 0.01:
            print(f"    Status: ❌ BROKEN - Reference has UNIT power (1.0)")
            print(f"    → Fix needed: Apply β-scaling to _precompute_reference_chirp()")
        elif abs(ref_chirp_power - beta**2) < 0.01:
            print(f"    Status: ✅ CORRECT - Reference is β-scaled (power = β²)")
        else:
            print(f"    Status: ⚠️  UNEXPECTED - Ref power = {ref_chirp_power:.6e}")
            print(f"    Expected: 1.0 (unit) or {beta**2:.6f} (β-scaled)")
        
        print(f"  ")
        print(f"  THEORETICAL VALIDATION:")
        print(f"    TX sends: β·chirp(t), power = β² = {beta**2:.6f}")
        print(f"    RX receives: β·H⊗chirp(t), power ≈ {beta**2:.6f} × |H|²")
        print(f"    De-chirp uses: ref_chirp_conj, power = {ref_chirp_power:.6f}")
        print(f"    Beat expected: β²·|H|² × ref_power ≈ {beta**2 * ref_chirp_power:.6f}")
        
        theoretical_beat_power = rx_power_pre * ref_chirp_power
        power_match = abs(beat_power_post - theoretical_beat_power) / (theoretical_beat_power + 1e-12)
        
        print(f"  ")
        print(f"  BEAT POWER VALIDATION:")
        print(f"    Theoretical: RX × |ref|² = {rx_power_pre:.6e} × {ref_chirp_power:.6e}")
        print(f"                             = {theoretical_beat_power:.6e}")
        print(f"    Measured: {beat_power_post:.6e}")
        print(f"    Relative error: {power_match*100:.2f}%")
        
        if power_match < 0.05:
            print(f"    Status: ✅ Beat power matches theory (< 5% error)")
        else:
            print(f"    Status: ⚠️  Beat power deviates ({power_match*100:.1f}% error)")
        
        print(f"{'='*80}\n")

        print(f"  De-chirping complete (vectorized)")
        print(f"  Output shape: {beat_signal.shape}")

        # ========== RANGE FFT ==========
        print(f"\n{'─'*80}")
        print(f"[STEP 2] Range FFT (Fast-Time Compression)")
        print(f"{'─'*80}")

        if self.config.range_window != 'none':
            range_window = self._get_window(n_samples, self.config.range_window)
            beat_signal_windowed = beat_signal * range_window[xp.newaxis, xp.newaxis, :, xp.newaxis]
            print(f"  Range window applied: {self.config.range_window}")
        else:
            beat_signal_windowed = beat_signal
            print(f"  Range window: none")

        range_fft = xp.fft.fft(beat_signal_windowed, n=self.config.range_fft_size, axis=2)
        range_fft_power = float(xp.mean(xp.abs(range_fft)**2))

        print(f"  Range FFT: {n_samples} → {self.config.range_fft_size} points")
        print(f"  Output shape: {range_fft.shape}")
        print(f"  Range FFT power: {range_fft_power:.6e}")

        # ═══════════════════════════════════════════════════════════════════════════
        # RANGE FFT BIN DIAGNOSTICS - FORCED
        # ═══════════════════════════════════════════════════════════════════════════
        print(f"\n{'='*80}")
        print(f"[RANGE FFT BIN DIAGNOSTICS - FORCED]")
        print(f"{'='*80}")

        # Average over all antennas and chirps to get range profile
        range_profile = xp.mean(xp.abs(range_fft)**2, axis=(0, 1, 3))  # Shape: (range_fft_size,)
        strongest_range_bin = int(xp.argmax(range_profile))
        strongest_range_power = float(range_profile[strongest_range_bin])

        print(f"  Range FFT size: {self.config.range_fft_size}")
        print(f"  Input samples: {n_samples}")
        print(f"  Zero-padding ratio: {self.config.range_fft_size / n_samples:.3f}×")
        print(f"  ")
        print(f"  STRONGEST RANGE BIN ANALYSIS:")
        print(f"    Bin index: {strongest_range_bin} / {self.config.range_fft_size}")
        print(f"    Power at bin: {strongest_range_power:.6e}")
        print(f"    Relative power: {strongest_range_power / range_fft_power:.2f}× avg")

        # Compute range from bin using mode-aware formula
        beat_freq      = strongest_range_bin * (self.config.fmcw_sampling_rate_hz / self.config.range_fft_size)
        range_from_bin = beat_freq * LIGHTSPEED / (_delay_factor * chirp_slope)

        print(f"  ")
        print(f"  RANGE CONVERSION (CURRENT FORMULA):")
        print(f"    Bin → Beat Freq: bin × (fs / N_fft)")
        print(f"    Beat frequency: {beat_freq:.2f} Hz")
        print(f"    Chirp slope μ: {chirp_slope:.6e} Hz/s")
        print(f"    Sampling rate fs: {self.config.fmcw_sampling_rate_hz:.6e} Hz")
        print(f"    Range formula: f_beat × c / ({_delay_factor:.0f}×μ)  "
              f"[{self.config.radar_mode}]")
        print(f"    Computed range: {range_from_bin:.2f} m")

        # ── Dynamic GT comparison (no hardcoded range) ────────────────────
        # Report strongest bin range — GT comparison is done in _compute_metrics
        bin_spacing_m = (LIGHTSPEED /
                         (_delay_factor * chirp_slope
                          * self.config.fmcw_sampling_rate_hz
                          / self.config.range_fft_size))
        print(f"  ")
        print(f"  RANGE BIN INFO ({self.config.radar_mode.upper()}):")
        print(f"    Range bin spacing: {bin_spacing_m:.4f} m/bin  "
              f"(delay_factor={_delay_factor:.0f})")
        print(f"    Strongest bin range: {range_from_bin:.2f} m")
        print(f"    Note: GT comparison performed in _compute_metrics with actual GT list")
        
        # Show top 5 range bins for context
        top5_bins = xp.argsort(range_profile)[-5:][::-1]
        print(f"  ")
        print(f"  TOP 5 RANGE BINS:")
        for i, bin_idx in enumerate(top5_bins, 1):
            bin_idx = int(bin_idx)
            bin_power = float(range_profile[bin_idx])
            bin_freq = bin_idx * (self.config.fmcw_sampling_rate_hz / self.config.range_fft_size)
            bin_range = bin_freq * LIGHTSPEED / (_delay_factor * chirp_slope)
            print(f"    {i}. Bin {bin_idx:4d}: {bin_range:7.2f} m, power={bin_power:.6e}")

        print(f"{'='*80}\n")
        # ═══════════════════════════════════════════════════════════════════════════

        # ========== DOPPLER FFT ==========
        print(f"\n{'─'*80}")
        print(f"[STEP 3] Doppler FFT (Slow-Time Compression)")
        print(f"{'─'*80}")

        if self.config.doppler_fft_size % 2 != 0:
            warnings.warn(f"Doppler FFT size ({self.config.doppler_fft_size}) is odd. May cause velocity errors.")

        if self.config.doppler_window != 'none':
            doppler_window = self._get_window(n_chirps, self.config.doppler_window)
            range_fft_windowed = range_fft * doppler_window[xp.newaxis, xp.newaxis, xp.newaxis, :]
            print(f"  Doppler window applied: {self.config.doppler_window}")
        else:
            range_fft_windowed = range_fft
            print(f"  Doppler window: none")

        range_doppler = xp.fft.fft(range_fft_windowed, n=self.config.doppler_fft_size, axis=3)

        # BEFORE fftshift
        doppler_power_before_shift = float(xp.mean(xp.abs(range_doppler)**2))
        dc_bin_before = 0  # DC is at bin 0 before shift

        # Apply fftshift
        range_doppler = xp.fft.fftshift(range_doppler, axes=-1)

        # AFTER fftshift
        doppler_fft_power = float(xp.mean(xp.abs(range_doppler)**2))
        dc_bin_after = self.config.doppler_fft_size // 2  # DC moves to center

        print(f"  Doppler FFT: {n_chirps} → {self.config.doppler_fft_size} points")
        print(f"  FFT shift applied (DC: bin {dc_bin_before} → bin {dc_bin_after})")
        print(f"  Output shape: {range_doppler.shape}")
        print(f"  Doppler FFT power: {doppler_fft_power:.6e}")

        # ═══════════════════════════════════════════════════════════════════════════
        # DOPPLER FFT BIN DIAGNOSTICS - FORCED
        # ═══════════════════════════════════════════════════════════════════════════
        print(f"\n{'='*80}")
        print(f"[DOPPLER FFT BIN DIAGNOSTICS - FORCED]")
        print(f"{'='*80}")

        # Extract Doppler profile at strongest range bin
        doppler_profile_at_peak_range = xp.mean(xp.abs(range_doppler[:, :, strongest_range_bin, :])**2, axis=(0, 1))
        strongest_doppler_bin = int(xp.argmax(doppler_profile_at_peak_range))
        strongest_doppler_power = float(doppler_profile_at_peak_range[strongest_doppler_bin])

        print(f"  Doppler FFT size: {self.config.doppler_fft_size}")
        print(f"  Input chirps: {n_chirps}")
        print(f"  Zero-padding ratio: {self.config.doppler_fft_size / n_chirps:.3f}×")
        print(f"  DC bin location (after fftshift): {dc_bin_after}")
        print(f"  ")
        print(f"  STRONGEST DOPPLER BIN (at range bin {strongest_range_bin}):")
        print(f"    Doppler bin: {strongest_doppler_bin} / {self.config.doppler_fft_size}")
        print(f"    Power at bin: {strongest_doppler_power:.6e}")

        # Compute velocity from Doppler bin
        # After fftshift, DC is at center, negative velocities in first half, positive in second half
        doppler_bin_centered = strongest_doppler_bin - dc_bin_after  # Shift to centered coordinates
        doppler_freq = doppler_bin_centered * (prf / self.config.doppler_fft_size)
        velocity_from_bin = doppler_freq * LIGHTSPEED / (_doppler_factor * self.config.carrier_freq_hz)

        print(f"  ")
        print(f"  VELOCITY CONVERSION (CURRENT FORMULA):")
        print(f"    Doppler bin (absolute): {strongest_doppler_bin}")
        print(f"    Doppler bin (centered): {doppler_bin_centered:+d} (DC = 0)")
        print(f"    PRF: {prf:.2f} Hz")
        print(f"    Doppler freq formula: bin_centered × (PRF / N_doppler)")
        print(f"    Doppler frequency: {doppler_freq:+.2f} Hz")
        print(f"    Carrier freq f_c: {self.config.carrier_freq_hz:.6e} Hz")
        print(f"    Velocity formula: f_d × c / ({_doppler_factor:.0f}·f_c)  "
              f"[{self.config.radar_mode}]")
        print(f"    Computed velocity: {velocity_from_bin:+.2f} m/s")

        # Compare with ground truth
        gt_velocity = 0.0  # Known from your data (static target)
        velocity_error = velocity_from_bin - gt_velocity

        print(f"  ")
        print(f"  GROUND TRUTH COMPARISON:")
        print(f"    GT velocity: {gt_velocity:+.2f} m/s")
        print(f"    Computed velocity: {velocity_from_bin:+.2f} m/s")
        print(f"    Velocity error: {velocity_error:+.2f} m/s")

        if abs(velocity_error) < 0.5:
            print(f"    Status: ✅ EXCELLENT - Velocity accurate")
        elif strongest_doppler_bin == dc_bin_after:
            print(f"    Status: ✅ CORRECT - Target at DC bin (static)")
        else:
            print(f"    Status: ❌ VELOCITY ERROR - Peak not at expected bin!")
            print(f"    Expected bin: {dc_bin_after} (DC)")
            print(f"    Actual bin: {strongest_doppler_bin}")

        # Show DC bin power (should be high for static targets)
        dc_bin_power = float(doppler_profile_at_peak_range[dc_bin_after])
        print(f"  ")
        print(f"  DC BIN POWER CHECK:")
        print(f"    DC bin ({dc_bin_after}) power: {dc_bin_power:.6e}")
        print(f"    Strongest bin ({strongest_doppler_bin}) power: {strongest_doppler_power:.6e}")
        print(f"    Ratio (strongest/DC): {strongest_doppler_power/dc_bin_power:.2f}×")

        if strongest_doppler_bin == dc_bin_after:
            print(f"    Status: ✅ Static target correctly at DC")
        else:
            print(f"    Status: ⚠️  Peak NOT at DC - unexpected for static target!")

        # Show top 5 Doppler bins
        top5_doppler_bins = xp.argsort(doppler_profile_at_peak_range)[-5:][::-1]
        print(f"  ")
        print(f"  TOP 5 DOPPLER BINS (at range bin {strongest_range_bin}):")
        for i, bin_idx in enumerate(top5_doppler_bins, 1):
            bin_idx = int(bin_idx)
            bin_power = float(doppler_profile_at_peak_range[bin_idx])
            bin_centered = bin_idx - dc_bin_after
            bin_doppler_freq = bin_centered * (prf / self.config.doppler_fft_size)
            bin_velocity = bin_doppler_freq * LIGHTSPEED / (_doppler_factor * self.config.carrier_freq_hz)
            dc_marker = " ← DC" if bin_idx == dc_bin_after else ""
            print(f"    {i}. Bin {bin_idx:3d} ({bin_centered:+4d}): {bin_velocity:+7.2f} m/s, power={bin_power:.6e}{dc_marker}")
        
        print(f"{'='*80}\n")
        # ═══════════════════════════════════════════════════════════════════════════

        # ═══════════════════════════════════════════════════════════════════════════
        # POST-DOPPLER FFT COMPLETE POWER MAP ANALYSIS
        # ═══════════════════════════════════════════════════════════════════════════
        print(f"\n{'='*80}")
        print(f"[POST-DOPPLER FFT - COMPLETE 4D POWER MAP CHECK]")
        print(f"{'='*80}")

        post_doppler_power = xp.abs(range_doppler)**2
        post_doppler_max = float(xp.max(post_doppler_power))
        post_doppler_mean = float(xp.mean(post_doppler_power))

        max_loc_4d = xp.unravel_index(xp.argmax(post_doppler_power), post_doppler_power.shape)
        pd_max_rx, pd_max_tx, pd_max_range, pd_max_doppler = [int(x) for x in max_loc_4d]

        # Convert to physical units
        pd_beat_freq = pd_max_range * (self.config.fmcw_sampling_rate_hz / self.config.range_fft_size)
        pd_range_m = pd_beat_freq * LIGHTSPEED / (_delay_factor * chirp_slope)

        pd_doppler_centered = pd_max_doppler - dc_bin_after
        pd_doppler_freq = pd_doppler_centered * (prf / self.config.doppler_fft_size)
        pd_velocity_ms = pd_doppler_freq * LIGHTSPEED / (_doppler_factor * self.config.carrier_freq_hz)

        print(f"  4D Array shape: {range_doppler.shape}")
        print(f"  Power stats: max={post_doppler_max:.6e}, mean={post_doppler_mean:.6e}")
        print(f"  ")
        print(f"  GLOBAL PEAK (across all RX/TX channels):")
        print(f"    RX: {pd_max_rx}/{n_rx}, TX: {pd_max_tx}/{n_tx}")
        print(f"    Range bin: {pd_max_range} → {pd_range_m:.2f} m")
        print(f"    Doppler bin: {pd_max_doppler} → {pd_velocity_ms:.2f} m/s")
        print(f"  ")

        print(f"{'='*80}\n")
        # ═══════════════════════════════════════════════════════════════════════════

        # ========== STATIC CLUTTER REMOVAL ==========
        print(f"\n{'─'*80}")
        print(f"[STEP 4] Static Clutter Removal")
        print(f"{'─'*80}")
        
        if self.config.remove_static_clutter:
            dc_bin = self.config.doppler_fft_size // 2
            protect_dc = False
            
            if ground_truth is not None:
                gt_check = [ground_truth] if isinstance(ground_truth, dict) else (ground_truth if isinstance(ground_truth, list) else [])
                protect_dc = any(abs(gt.get('velocity_ms', 0.0)) < self.mti_velocity_threshold for gt in gt_check)
            
            if protect_dc:
                print(f"  GT velocity near zero → SKIPPING DC zeroing to protect target")
            else:
                power_dc_before = float(xp.sum(xp.abs(range_doppler[:, :, :, dc_bin])**2))
                range_doppler[:, :, :, dc_bin] = 0
                print(f"  Removed static clutter at DC bin {dc_bin}")
                print(f"  Power removed: {power_dc_before:.6e}")
        else:
            print(f"  Static clutter removal: DISABLED")
            
        # ═══════════════════════════════════════════════════════════════════════════
        # POST-STATIC CLUTTER REMOVAL CHECK
        # ═══════════════════════════════════════════════════════════════════════════
        print(f"\n{'='*80}")
        print(f"[POST-STATIC CLUTTER - POWER MAP CHECK]")
        print(f"{'='*80}")

        sc_power = xp.abs(range_doppler)**2
        sc_max = float(xp.max(sc_power))

        sc_max_loc = xp.unravel_index(xp.argmax(sc_power), sc_power.shape)
        sc_rx, sc_tx, sc_range, sc_doppler = [int(x) for x in sc_max_loc]

        # Convert to range
        sc_beat_freq = sc_range * (self.config.fmcw_sampling_rate_hz / self.config.range_fft_size)
        sc_range_m = sc_beat_freq * LIGHTSPEED / (_delay_factor * chirp_slope)

        # Check DC bin power
        dc_bin = self.config.doppler_fft_size // 2
        dc_slice_power = float(xp.sum(xp.abs(range_doppler[:, :, :, dc_bin])**2))

        print(f"  Max power: {sc_max:.6e}")
        print(f"  DC bin total power: {dc_slice_power:.6e}")
        if dc_slice_power < 1e-20:
            print(f"    Status: ✅ DC successfully zeroed")
        else:
            print(f"    Status: ⚠️  DC NOT zeroed (protection active)")
        print(f"  ")
        print(f"  PEAK AFTER STATIC CLUTTER REMOVAL:")
        print(f"    Range bin: {sc_range} → {sc_range_m:.2f} m")
        print(f"    Doppler bin: {sc_doppler}")

        print(f"{'='*80}\n")
        # ═══════════════════════════════════════════════════════════════════════════

        # ========== MTI ==========
        print(f"\n{'─'*80}")
        print(f"[STEP 4.5] MTI - Doppler Clutter Suppression")
        print(f"{'─'*80}")
        
        if self.enable_mti:
            dc_bin = self.config.doppler_fft_size // 2
            power_before_mti = float(xp.sum(xp.abs(range_doppler)**2))
            
            # DC Notch
            range_doppler[:, :, :, dc_bin] = 0.0
            print(f"  DC Notch Filter applied (bin {dc_bin} zeroed)")
            
            # FIR High-Pass Filter
            try:
                from scipy.signal import firwin, lfilter
                taps = firwin(11, cutoff=0.05, window='hamming', pass_zero=False, fs=1.0)
                
                if hasattr(range_doppler, 'get'):
                    range_doppler = range_doppler.get()
                
                original_shape = range_doppler.shape
                N_rx, N_tx, N_range, N_doppler = original_shape
                rd_reshaped = range_doppler.reshape(-1, N_doppler)
                rd_filtered = np.zeros_like(rd_reshaped)
                
                for i in range(rd_reshaped.shape[0]):
                    rd_filtered[i, :] = lfilter(taps, 1.0, rd_reshaped[i, :])
                
                range_doppler = rd_filtered.reshape(original_shape)
                if self.use_gpu:
                    range_doppler = xp.asarray(range_doppler)
                
                print(f"  FIR High-Pass Filter applied (11 taps, cutoff=5% PRF)")
                
            except ImportError:
                print(f"  scipy.signal not available → DC notch only")
            
            power_after_mti = float(xp.sum(xp.abs(range_doppler)**2))
            print(f"  Power before MTI: {power_before_mti:.6e}")
            print(f"  Power after MTI: {power_after_mti:.6e}")
            print(f"  MTI suppression: {10*np.log10(power_before_mti/(power_after_mti+1e-12)):.2f} dB")
            print(f"  Velocity threshold: ~{self.mti_velocity_threshold} m/s")
        else:
            print(f"  MTI: DISABLED")
            
        # ═══════════════════════════════════════════════════════════════════════════
        # POST-MTI POWER MAP CHECK
        # ═══════════════════════════════════════════════════════════════════════════
        print(f"\n{'='*80}")
        print(f"[POST-MTI - POWER MAP CHECK]")
        print(f"{'='*80}")

        mti_power = xp.abs(range_doppler)**2
        mti_max = float(xp.max(mti_power))

        mti_max_loc = xp.unravel_index(xp.argmax(mti_power), mti_power.shape)
        mti_rx, mti_tx, mti_range, mti_doppler = [int(x) for x in mti_max_loc]

        # Convert to range
        mti_beat_freq = mti_range * (self.config.fmcw_sampling_rate_hz / self.config.range_fft_size)
        mti_range_m = mti_beat_freq * LIGHTSPEED / (_delay_factor * chirp_slope)

        print(f"  Max power: {mti_max:.6e}")
        print(f"  ")
        print(f"  PEAK AFTER MTI:")
        print(f"    Range bin: {mti_range} → {mti_range_m:.2f} m")
        print(f"    Doppler bin: {mti_doppler}")

        print(f"{'='*80}\n")
        # ═══════════════════════════════════════════════════════════════════════════

        # ═══════════════════════════════════════════════════════════════════════════
        # PRE-NORMALIZATION DIAGNOSTICS - FORCED (CRITICAL FOR DEBUG)
        # ═══════════════════════════════════════════════════════════════════════════
        print(f"\n{'='*80}")
        print(f"[PRE-NORMALIZATION POWER DIAGNOSTICS - FORCED]")
        print(f"{'='*80}")

        # Compute power map before any normalization/combining
        pre_norm_power_map = xp.abs(range_doppler)**2
        pre_norm_max_power = float(xp.max(pre_norm_power_map))
        pre_norm_mean_power = float(xp.mean(pre_norm_power_map))
        pre_norm_std_power = float(xp.std(pre_norm_power_map))

        # Find location of maximum power across all channels
        max_power_loc = xp.unravel_index(xp.argmax(pre_norm_power_map), pre_norm_power_map.shape)
        max_rx, max_tx, max_range, max_doppler = [int(x) for x in max_power_loc]

        print(f"  Range-Doppler array shape: {range_doppler.shape}")
        print(f"  Array dtype: {range_doppler.dtype}")
        print(f"  Dimensions: (n_rx={n_rx}, n_tx={n_tx}, n_range={self.config.range_fft_size}, n_doppler={self.config.doppler_fft_size})")
        print(f"  ")
        print(f"  POWER STATISTICS (BEFORE normalization):")
        print(f"    Max power: {pre_norm_max_power:.6e}")
        print(f"    Mean power: {pre_norm_mean_power:.6e}")
        print(f"    Std power: {pre_norm_std_power:.6e}")
        print(f"    Max/Mean ratio: {pre_norm_max_power/pre_norm_mean_power:.2f}×")
        print(f"    Max/Std ratio: {pre_norm_max_power/pre_norm_std_power:.2f}×")
        print(f"  ")
        print(f"  PEAK LOCATION (across all channels):")
        print(f"    RX antenna: {max_rx}/{n_rx}")
        print(f"    TX antenna: {max_tx}/{n_tx}")
        print(f"    Range bin: {max_range}/{self.config.range_fft_size}")
        print(f"    Doppler bin: {max_doppler}/{self.config.doppler_fft_size}")
        print(f"  ")

        # Cross-check with previously identified strongest bins
        try:
            if max_range == strongest_range_bin:
                print(f"  ✅ Peak range bin ({max_range}) matches Range FFT strongest bin")
            else:
                print(f"  ⚠️  Peak range bin ({max_range}) differs from Range FFT strongest ({strongest_range_bin})")
                print(f"     → May indicate antenna-specific peaks or processing artifact")
            
            if max_doppler == strongest_doppler_bin:
                print(f"  ✅ Peak Doppler bin ({max_doppler}) matches Doppler FFT strongest bin")
            else:
                print(f"  ⚠️  Peak Doppler bin ({max_doppler}) differs from Doppler FFT strongest ({strongest_doppler_bin})")
        except NameError:
            print(f"  ⚠️  Cannot cross-check bins (strongest_range_bin not defined)")

        # Estimate SNR BEFORE normalization (ground truth check)
        # Use percentile-based noise estimate for robustness
        power_sorted = xp.sort(pre_norm_power_map.flatten())
        n_cells = len(power_sorted)
        noise_estimate_prenorm = float(power_sorted[n_cells // 10])  # 10th percentile

        snr_prenorm_linear = pre_norm_max_power / (noise_estimate_prenorm + 1e-12)
        snr_prenorm_db = 10 * np.log10(snr_prenorm_linear + 1e-12)

        print(f"  ")
        print(f"  PRE-NORMALIZATION SNR ESTIMATE:")
        print(f"    Signal power (peak): {pre_norm_max_power:.6e}")
        print(f"    Noise estimate (10th percentile): {noise_estimate_prenorm:.6e}")
        print(f"    SNR (linear): {snr_prenorm_linear:.2f}")
        print(f"    SNR (dB): {snr_prenorm_db:.2f} dB")

        _snr_tgt = (self._input_snr_db if self._input_snr_db is not None else 25.0)
        if (_snr_tgt - 5) < snr_prenorm_db < (_snr_tgt + 10):
            print(f"    Status: ✅ SNR in expected range "
                  f"({_snr_tgt-5:.0f}–{_snr_tgt+10:.0f} dB for {_snr_tgt:.0f} dB target)")
        elif snr_prenorm_db > 100:
            print(f"    Status: ❌ SNR UNREALISTICALLY HIGH (>100 dB)")
            print(f"    → Issue: Likely noise floor under-estimation")
            print(f"    → Check: CFAR guards, per-channel effects, or power scaling")
        elif snr_prenorm_db < 10:
            print(f"    Status: ⚠️  SNR LOW (<10 dB)")
            print(f"    → Target may be weak or noise high")
        else:
            print(f"    Status: ⚠️  SNR = {snr_prenorm_db:.1f} dB - Verify if expected")

        # Per-channel power distribution (detect channel imbalance)
        print(f"  ")
        print(f"  PER-CHANNEL POWER DISTRIBUTION:")
        for rx_idx in range(n_rx):
            for tx_idx in range(n_tx):
                ch_power = float(xp.mean(xp.abs(range_doppler[rx_idx, tx_idx, :, :])**2))
                ch_max = float(xp.max(xp.abs(range_doppler[rx_idx, tx_idx, :, :])**2))
                print(f"    RX{rx_idx}-TX{tx_idx}: mean={ch_power:.6e}, max={ch_max:.6e}, max/mean={ch_max/ch_power:.1f}×")

        print(f"{'='*80}\n")
        # ═══════════════════════════════════════════════════════════════════════════

        # ========== PER-CHANNEL SCALING ==========
        print(f"\n{'─'*80}")
        print(f"[STEP 5] Per-Channel Amplitude Scaling")
        print(f"{'─'*80}")

        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL FIX: DISABLE PER-CHANNEL NORMALIZATION
        # ═══════════════════════════════════════════════════════════════════
        # Per-channel normalization destroys absolute power scale, which:
        #   1. Makes CFAR threshold computation unreliable
        #   2. Corrupts SNR estimation (can show 70 dB when it's 25 dB!)
        #   3. Reduces dynamic range (makes weak targets harder to detect)
        #
        # DECISION: ALWAYS DISABLE unless explicitly debugging channel imbalance
        # ═══════════════════════════════════════════════════════════════════

        FORCE_DISABLE_NORMALIZATION = True  # ← HARDCODED OVERRIDE

        if FORCE_DISABLE_NORMALIZATION:
            print(f"  Mode: ✅ DISABLED (FORCED - preserving power scale)")
            print(f"  Reason: Per-channel normalization corrupts CFAR and SNR")
            print(f"  ")
            print(f"  POWER PRESERVED:")
            
            # Measure current power
            pre_norm_power_map = xp.abs(range_doppler)**2
            pre_norm_max_power = float(xp.max(pre_norm_power_map))
            pre_norm_mean_power = float(xp.mean(pre_norm_power_map))
            
            print(f"    Max power: {pre_norm_max_power:.6e}")
            print(f"    Mean power: {pre_norm_mean_power:.6e}")
            
            # Estimate SNR without normalization
            power_sorted = xp.sort(pre_norm_power_map.flatten())
            n_cells = len(power_sorted)
            noise_estimate_prenorm = float(power_sorted[n_cells // 10])
            snr_prenorm_linear = pre_norm_max_power / (noise_estimate_prenorm + 1e-12)
            snr_prenorm_db = 10 * np.log10(snr_prenorm_linear + 1e-12)
            
            print(f"    SNR estimate: {snr_prenorm_db:.2f} dB")
            print(f"  ✅ Absolute power scale intact for accurate CFAR")

        elif self.config.amplitude_scaling_mode == 'per_channel':
            # ═══════════════════════════════════════════════════════════════
            # LEGACY CODE: Per-channel normalization (NOW DISABLED BY DEFAULT)
            # Only enable this if you explicitly set FORCE_DISABLE_NORMALIZATION = False
            # ═══════════════════════════════════════════════════════════════
            print(f"  Mode: PER-CHANNEL NORMALIZATION (⚠️  NOT RECOMMENDED!)")
            print(f"  ⚠️  WARNING: This will normalize each RX-TX channel independently")
            print(f"  ⚠️  Absolute power scale will be DESTROYED")
            print(f"  ⚠️  SNR estimation downstream may become unreliable")
            print(f"  ")
            
            # Store pre-normalization state for comparison
            pre_norm_total_power = float(xp.sum(xp.abs(range_doppler)**2))
            pre_norm_power_map = xp.abs(range_doppler)**2
            pre_norm_max_power = float(xp.max(pre_norm_power_map))
            pre_norm_mean_power = float(xp.mean(pre_norm_power_map))
            
            # Apply per-channel normalization
            for rx_idx in range(n_rx):
                for tx_idx in range(n_tx):
                    power = xp.abs(range_doppler[rx_idx, tx_idx, :, :]) ** 2
                    mean_power = xp.mean(power)
                    scale = 1.0 / xp.sqrt(mean_power + 1e-12)
                    range_doppler[rx_idx, tx_idx, :, :] *= scale
                    
                    if self.verbose:
                        post_power = float(xp.mean(xp.abs(range_doppler[rx_idx, tx_idx, :, :])**2))
                        print(f"    RX{rx_idx}-TX{tx_idx}: scale={float(scale):.4f}, post_power={post_power:.6f}")
            
            # Measure power after normalization
            post_norm_power_map = xp.abs(range_doppler)**2
            post_norm_max_power = float(xp.max(post_norm_power_map))
            post_norm_mean_power = float(xp.mean(post_norm_power_map))
            post_norm_total_power = float(xp.sum(post_norm_power_map))
            
            print(f"  ")
            print(f"  POST-NORMALIZATION POWER:")
            print(f"    Max power: {post_norm_max_power:.6e} (was {pre_norm_max_power:.6e})")
            print(f"    Mean power: {post_norm_mean_power:.6e} (was {pre_norm_mean_power:.6e})")
            print(f"    Total power: {post_norm_total_power:.6e} (was {pre_norm_total_power:.6e})")
            print(f"  ")
            print(f"  POWER SCALING FACTORS:")
            print(f"    Max power reduction: {pre_norm_max_power/post_norm_max_power:.2e}×")
            print(f"    Mean power change: {post_norm_mean_power/pre_norm_mean_power:.2e}×")
            print(f"    Total power change: {post_norm_total_power/pre_norm_total_power:.2e}×")
            
            # Re-estimate SNR
            noise_estimate_postnorm = float(xp.median(post_norm_power_map))
            snr_postnorm_linear = post_norm_max_power / (noise_estimate_postnorm + 1e-12)
            snr_postnorm_db = 10 * np.log10(snr_postnorm_linear + 1e-12)
            
            # Get pre-norm SNR
            power_sorted = xp.sort(pre_norm_power_map.flatten())
            n_cells = len(power_sorted)
            noise_estimate_prenorm = float(power_sorted[n_cells // 10])
            snr_prenorm_linear = pre_norm_max_power / (noise_estimate_prenorm + 1e-12)
            snr_prenorm_db = 10 * np.log10(snr_prenorm_linear + 1e-12)
            
            print(f"  ")
            print(f"  POST-NORMALIZATION SNR ESTIMATE:")
            print(f"    Signal power (peak): {post_norm_max_power:.6e}")
            print(f"    Noise estimate (median): {noise_estimate_postnorm:.6e}")
            print(f"    SNR (dB): {snr_postnorm_db:.2f} dB")
            print(f"  ")
            print(f"  SNR CHANGE DUE TO NORMALIZATION:")
            print(f"    Before: {snr_prenorm_db:.2f} dB")
            print(f"    After: {snr_postnorm_db:.2f} dB")
            print(f"    Delta: {snr_postnorm_db - snr_prenorm_db:+.2f} dB")
            
            if abs(snr_postnorm_db - snr_prenorm_db) > 10:
                print(f"    Status: ❌ CRITICAL - SNR changed by >10 dB!")
            else:
                print(f"    Status: ✅ SNR relatively stable")
            
            print(f"  ⚠️  Per-channel normalization APPLIED")
            
        else:
            # Original "disabled" path (no normalization)
            print(f"  Mode: DISABLED (no normalization)")
            print(f"  ✅ Absolute power scale PRESERVED")
            
            # Measure current power
            pre_norm_power_map = xp.abs(range_doppler)**2
            pre_norm_max_power = float(xp.max(pre_norm_power_map))
            pre_norm_mean_power = float(xp.mean(pre_norm_power_map))
            
            print(f"  ")
            print(f"  POWER PRESERVED:")
            print(f"    Max power: {pre_norm_max_power:.6e}")
            print(f"    Mean power: {pre_norm_mean_power:.6e}")
            
            # Estimate SNR
            power_sorted = xp.sort(pre_norm_power_map.flatten())
            n_cells = len(power_sorted)
            noise_estimate_prenorm = float(power_sorted[n_cells // 10])
            snr_prenorm_linear = pre_norm_max_power / (noise_estimate_prenorm + 1e-12)
            snr_prenorm_db = 10 * np.log10(snr_prenorm_linear + 1e-12)
            
            print(f"    SNR estimate: {snr_prenorm_db:.2f} dB")

        print(f"{'─'*80}\n")

        # ========== VIRTUAL ARRAY EXTRACTION ==========
        virtual_array_4d = None
        if extract_virtual_array:
            virtual_array_4d = range_doppler.copy()
            print(f"  Virtual array saved: {virtual_array_4d.shape}")
            if self.verbose:
                va_power = float(xp.mean(xp.abs(virtual_array_4d)**2))
                print(f"  Virtual array power: {va_power:.6e}")

        # ========== TX COMBINING ==========
        print(f"\n{'─'*80}")
        print(f"[STEP 6A] Coherent TX Combining")
        print(f"{'─'*80}")
        
        if self.config.coherent_combining == 'sum':
            range_doppler_tx = xp.sum(range_doppler, axis=1)
        elif self.config.coherent_combining == 'mean':
            range_doppler_tx = xp.mean(range_doppler, axis=1)
        else:
            raise ValueError(f"Unknown coherent combining: {self.config.coherent_combining}")
        
        tx_sum_power = float(xp.mean(xp.abs(range_doppler_tx)**2))
        expected_gain = 10*np.log10(n_tx**2 if self.config.coherent_combining=='sum' else 1)
        
        print(f"  Coherent TX {self.config.coherent_combining}: {n_tx} antennas")
        print(f"  Expected gain: +{expected_gain:.1f} dB")
        print(f"  Output shape: {range_doppler_tx.shape}")
        print(f"  Power after TX combining: {tx_sum_power:.6e}")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # POST-COHERENT TX COMBINING CHECK
        # ═══════════════════════════════════════════════════════════════════════════
        print(f"\n{'='*80}")
        print(f"[POST-TX COMBINING - 3D POWER MAP CHECK]")
        print(f"{'='*80}")

        tx_comb_power = xp.abs(range_doppler_tx)**2
        tx_max = float(xp.max(tx_comb_power))
        tx_mean = float(xp.mean(tx_comb_power))

        tx_max_loc = xp.unravel_index(xp.argmax(tx_comb_power), tx_comb_power.shape)
        tx_rx, tx_range, tx_doppler = [int(x) for x in tx_max_loc]

        # Convert to range
        tx_beat_freq = tx_range * (self.config.fmcw_sampling_rate_hz / self.config.range_fft_size)
        tx_range_m = tx_beat_freq * LIGHTSPEED / (_delay_factor * chirp_slope)

        print(f"  3D Array shape: {range_doppler_tx.shape}")
        print(f"  Expected: (n_rx={n_rx}, n_range={self.config.range_fft_size}, n_doppler={self.config.doppler_fft_size})")
        print(f"  Max power: {tx_max:.6e}, Mean: {tx_mean:.6e}")
        print(f"  ")
        print(f"  PEAK AFTER TX COMBINING:")
        print(f"    RX: {tx_rx}/{n_rx}")
        print(f"    Range bin: {tx_range} → {tx_range_m:.2f} m")
        print(f"    Doppler bin: {tx_doppler}")

        print(f"{'='*80}\n")
        # ═══════════════════════════════════════════════════════════════════════════

        # ========== RX INTEGRATION ==========
        print(f"\n{'─'*80}")
        print(f"[STEP 6B] Non-Coherent RX Integration")
        print(f"{'─'*80}")
        
        if self.config.use_non_coherent_integration:
            power_per_rx = xp.abs(range_doppler_tx) ** 2
            range_doppler_map = xp.sum(power_per_rx, axis=0)
            range_doppler_map = range_doppler_map.astype(xp.float64, copy=False)
            rdm_power = float(xp.mean(range_doppler_map))
            
            print(f"  Non-coherent RX integration: {n_rx} antennas")
            print(f"  Expected diversity gain: +{10*np.log10(n_rx):.1f} dB")
            print(f"  Output shape: {range_doppler_map.shape}")
            print(f"  Output dtype: {range_doppler_map.dtype} (real-valued power)")
            print(f"  RDM power: {rdm_power:.6e}")
        else:
            range_doppler_map = xp.mean(range_doppler_tx, axis=0)
            print(f"  RX averaging (coherent): {n_rx} antennas")
            print(f"  Output shape: {range_doppler_map.shape}")
            print(f"  Output dtype: {range_doppler_map.dtype} (complex-valued)")
        
        total_gain_db = (
            (10*np.log10(n_tx**2) if self.config.coherent_combining=='sum' else 0) +
            (10*np.log10(n_rx) if self.config.use_non_coherent_integration else 0)
        )
        print(f"  Total expected SNR gain: +{total_gain_db:.1f} dB")
        print(f"{'─'*80}\n")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # FINAL RDM CHECK - WHAT GOES TO CFAR
        # ═══════════════════════════════════════════════════════════════════════════
        print(f"\n{'='*80}")
        print(f"[FINAL RDM - PRE-CFAR INPUT]")
        print(f"{'='*80}")

        final_rdm = range_doppler_map
        final_max = float(xp.max(final_rdm))
        final_mean = float(xp.mean(final_rdm))

        final_max_loc = xp.unravel_index(xp.argmax(final_rdm), final_rdm.shape)
        final_range, final_doppler = [int(x) for x in final_max_loc]

        # Convert to physical units
        final_beat_freq = final_range * (self.config.fmcw_sampling_rate_hz / self.config.range_fft_size)
        final_range_m = final_beat_freq * LIGHTSPEED / (_delay_factor * chirp_slope)
        final_doppler_centered = final_doppler - (self.config.doppler_fft_size // 2)
        final_doppler_freq     = final_doppler_centered * (prf / self.config.doppler_fft_size)
        final_velocity_ms      = (final_doppler_freq * LIGHTSPEED
                                  / (_doppler_factor * self.config.carrier_freq_hz))

        print(f"  Final RDM shape: {final_rdm.shape}")
        print(f"  Final RDM dtype: {final_rdm.dtype}")
        print(f"  Expected: (n_range={self.config.range_fft_size}, n_doppler={self.config.doppler_fft_size})")
        print(f"  ")
        print(f"  FINAL POWER STATISTICS:")
        print(f"    Max: {final_max:.6e}")
        print(f"    Mean: {final_mean:.6e}")
        print(f"    Max/Mean: {final_max/final_mean:.2f}×")
        print(f"  ")
        print(f"  FINAL PEAK (what CFAR will detect):")
        print(f"    Range bin: {final_range}/{self.config.range_fft_size}")
        print(f"    Doppler bin: {final_doppler}/{self.config.doppler_fft_size}")
        print(f"    Range: {final_range_m:.2f} m")
        print(f"    Velocity: {final_velocity_ms:.2f} m/s")
        print(f"  ")
        print(f"  FINAL PEAK (physical):")
        print(f"    Mode: {self.config.radar_mode.upper()}  "
              f"(delay_factor={_delay_factor:.0f})")
        print(f"    Range: {final_range_m:.2f} m")
        print(f"    Velocity: {final_velocity_ms:.2f} m/s")
        print(f"    Note: GT matching done in _compute_metrics")
        print(f"  ")

        print(f"  STATUS: Peak at range bin {final_range} → {final_range_m:.2f} m  "
              f"(mode-aware, no hardcoded GT bin)")

        # Show top 10 peaks
        print(f"  ")
        print(f"  TOP 10 PEAKS (what CFAR sees):")
        flat_rdm = final_rdm.flatten()
        top10_indices = xp.argsort(flat_rdm)[-10:][::-1]
        for i, idx in enumerate(top10_indices, 1):
            idx = int(idx)
            r_bin, d_bin = np.unravel_index(idx, final_rdm.shape)
            power = float(flat_rdm[idx])
            beat = r_bin * (self.config.fmcw_sampling_rate_hz / self.config.range_fft_size)
            r_m = beat * LIGHTSPEED / (_delay_factor * chirp_slope)
            print(f"    {i}. Bin({r_bin:4d}, {d_bin:3d}) → {r_m:7.2f}m, power={power:.3e}")

        print(f"{'='*80}\n")
        # ═══════════════════════════════════════════════════════════════════════════

        return range_doppler_map, virtual_array_4d
    
    def _get_window(
        self,
        length: int,
        window_type: str
    ) -> Union[np.ndarray, 'cp.ndarray']:
        
        xp = self.xp
        
        if window_type == 'hann':
            window = xp.hanning(length) if hasattr(xp, 'hanning') else xp.asarray(np.hanning(length))
        elif window_type == 'hamming':
            window = xp.hamming(length) if hasattr(xp, 'hamming') else xp.asarray(np.hamming(length))
        elif window_type == 'blackman':
            window = xp.blackman(length) if hasattr(xp, 'blackman') else xp.asarray(np.blackman(length))
        elif window_type == 'kaiser':
            beta = self.config.kaiser_beta_range
            window = xp.asarray(np.kaiser(length, beta))
        else:
            window = xp.ones(length)
        
        return window
    
    def _mitigate_ofdm_interference_nlms(
        self,
        received_signal: Union[np.ndarray, 'cp.ndarray'],
        tx_comm_signal: Union[np.ndarray, 'cp.ndarray']
    ) -> Union[np.ndarray, 'cp.ndarray']:

        xp = self.xp
    
        # ========== INPUT VALIDATION & BROADCASTING ==========
    
        print(f"\n{'='*80}")
        print(f"[NLMS OFDM MITIGATION - DETAILED DIAGNOSTICS]")
        print(f"{'='*80}")
    
        # Ensure tx_comm_signal is on the same device
        if self.use_gpu and isinstance(tx_comm_signal, np.ndarray):
            print(f"  Converting tx_comm_signal from NumPy to CuPy...")
            tx_comm_signal = cp.asarray(tx_comm_signal)
        elif not self.use_gpu and hasattr(tx_comm_signal, 'get'):
            print(f"  Converting tx_comm_signal from CuPy to NumPy...")
            tx_comm_signal = tx_comm_signal.get()
    
        # Get dimensions
        n_rx, n_tx, n_samples, n_chirps = received_signal.shape
    
        print(f"\n[STEP 1: INPUT VALIDATION]")
        print(f"  received_signal shape: {received_signal.shape}")
        print(f"  received_signal dtype: {received_signal.dtype}")
        print(f"  tx_comm_signal shape (original): {tx_comm_signal.shape}")
        print(f"  tx_comm_signal dtype: {tx_comm_signal.dtype}")
    
        # Compute initial statistics
        rx_power_initial = float(xp.mean(xp.abs(received_signal)**2))
        tx_comm_power_initial = float(xp.mean(xp.abs(tx_comm_signal)**2))
    
        print(f"\n  Initial Power Levels:")
        print(f"    RX signal power: {rx_power_initial:.6e}")
        print(f"    TX OFDM power: {tx_comm_power_initial:.6e}")
        print(f"    Power ratio (RX/TX): {rx_power_initial/tx_comm_power_initial:.4f}")
    
        # Broadcast tx_comm_signal to (n_rx, n_tx, n_samples, n_chirps) if needed
        if tx_comm_signal.ndim == 2:
            # Shape: (n_samples, n_chirps) → Single TX, broadcast to all
            print(f"\n  Broadcasting tx_comm_signal from 2D to 4D...")
            tx_comm_signal = tx_comm_signal[None, None, :, :]
            tx_comm_signal = xp.broadcast_to(tx_comm_signal, (n_rx, n_tx, n_samples, n_chirps))
            print(f"    After broadcast: {tx_comm_signal.shape}")
    
        elif tx_comm_signal.ndim == 3:
            # Shape: (n_tx, n_samples, n_chirps) → Per-TX waveform
            print(f"\n  Broadcasting tx_comm_signal from 3D to 4D...")
            tx_comm_signal = tx_comm_signal[None, :, :, :]
            tx_comm_signal = xp.broadcast_to(tx_comm_signal, (n_rx, n_tx, n_samples, n_chirps))
            print(f"    After broadcast: {tx_comm_signal.shape}")
    
        elif tx_comm_signal.ndim == 4:
            # Shape: (n_rx, n_tx, n_samples, n_chirps) → Per-channel references (OPTIMAL!)
            if tx_comm_signal.shape != received_signal.shape:
                raise ValueError(
                    f"tx_comm_signal shape {tx_comm_signal.shape} does not match "
                    f"received_signal shape {received_signal.shape}"
                )
            print(f"\n  ✅ Using 4D per-channel OFDM reference (optimal for correlation)")
            print(f"     Each RX-TX pair has its own matched interference pattern")
            print(f"     This preserves phase/amplitude differences across channels")
            print(f"     Expected correlation: 40-80% (vs 1-4% with averaged reference)")
        else:
            raise ValueError(
                f"tx_comm_signal has invalid shape: {tx_comm_signal.shape}. "
                f"Expected 2D, 3D, or 4D array."
            )
    
        # ========== NLMS FILTERING PER CHANNEL ==========
    
        M = self.nlms_filter_length  # Filter length (number of taps)
        mu = self.nlms_step_size     # Step size (learning rate)
    
        cleaned_signal = xp.zeros_like(received_signal, dtype=complex)
    
        print(f"\n{'─'*80}")
        print(f"[STEP 2: NLMS ALGORITHM SETUP]")
        print(f"{'─'*80}")
        print(f"  Algorithm: Normalized Least-Mean-Squares (NLMS)")
        print(f"  Filter length M: {M} taps")
        print(f"  Step size μ: {mu}")
        print(f"  Total channels to process: {n_rx} RX × {n_tx} TX = {n_rx*n_tx}")
        print(f"  Samples per channel: {n_samples} × {n_chirps} = {n_samples*n_chirps}")
        print(f"  Total iterations per channel: {n_samples*n_chirps - M}")
    
        # Statistics tracking
        convergence_stats = []
    
        # Process each RX-TX channel independently
        for rx in range(n_rx):
            for tx in range(n_tx):
                channel_idx = rx * n_tx + tx
            
                if self.verbose:
                    print(f"\n{'─'*80}")
                    print(f"[PROCESSING CHANNEL {channel_idx+1}/{n_rx*n_tx}] RX={rx}, TX={tx}")
                    print(f"{'─'*80}")
            
                # Flatten to 1D — explicit copy prevents issues with
                # broadcast views (xp.broadcast_to returns read-only arrays
                # whose .flatten() may not copy data on all NumPy versions)
                d = np.array(received_signal[rx, tx], dtype=np.complex128).flatten()
                u = np.array(tx_comm_signal[rx, tx], dtype=np.complex128).flatten()
            
                n_total = len(d)
            
                if self.verbose:
                    d_power = float(xp.mean(xp.abs(d)**2))
                    u_power = float(xp.mean(xp.abs(u)**2))
                    print(f"  Flattened to 1D: {n_total} samples")
                    print(f"  Desired signal power: {d_power:.6e}")
                    print(f"  Reference signal power: {u_power:.6e}")
                    print(f"  Power ratio (d/u): {d_power/u_power:.4f}")
            
                # Initialize filter weights
                w = xp.zeros(M, dtype=complex)

                # Error signal (will contain cleaned signal after convergence)
                e = xp.zeros_like(d, dtype=complex)

                # Copy first M samples (before filter has warmed up)
                e[:M] = d[:M]

                # ── Per-channel pre-convergence correlation ───────────────
                # Measure how correlated this channel's reference is with
                # the received signal BEFORE NLMS — this is the key diagnostic.
                n_check = min(3000, n_total)
                corr_ch = float(xp.abs(xp.dot(u[:n_check].conj(), d[:n_check])) /
                                (xp.sqrt(xp.dot(u[:n_check].conj(), u[:n_check])) *
                                 xp.sqrt(xp.dot(d[:n_check].conj(), d[:n_check])) + 1e-12))
                
                if self.verbose:
                    print(f"  Pre-NLMS correlation (ch {channel_idx+1}): "
                          f"{corr_ch:.4f} ({corr_ch*100:.1f}%)")
                    if corr_ch > 0.3:
                        print(f"    ✅ Good — NLMS will converge effectively")
                    elif corr_ch > 0.2:
                        print(f"    ⚠️  Moderate — partial cancellation expected")
                    elif corr_ch > 0.1:
                        print(f"    ⚠️  LOW WARNING (< 20%) — weak NLMS convergence likely")
                        print(f"       Check: is 4D rx_ofdm being passed (not averaged)?")
                    else:
                        print(f"    ❌ VERY LOW (< 10%) — NLMS will not converge")
                        print(f"       Check: tx_comm_signal shape, tiling, power scaling")
                        
                # d and u are already np.complex128 from the explicit copy above.
                # ascontiguousarray is a no-op here but kept for safety.
                d_np = np.ascontiguousarray(d, dtype=np.complex128)
                u_np = np.ascontiguousarray(u, dtype=np.complex128)

                # ── NaN guard: detect poisoned input before NLMS starts ──
                d_nan = int(np.sum(~np.isfinite(d_np)))
                u_nan = int(np.sum(~np.isfinite(u_np)))
                if d_nan > 0 or u_nan > 0:
                    print(f"  ⚠️  NaN/Inf detected BEFORE NLMS: "
                          f"d_nan={d_nan}, u_nan={u_nan} — replacing with 0")
                    d_np = np.where(np.isfinite(d_np), d_np, 0.0 + 0.0j)
                    u_np = np.where(np.isfinite(u_np), u_np, 0.0 + 0.0j)

                n_total_ch = len(d_np)
                w_np = np.zeros(M, dtype=np.complex128)
                e_np = np.zeros(n_total_ch, dtype=np.complex128)

                # Pre-filter samples (filter not yet warmed up for chirp 0)
                e_np[:M] = d_np[:M]

                # Stability constants
                # W_MAX: steady-state ||w||² ≈ 1.0 at convergence for M=32,
                # μ=0.01, P_u≈0.8. 1e4 gives 10000× headroom.
                # E_MAX: OFDM PAPR ≈ 9 dB → peak amplitude ≈ 9× RMS ≈ 9.
                # 1e4 gives safe headroom without false triggering.
                W_MAX = 1e10
                E_MAX = 1e3

                # Per-chirp divergence counting.
                # KEY FIX: The old single `diverged` boolean was set True on
                # the very first weight reset (chirp 0 warm-up transient) and
                # NEVER reset between chirps. This caused all 128 chirps to be
                # discarded even though chirps 1-127 converged correctly.
                # Fix: count diverged chirps per channel. Only trigger full
                # pass-through if >50% of chirps were unstable.
                diverged_chirps = 0

                chirp_error_powers = []

                d_2d = d_np.reshape(n_samples, n_chirps)   # (1664, 128)
                u_2d = u_np.reshape(n_samples, n_chirps)   # (1664, 128)
                e_2d = np.zeros((n_samples, n_chirps), dtype=np.complex128)

                for chirp_idx in range(n_chirps):
                    d_ch = np.ascontiguousarray(d_2d[:, chirp_idx])
                    u_ch = np.ascontiguousarray(u_2d[:, chirp_idx])
                    e_ch = np.zeros(n_samples, dtype=np.complex128)

                    # ── Per-chirp weight reset ────────────────────────────
                    # The OFDM slot (43008 samples) is shorter than the radar
                    # frame (212992 samples) so the reference u is TILED.
                    # At every tile boundary (~every 26 chirps) the phase of
                    # u_ch jumps discontinuously.  Weights trained on chirp k
                    # predict chirp k+1 poorly at those boundaries:
                    #   y_n = w.conj() @ u_vec  →  large spike  →  |e_val| > E_MAX
                    # This triggers the divergence guard and sets chirp_diverged=True
                    # for ~90% of chirps, forcing full pass-through.
                    #
                    # Fix: reset w_np to zero at the start of every chirp.
                    # Cost: M-sample warm-up overhead per chirp (32/1664 = 1.9%).
                    # Benefit: eliminates all inter-chirp divergence for both
                    #          monostatic (tiling factor ~5) and bistatic (same).
                    #
                    # This is valid because FMCW processing is chirp-by-chirp:
                    # each chirp is an independent slow-time snapshot, so there
                    # is no coherence benefit in carrying weights across chirps
                    # when the reference phase is discontinuous at tile edges.
                    w_np[:] = 0.0    # ← reset weights every chirp

                    # First M samples: pass-through (filter warming up)
                    e_ch[:M] = d_ch[:M]

                    # Per-chirp flag
                    chirp_diverged = False

                    for n in range(M, n_samples):
                        u_vec = u_ch[n-M:n][::-1].copy()

                        y_n   = np.dot(w_np.conj(), u_vec)
                        e_val = d_ch[n] - y_n

                        # Error guard — do NOT reset weights on bad sample.
                        # Resetting w mid-chirp causes transient overshoot
                        # that amplifies the signal instead of cancelling it.
                        # Instead: freeze weights, output pass-through for
                        # this one sample, and continue with frozen weights.
                        if not np.isfinite(e_val) or abs(e_val) > E_MAX:
                            e_ch[n] = d_ch[n]   # pass-through this sample
                            chirp_diverged = True
                            continue            # skip weight update, keep w

                        e_ch[n] = e_val

                        norm   = float(np.real(np.dot(u_vec.conj(), u_vec))) + 1e-10
                        w_np  += (mu / norm) * e_val * u_vec.conj()

                        # Weight guard — only reset on true NaN/Inf.
                        # Do NOT reset on overflow alone: an overshooting
                        # weight that decays is less harmful than a cold
                        # restart which causes constructive interference.
                        w_norm = float(np.real(np.dot(w_np.conj(), w_np)))
                        if not np.isfinite(w_norm):
                            w_np[:] = 0.0
                            chirp_diverged = True

                    if chirp_diverged:
                        diverged_chirps += 1

                    e_2d[:, chirp_idx] = e_ch

                    # Per-chirp convergence tracking (every 16 chirps)
                    if self.verbose and (chirp_idx % 16 == 0
                                        or chirp_idx == n_chirps - 1):
                        ep = float(np.mean(np.abs(e_ch)**2))
                        dp = float(np.mean(np.abs(d_ch)**2))
                        chirp_error_powers.append((chirp_idx, ep, dp))

                # Decision: pass-through only if majority of chirps failed
                divergence_rate = diverged_chirps / n_chirps
                channel_diverged = divergence_rate > 0.5

                if diverged_chirps > 0:
                    status = "❌ pass-through" if channel_diverged else "⚠️  partial resets kept"
                    print(f"    Ch {channel_idx+1}: {diverged_chirps}/{n_chirps} chirps "
                          f"had weight resets — {status} "
                          f"(rate={divergence_rate*100:.1f}%)")

                if channel_diverged:
                    e_2d = d_2d.copy()

                e_np = e_2d.flatten()

                if self.verbose and chirp_error_powers:
                    print(f"\n  Per-chirp convergence (ch {channel_idx+1}):")
                    for ci, ep, dp in chirp_error_powers:
                        ratio = ep / (dp + 1e-12)
                        print(f"    Chirp {ci:3d}/{n_chirps}: "
                              f"e_power={ep:.4e}, "
                              f"d_power={dp:.4e}, "
                              f"ratio={ratio:.4f}")
                    if len(chirp_error_powers) >= 2:
                        r0 = chirp_error_powers[0][1] / (chirp_error_powers[0][2] + 1e-12)
                        r1 = chirp_error_powers[-1][1] / (chirp_error_powers[-1][2] + 1e-12)
                        improvement_db = 10 * np.log10((r0 + 1e-12) / (r1 + 1e-12))
                        print(f"    Convergence improvement: "
                              f"{improvement_db:.2f} dB (chirp 0 → last)")

                # Back to original device if GPU
                if self.use_gpu:
                    e = cp.asarray(e_np)
                else:
                    e = e_np

                # Reshape back to 2D (n_samples, n_chirps)
                cleaned_signal[rx, tx] = e.reshape(n_samples, n_chirps)
            
                # Channel-specific statistics
                if self.verbose:
                    d_power_final = float(xp.mean(xp.abs(d)**2))
                    e_power_final = float(xp.mean(xp.abs(e)**2))
                    residual = d - e
                    residual_power = float(xp.mean(xp.abs(residual)**2))
                
                    cancellation_ratio = residual_power / d_power_final
                    cancellation_db = 10 * np.log10(d_power_final / (residual_power + 1e-12))
                
                    print(f"\n  Channel {channel_idx+1} Results:")
                    print(f"    Original signal power: {d_power_final:.6e}")
                    print(f"    Cleaned signal power: {e_power_final:.6e}")
                    print(f"    Cancelled interference power: {residual_power:.6e}")
                    print(f"    Cancellation ratio: {cancellation_ratio*100:.2f}%")
                    print(f"    Cancellation gain: {cancellation_db:.2f} dB")
                
                    # Convergence quality from per-chirp tracking
                    if len(chirp_error_powers) >= 2:
                        r0 = chirp_error_powers[0][1] / (chirp_error_powers[0][2] + 1e-12)
                        r1 = chirp_error_powers[-1][1] / (chirp_error_powers[-1][2] + 1e-12)
                        conv_db = 10 * np.log10((r0 + 1e-12) / (r1 + 1e-12))
                        if conv_db > 1.0:
                            print(f"    Convergence: ✅ GOOD ({conv_db:.1f} dB improvement)")
                        elif conv_db > 0.0:
                            print(f"    Convergence: ⚠️  OK ({conv_db:.1f} dB improvement)")
                        else:
                            print(f"    Convergence: ❌ POOR ({conv_db:.1f} dB)")

                    convergence_stats.append({
                        'rx': rx, 'tx': tx,
                        'cancellation_db': cancellation_db,
                        'cancellation_ratio': cancellation_ratio,
                        'weight_norm': float(np.real(np.dot(w_np.conj(), w_np)))
                    })
    
        # ========== FINAL STATISTICS ==========
    
        print(f"\n{'='*80}")
        print(f"[STEP 3: OVERALL MITIGATION PERFORMANCE]")
        print(f"{'='*80}")
    
        # Compute global mitigation effectiveness
        orig_power = float(xp.mean(xp.abs(received_signal)**2))
        cleaned_power = float(xp.mean(xp.abs(cleaned_signal)**2))
        residual_power = float(xp.mean(xp.abs(received_signal - cleaned_signal)**2))
    
        mitigation_db = 10 * np.log10(orig_power / (cleaned_power + 1e-12))
        residual_ratio = residual_power / (orig_power + 1e-12)
    
        print(f"\n  Global Statistics (all channels averaged):")
        print(f"    Original RX power: {orig_power:.6e}")
        print(f"    Cleaned RX power: {cleaned_power:.6e}")
        print(f"    Residual (cancelled) power: {residual_power:.6e}")
        print(f"    Residual ratio: {residual_ratio*100:.2f}%")
        print(f"    Mitigation gain: {mitigation_db:.2f} dB")
    
        # Per-channel statistics
        if convergence_stats:
            avg_cancellation_db = np.mean([s['cancellation_db'] for s in convergence_stats])
            min_cancellation_db = np.min([s['cancellation_db'] for s in convergence_stats])
            max_cancellation_db = np.max([s['cancellation_db'] for s in convergence_stats])
        
            print(f"\n  Per-Channel Cancellation Performance:")
            print(f"    Average: {avg_cancellation_db:.2f} dB")
            print(f"    Min: {min_cancellation_db:.2f} dB")
            print(f"    Max: {max_cancellation_db:.2f} dB")
            print(f"    Std: {np.std([s['cancellation_db'] for s in convergence_stats]):.2f} dB")
    
        # ========== QUALITY CHECKS ==========
    
        print(f"\n{'─'*80}")
        print(f"[STEP 4: QUALITY ASSESSMENT]")
        print(f"{'─'*80}")
    
        # Check 1: Power preservation
        power_ratio = cleaned_power / orig_power
        print(f"\n  Check 1: Power Preservation")
        print(f"    Power ratio (cleaned/original): {power_ratio:.4f}")
        if 0.3 < power_ratio < 0.9:
            print(f"    Status: ✅ GOOD - OFDM interference removed, radar echo preserved")
        elif power_ratio >= 0.9:
            print(f"    Status: ⚠️  WARNING - Very little interference removed (ratio too high)")
            print(f"    Possible issues:")
            print(f"      - OFDM waveform may not match actual interference")
            print(f"      - Step size μ too small (current: {mu})")
            print(f"      - Filter length M too short (current: {M})")
        else:
            print(f"    Status: ❌ ERROR - Too much signal removed (ratio too low)")
            print(f"    Possible issues:")
            print(f"      - NLMS removing radar echo along with OFDM")
            print(f"      - Step size μ too large (current: {mu})")
    
        # Check 2: Cancellation effectiveness
        print(f"\n  Check 2: Cancellation Effectiveness")
        if residual_ratio <= 0.0:
            print(f"    Cancellation: N/A (pass-through — all channels diverged)")
            print(f"    Status: ❌ POOR - No cancellation")
        else:
            expected_cancellation_db = 10 * np.log10(1 / residual_ratio)
            print(f"    Cancellation: {expected_cancellation_db:.2f} dB")
            if expected_cancellation_db > 8:
                print(f"    Status: ✅ EXCELLENT - Strong suppression (>8 dB)")
            elif expected_cancellation_db > 4:
                print(f"    Status: ✅ GOOD - Moderate suppression (4-8 dB)")
            elif expected_cancellation_db > 1.5:
                print(f"    Status: ⚠️  MARGINAL - Weak suppression (1.5-4 dB)")
                print(f"    Note: Tiling discontinuities limit theoretical max to ~3 dB")
            else:
                print(f"    Status: ❌ POOR - Below 1.5 dB suppression")
                print(f"    Check: Is tx_ofdm_td the 4D rx_ofdm array (not the mean)?")
                print(f"    Recommendations:")
                print(f"      - Increase filter length M (try {M*2})")
                print(f"      - Adjust step size μ (try {mu*2:.4f} or {mu/2:.4f})")
                print(f"      - Verify OFDM waveform matches transmitted signal")
    
        # Check 3: Signal characteristics
        print(f"\n  Check 3: Signal Characteristics")
    
        # Compute correlation between reference and received
        correlation_samples = min(1000, n_samples * n_chirps)
        u_sample = tx_comm_signal[0, 0].flatten()[:correlation_samples]
        d_sample = received_signal[0, 0].flatten()[:correlation_samples]
    
        correlation = float(xp.abs(xp.dot(u_sample.conj(), d_sample)) / 
                            (xp.sqrt(xp.dot(u_sample.conj(), u_sample)) * 
                            xp.sqrt(xp.dot(d_sample.conj(), d_sample))))
    
        print(f"    Global correlation (TX OFDM vs RX[0,0]): {correlation:.4f} "
              f"({correlation*100:.1f}%)")
        if correlation > 0.3:
            print(f"    Status: ✅ GOOD — strong correlation, NLMS effective")
        elif correlation > 0.2:
            print(f"    Status: ⚠️  MODERATE — partial NLMS benefit expected")
        elif correlation > 0.1:
            print(f"    Status: ⚠️  LOW (<20%) — weak NLMS convergence")
            print(f"      Check: pass 4D rx_ofdm (not mean) as tx_comm_signal")
            print(f"      Check: OFDM tiling introduces discontinuities → limits corr to ~30%")
        else:
            print(f"    Status: ❌ VERY LOW (<10%) — NLMS will not converge")
            print(f"      Check: tx_comm_signal waveform, timing, power scaling")
    
        print(f"\n{'='*80}")
        print(f"[NLMS MITIGATION COMPLETE]")
        print(f"{'='*80}\n")
    
        return cleaned_signal
    
    def compute_false_alarm_probability(
        self,
        detections: np.ndarray,
        targets: List[Dict],
        ground_truth: Union[Dict, List[Dict]],
        range_threshold_m: float = 6.3,
        velocity_threshold_ms: float = 0.70
    ) -> Tuple[float, int]:
    
        if len(targets) == 0:
            return 0.0, 0
    
        if isinstance(ground_truth, dict):
            gt_list = [ground_truth]
        elif isinstance(ground_truth, list):
            gt_list = ground_truth
        else:
            gt_list = []
    
        if len(gt_list) == 0:
            return 0.0, 0
    
        n_false_alarms = 0
    
        for target in targets:
            r_det = target['range_m']
            v_det = target['velocity_ms']
        
            matched = False
            for gt in gt_list:
                r_err = abs(r_det - gt['range_m'])
                v_err = abs(v_det - gt.get('velocity_ms', 0.0))
            
                if r_err < range_threshold_m and v_err < velocity_threshold_ms:
                    matched = True
                    break
        
            if not matched:
                n_false_alarms += 1
    
        p_fa = n_false_alarms / len(targets) if len(targets) > 0 else 0.0
    
        return p_fa, n_false_alarms
    
    def compute_scnr(
        self,
        range_doppler_map: np.ndarray,
        targets: List[Dict],
        clutter_guard_cells: int = 5,
        thermal_noise_power: Optional[float] = None
    ) -> Tuple[float, float]:
        
        if len(targets) == 0:
            return None, None

        # Power map — guard against double-squaring real-valued maps
        if np.iscomplexobj(range_doppler_map):
            power_map = np.abs(range_doppler_map) ** 2
        else:
            power_map = np.asarray(range_doppler_map, dtype=np.float64)

        n_range, n_doppler = power_map.shape
        target    = targets[0]
        range_idx  = target['range_idx']
        doppler_idx = target['doppler_idx']

        # ── Signal power at peak ─────────────────────────────────────────
        signal_power = float(power_map[range_idx, doppler_idx])

        # ── Noise floor (thermal model) ──────────────────────────────────
        _tnp = thermal_noise_power if thermal_noise_power is not None \
               else self._thermal_noise_power

        if _tnp is not None:
            n_tx      = self.config.n_radar_tx_antennas
            n_rx      = self.config.n_radar_rx_antennas
            n_samples = self.config.fmcw_n_samples_per_chirp
            n_chirps  = self.config.fmcw_n_chirps
            hann      = 0.375
            noise_floor = (
                _tnp
                * n_tx * n_rx
                * n_samples * hann
                * n_chirps  * hann
            )
        else:
            # Fallback: very low percentile far from any target
            flat = power_map.flatten()
            noise_floor = float(np.percentile(flat, 0.5))

        # ── Background power (clutter + noise) ──────────────────────────
        # Build a mask that excludes guard zones around ALL detected targets.
        # Cells outside guard zones contain background = OFDM clutter + noise.
        # We use a large guard (3× clutter_guard_cells) to avoid sidelobe
        # contamination in the background estimate.
        clutter_mask = np.ones((n_range, n_doppler), dtype=bool)
        wide_guard   = clutter_guard_cells * 3

        for t in targets[:20]:   # guard around top 20 detections
            r = t.get('range_idx', 0)
            d = t.get('doppler_idx', 0)
            r0 = max(0, r - wide_guard)
            r1 = min(n_range,  r + wide_guard + 1)
            d0 = max(0, d - wide_guard)
            d1 = min(n_doppler, d + wide_guard + 1)
            clutter_mask[r0:r1, d0:d1] = False

        background_cells = power_map[clutter_mask]

        # ── Clutter power: analytical model (preferred) ──────────────────
        # In a dense multi-target scene, far-field RDM cells are contaminated
        # by target sidelobes, making background estimation unreliable.
        # Analytical model: OFDM clutter power in RDM = noise_floor × ISR
        # where ISR = α²/β² (interference-to-signal ratio).
        # At equal power split (α²=β²=0.5): ISR = 1.0 (0 dB)
        # → SCNR = SNR_RDM - 3 dB (clutter doubles the noise floor)
        beta_cfg  = getattr(self.config, 'radar_power_factor',  float(np.sqrt(0.5)))
        alpha_cfg = getattr(self.config, 'comm_power_factor',   float(np.sqrt(0.5)))
        beta_sq   = float(beta_cfg  ** 2)
        alpha_sq  = float(alpha_cfg ** 2)
        isr       = alpha_sq / (beta_sq + 1e-12)

        if _tnp is not None:
            # Analytical: OFDM clutter = noise_floor × ISR
            clutter_power    = noise_floor * isr
            clutter_method   = f"analytical (ISR = α²/β² = {isr:.2f})"
        else:
            # Fallback: estimate from background cells
            if len(background_cells) > 1000:
                background_power = float(np.mean(background_cells))
                clutter_power    = max(background_power - noise_floor, 0.0)
            else:
                clutter_power    = 0.0
            clutter_method = "background estimation (fallback)"

        # ── SCNR = S / (C + N) ───────────────────────────────────────────
        cn_total    = clutter_power + noise_floor
        scnr_linear = signal_power / (cn_total + 1e-12)
        scnr_db     = float(10 * np.log10(scnr_linear + 1e-12))
        scr_db      = float(10 * np.log10(signal_power / (clutter_power + 1e-12)))
        snr_rdm_db  = float(10 * np.log10(signal_power / (noise_floor + 1e-12)))

        if self.verbose:
            print(f"\n  [SCNR COMPUTATION]")
            print(f"    Signal power (peak):         {signal_power:.3e}")
            print(f"    Noise floor (thermal RDM):   {noise_floor:.3e}  "
                  f"({snr_rdm_db:.1f} dB below signal)")
            print(f"    Clutter method:              {clutter_method}")
            print(f"    α² (COMM power fraction):    {alpha_sq:.4f}")
            print(f"    β² (radar power fraction):   {beta_sq:.4f}")
            print(f"    ISR (α²/β²):                 {isr:.4f}  "
                  f"({10*np.log10(isr+1e-12):.1f} dB)")
            print(f"    OFDM clutter power:          {clutter_power:.3e}  "
                  f"({10*np.log10(clutter_power+1e-12):.1f} dB)")
            print(f"    C+N total:                   {cn_total:.3e}  "
                  f"({10*np.log10(cn_total+1e-12):.1f} dB)")
            print(f"    Background cells available:  {len(background_cells)}")
            print(f"    ─────────────────────────────────────────────────")
            print(f"    SNR_rdm (S/N):               {snr_rdm_db:.2f} dB")
            print(f"    SCR     (S/C):               {scr_db:.2f} dB")
            print(f"    SCNR    (S/(C+N)):            {scnr_db:.2f} dB")
            if abs(isr - 1.0) < 0.01:
                print(f"    ✅ Equal power split → SCNR ≈ SNR_rdm − 3 dB  "
                      f"(expected: {snr_rdm_db - 3.01:.1f} dB)")

        return scnr_db, float(scnr_linear)
    
    def _compute_metrics(
        self,
        range_doppler_map: np.ndarray,
        targets: List[Dict],
        ground_truth: Optional[Union[Dict, List[Dict]]],
        detection_match_threshold_range_m: float,
        detection_match_threshold_velocity_ms: float
    ) -> Dict:
        metrics = {}

        # Normalize ground_truth to list format
        if ground_truth is not None:
            if isinstance(ground_truth, dict):
                gt_list = [ground_truth]
            elif isinstance(ground_truth, list):
                gt_list = ground_truth
            else:
                gt_list = []
        else:
            gt_list = []

        # ── FoV gate: keep only GT targets visible to the RX array ───────
        # Array boresight: az=330°, el=-10° (from DeepVerse-6G config).
        # FoV = ±90° azimuth around boresight → local_az in (-90°, +90°).
        # Targets outside this window produce zero array response and must
        # be excluded from detection matching, detection rate, and coverage
        # estimates — they can never appear in the RDM regardless of SNR.
        # If doa fields are missing → conservative: keep target in evaluation.
        gt_fov_visible = []
        gt_fov_invisible = []
        for gt in gt_list:
            phi   = gt.get('doa_phi_deg',   None)
            theta = gt.get('doa_theta_deg', None)
            if phi is None or theta is None:
                # No angle info — conservative: keep in evaluation
                gt_fov_visible.append(gt)
                continue
            try:
                from veisac.rx.estimation import \
                    _global_doa_to_local_az
                local_az = _global_doa_to_local_az(float(phi), float(theta))
                if abs(local_az) <= 90.0:
                    gt_fov_visible.append(gt)
                else:
                    gt_fov_invisible.append(gt)
            except Exception:
                # Conversion failed — conservative: keep
                gt_fov_visible.append(gt)

        gt_for_evaluation = gt_fov_visible

        if self.verbose and len(gt_list) > 0:
            print(f"\n{'─'*80}")
            print(f"[GROUND TRUTH TARGETS FOR EVALUATION]")
            print(f"{'─'*80}")
            print(f"  Total GT targets (scene):      {len(gt_list)}")
            print(f"  GT targets in FoV (visible):   {len(gt_fov_visible)}")
            print(f"  GT targets outside FoV:        {len(gt_fov_invisible)}  "
                  f"← excluded from matching")
            print(f"  FoV gate: local azimuth ≤ ±90°  "
                  f"(boresight az=330°, el=-10°)")
            if gt_fov_invisible:
                print(f"  Excluded targets:")
                for gt in gt_fov_invisible[:3]:
                    phi   = gt.get('doa_phi_deg',   float('nan'))
                    theta = gt.get('doa_theta_deg', float('nan'))
                    try:
                        from veisac.rx.estimation import \
                            _global_doa_to_local_az
                        laz = _global_doa_to_local_az(float(phi), float(theta))
                    except Exception:
                        laz = float('nan')
                    print(f"    R={gt.get('range_m',0.0):.1f}m  "
                          f"φ={phi:.1f}°  θ={theta:.1f}°  "
                          f"local_az={laz:.1f}°")
                if len(gt_fov_invisible) > 3:
                    print(f"    ... and {len(gt_fov_invisible)-3} more")
            print(f"\n  GT targets (FoV-gated, used for evaluation):")
            for i, gt in enumerate(gt_fov_visible[:5], 1):
                phi   = gt.get('doa_phi_deg',   float('nan'))
                theta = gt.get('doa_theta_deg', float('nan'))
                try:
                    from veisac.rx.estimation import \
                        _global_doa_to_local_az
                    laz = _global_doa_to_local_az(float(phi), float(theta))
                    laz_str = f"local_az={laz:.1f}°"
                except Exception:
                    laz_str = "local_az=N/A"
                print(f"    {i}. R={gt.get('range_m', 0.0):.2f}m  "
                      f"V={gt.get('velocity_ms', 0.0):+.2f}m/s  {laz_str}")
            if len(gt_fov_visible) > 5:
                print(f"    ... and {len(gt_fov_visible) - 5} more")
            print(f"{'─'*80}\n")

        metrics['n_total_gt']          = len(gt_list)
        metrics['n_gt_fov_visible']    = len(gt_fov_visible)
        metrics['n_gt_fov_invisible']  = len(gt_fov_invisible)

        # Compute power map — guard against double-squaring real-valued power maps
        if np.iscomplexobj(range_doppler_map):
            power_map = np.abs(range_doppler_map) ** 2
        else:
            power_map = np.asarray(range_doppler_map, dtype=np.float64)

        signal_power = float(np.max(power_map))

        # ═══════════════════════════════════════════════════════════════════
        # ADAPTIVE SNR COMPUTATION — THREE-TIER NOISE ESTIMATION
        # ═══════════════════════════════════════════════════════════════════
        # ARCHITECTURE NOTE:
        #   n_detections = len(targets) uses the EXTRACTED list (capped at 100),
        #   not the raw CFAR count (13,954). This makes scene_density artificially
        #   low (100/524288 = 0.019%) → the old density check always fell into the
        #   sparse branch → Tier 1 thermal model was never reached.
        #   Fix: Thermal model is checked FIRST, before any density branching.
        # ═══════════════════════════════════════════════════════════════════
        n_range, n_doppler = power_map.shape
        n_total_cells = n_range * n_doppler
        n_detections = len(targets)
        scene_density = n_detections / n_total_cells if n_total_cells > 0 else 0

        if self.verbose:
            print(f"  [SNR COMPUTATION - ADAPTIVE]")
            print(f"    Detections: {n_detections}")
            print(f"    Total cells: {n_total_cells}")
            print(f"    Density: {scene_density*100:.2f}%")
            print(f"    Thermal noise available: {self._thermal_noise_power is not None}")

        # ── TIER 1: Thermal model — always first, scene-density independent ──
        if self._thermal_noise_power is not None:
            n_tx = self.config.n_radar_tx_antennas
            n_rx = self.config.n_radar_rx_antennas

            # Array combining noise gain:
            #   Coherent TX sum:     noise power × N_TX (incoherent accumulation)
            #   Non-coherent RX sum: noise power × N_RX (power summation)
            array_noise_gain = n_tx * n_rx  # = 16 for 4×4

            # FFT processing noise gain:
            #   numpy fft() does NOT normalize → per-bin noise power scales as N
            #   Zero-padding does NOT add real noise, so effective gain uses
            #   actual sample counts, not the (larger) FFT sizes:
            #     Range:   N_samples_per_chirp = 1664
            #     Doppler: N_chirps            = 128
            range_fft_noise_gain   = self.config.fmcw_n_samples_per_chirp  # 1664
            doppler_fft_noise_gain = self.config.fmcw_n_chirps              # 128

            # Hann window reduces noise power by its normalized power ≈ 3/8 = 0.375
            # (sum(w²)/N = 3/8 for a Hann window — applied to both dimensions)
            hann_power_loss = 0.375

            noise_power = (
                self._thermal_noise_power
                * array_noise_gain
                * range_fft_noise_gain   * hann_power_loss
                * doppler_fft_noise_gain * hann_power_loss
            )

            if self.verbose:
                print(f"    Mode: THERMAL MODEL (Tier 1) — overrides density check")
                print(f"      Per-channel thermal:              {self._thermal_noise_power:.6e}")
                print(f"      After array combining (×{array_noise_gain}):      "
                      f"{self._thermal_noise_power * array_noise_gain:.6e}")
                print(f"      After range FFT (×{range_fft_noise_gain}×{hann_power_loss}):  "
                      f"{self._thermal_noise_power * array_noise_gain * range_fft_noise_gain * hann_power_loss:.6e}")
                print(f"      After Doppler FFT (×{doppler_fft_noise_gain}×{hann_power_loss}): "
                      f"{noise_power:.6e} ({10*np.log10(noise_power + 1e-12):.1f} dB)")

        # ── TIER 2: Sparse guard-zone (few detections, thermal not available) ──
        elif n_detections <= 10 or scene_density < 0.01:
            if self.verbose:
                print(f"    Mode: SPARSE (guard zones)")

            noise_mask = np.ones_like(power_map, dtype=bool)
            n_guard_targets = min(len(targets), 8)
            guard_cells_range = 6
            guard_cells_doppler = 4

            for i in range(n_guard_targets):
                if i >= len(targets):
                    break
                r_idx = targets[i].get('range_idx', 0)
                d_idx = targets[i].get('doppler_idx', 0)
                r_start = max(0, r_idx - guard_cells_range)
                r_end = min(n_range, r_idx + guard_cells_range + 1)
                d_start = max(0, d_idx - guard_cells_doppler)
                d_end = min(n_doppler, d_idx + guard_cells_doppler + 1)
                noise_mask[r_start:r_end, d_start:d_end] = False

            noise_cells = power_map[noise_mask]
            if len(noise_cells) > 100:
                noise_power = float(np.median(noise_cells))
            else:
                powers_sorted = np.sort(power_map.flatten())
                noise_power = float(np.median(powers_sorted[:len(powers_sorted) // 4]))

            if self.verbose:
                print(f"    Clean cells: {len(noise_cells)}")

        # ── TIER 3: Dense scene — CFAR threshold then percentile fallback ──
        else:
            if self.verbose:
                print(f"    Mode: DENSE (CFAR-aware adaptive)")

            if hasattr(self, '_last_threshold_map') and self._last_threshold_map is not None:
                threshold_map = self._last_threshold_map
                if hasattr(threshold_map, 'get'):
                    threshold_map = threshold_map.get()

                # CFAR ran on dB-scale RDM → threshold is in dB → must convert to linear
                threshold_linear = 10 ** (threshold_map / 10.0)
                noise_mask = power_map < (threshold_linear * 0.05)
                n_noise_cells = int(np.sum(noise_mask))

                if self.verbose:
                    print(f"    CFAR THRESHOLD:")
                    print(f"      Threshold range: {float(np.min(threshold_map)):.1f}"
                        f" to {float(np.max(threshold_map)):.1f} dB")
                    print(f"      Clean cells (< threshold×0.05): {n_noise_cells}/{power_map.size}")

                if n_noise_cells > 100:
                    noise_power = float(np.median(power_map[noise_mask]))
                    noise_power = max(noise_power, 1e-4)
                    if self.verbose:
                        print(f"      Noise power: {noise_power:.6e} "
                            f"({10*np.log10(noise_power + 1e-12):.1f} dB)")
                        print(f"      Status: ✅ CFAR method succeeded")
                else:
                    if self.verbose:
                        print(f"      Status: ⚠️  Only {n_noise_cells} clean cells. Falling to percentile.")
                    powers_sorted = np.sort(power_map.flatten())
                    n_samples = len(powers_sorted)
                    if n_samples > 500000:
                        percentile_idx = max(1, n_samples // 20000)
                        percentile_name = "0.005th"
                    elif n_samples > 100000:
                        percentile_idx = max(1, n_samples // 10000)
                        percentile_name = "0.01th"
                    elif n_samples > 10000:
                        percentile_idx = max(1, n_samples // 2000)
                        percentile_name = "0.05th"
                    else:
                        percentile_idx = max(1, n_samples // 200)
                        percentile_name = "0.5th"
                    noise_power = max(float(powers_sorted[percentile_idx]), 1e-4)
                    if self.verbose:
                        print(f"      {percentile_name} percentile + hard floor: "
                            f"{noise_power:.6e} ({10*np.log10(noise_power + 1e-12):.1f} dB)")

            else:
                # No CFAR threshold map available — use percentile only
                powers_sorted = np.sort(power_map.flatten())
                n_samples = len(powers_sorted)
                if n_samples > 500000:
                    percentile_idx = max(1, n_samples // 20000)
                    percentile_name = "0.005th"
                elif n_samples > 100000:
                    percentile_idx = max(1, n_samples // 10000)
                    percentile_name = "0.01th"
                elif n_samples > 10000:
                    percentile_idx = max(1, n_samples // 2000)
                    percentile_name = "0.05th"
                else:
                    percentile_idx = max(1, n_samples // 200)
                    percentile_name = "0.5th"
                noise_power = max(float(powers_sorted[percentile_idx]), 1e-4)
                if self.verbose:
                    print(f"    PERCENTILE ONLY ({percentile_name}) + hard floor: "
                        f"{noise_power:.6e} ({10*np.log10(noise_power + 1e-12):.1f} dB)")

        # Hard floor — prevents absurd inflation regardless of which tier ran
        noise_power = max(noise_power, 1e-4)

        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear + 1e-12)

        if self.verbose:
            print(f"    Signal power (peak): {float(signal_power):.6e}")
            print(f"    Noise power estimate: {float(noise_power):.6e}")
            print(f"    SNR: {float(snr_db):.2f} dB")
            if float(snr_db) > 60:
                print(f"    ⚠️  SNR > 60 dB — verify thermal_noise_power is being passed")
            elif float(snr_db) > 40:
                print(f"    ⚠️  SNR > 40 dB — verify if realistic for your scenario")
            elif float(snr_db) < 5:
                print(f"    ⚠️  SNR < 5 dB — very noisy signal")
            else:
                print(f"    ✅ SNR in typical range (5-40 dB)")

        metrics['snr_db'] = float(snr_db)
        metrics['snr_linear'] = float(snr_linear)
        metrics['n_detections'] = len(targets)

        # Average target statistics + CRB from input-SNR-referenced estimator
        if len(targets) > 0:
            metrics['avg_range_m']     = float(np.mean([t['range_m'] for t in targets]))
            metrics['avg_velocity_ms'] = float(np.mean([t['velocity_ms'] for t in targets]))

            # ── radar_snr_rdm: post-processing SNR, used for CFAR / ranking ──
            metrics['avg_radar_snr_rdm_db'] = float(
                np.mean([t.get('radar_snr_rdm_db', t.get('snr_db', float('nan')))
                         for t in targets]))
            metrics['top_radar_snr_rdm_db'] = float(
                targets[0].get('radar_snr_rdm_db', targets[0].get('snr_db', float('nan'))))

            # ── radar_snr_input: pre-processing SNR, used for CRB ────────────
            metrics['avg_radar_snr_input_db'] = float(
                np.mean([t.get('radar_snr_input_db', t.get('snr_input_db', float('nan')))
                         for t in targets]))
            metrics['top_radar_snr_input_db'] = float(
                targets[0].get('radar_snr_input_db', targets[0].get('snr_input_db', float('nan'))))
            metrics['radar_snr_input_source'] = targets[0].get(
                'radar_snr_input_source', targets[0].get('snr_input_source', 'unknown'))

            # ── CRB lower bounds (referenced to radar_snr_input) ─────────────
            metrics['avg_crb_range_m']     = float(np.mean([t['crb_range_m'] for t in targets]))
            metrics['avg_crb_velocity_ms'] = float(np.mean([t['crb_velocity_ms'] for t in targets]))
            metrics['top_crb_range_m']     = float(targets[0]['crb_range_m'])
            metrics['top_crb_velocity_ms'] = float(targets[0]['crb_velocity_ms'])

            if self.verbose:
                print(f"    [METRICS SNR/CRB]")
                print(f"      top_radar_snr_rdm:   {metrics['top_radar_snr_rdm_db']:.2f} dB  "
                      f"← post-processing")
                print(f"      top_radar_snr_input: {metrics['top_radar_snr_input_db']:.2f} dB  "
                      f"← pre-processing (CRB ref)")
                print(f"      top_crb_range:       {metrics['top_crb_range_m']:.6f} m")
                print(f"      top_crb_velocity:    {metrics['top_crb_velocity_ms']:.8f} m/s")
                print(f"      SNR source:          {metrics['radar_snr_input_source']}")

        # ═══════════════════════════════════════════════════════════════════
        # GT PAIRING — GREEDY NEAREST-NEIGHBOUR MATCHING
        # ═══════════════════════════════════════════════════════════════════
        if len(gt_for_evaluation) > 0 and len(targets) > 0:

            if self.verbose:
                print(f"\n{'─'*80}")
                print(f"[GT PAIRING - ALL TARGETS]")
                print(f"{'─'*80}")
                print(f"  GT targets: {len(gt_for_evaluation)}")
                print(f"  Detected targets: {len(targets)}")
                print(f"  Pairing method: Nearest-neighbor (greedy)")

            matched_pairs = []
            used_dets = set()          # ← was used_gts; now tracks used detections

            # Direction reversed: iterate GT→detection (one best det per GT).
            # This guarantees exactly N pairs for N GT targets, identical to
            # the visualization logic, so range_error_m / velocity_error_ms
            # are consistent with what is shown on the RD map plot.
            for gt_idx, gt in enumerate(gt_for_evaluation):
                gt_range = gt['range_m']
                gt_vel   = gt.get('velocity_ms', 0.0)

                best_det_idx  = None
                best_distance = float('inf')

                for det_idx, det in enumerate(targets):
                    if det_idx in used_dets:
                        continue
                    r_err    = abs(det['range_m']    - gt_range)
                    v_err    = abs(det['velocity_ms'] - gt_vel)
                    distance = r_err + v_err * 10.0
                    if distance < best_distance:
                        best_distance = distance
                        best_det_idx  = det_idx

                if best_det_idx is not None:
                    det   = targets[best_det_idx]
                    r_err = abs(det['range_m']    - gt_range)
                    v_err = abs(det['velocity_ms'] - gt_vel)
                    if (r_err < detection_match_threshold_range_m and
                            v_err < detection_match_threshold_velocity_ms):
                        # Tuple layout (det_idx, gt_idx, r_err, v_err) unchanged
                        # so all downstream code that reads matched_pairs is intact
                        matched_pairs.append((best_det_idx, gt_idx, r_err, v_err))
                        used_dets.add(best_det_idx)

            if len(matched_pairs) > 0:
                range_errors = [pair[2] for pair in matched_pairs]
                velocity_errors = [pair[3] for pair in matched_pairs]

                metrics['range_error_m'] = float(np.mean(range_errors))
                metrics['range_error_median_m'] = float(np.median(range_errors))
                
                metrics['velocity_error_ms'] = float(np.mean(velocity_errors))
                metrics['velocity_error_median_ms'] = float(np.median(velocity_errors))
                
                metrics['range_rmse_m'] = float(np.sqrt(np.mean(np.array(range_errors)**2)))
                metrics['velocity_rmse_ms'] = float(np.sqrt(np.mean(np.array(velocity_errors)**2)))
                metrics['range_std_m'] = float(np.std(range_errors))
                metrics['velocity_std_ms'] = float(np.std(velocity_errors))
                # ── Angle error vs GT — only for array-visible GT targets ──
                # GT angles outside ±90° local azimuth are behind the array.
                # MUSIC cannot estimate them → exclude from angle error metric.
                angle_errors = []
                n_invisible   = 0

                for det_idx, gt_idx, _, _ in matched_pairs:
                    det = targets[det_idx]
                    gt  = gt_for_evaluation[gt_idx]

                    est_angle    = det.get('angle_deg', None)
                    gt_phi_deg   = gt.get('doa_phi_deg',   None)
                    gt_theta_deg = gt.get('doa_theta_deg', None)

                    if gt_phi_deg is None or gt_theta_deg is None:
                        continue

                    try:
                        from veisac.rx.estimation import \
                            _global_doa_to_local_az
                        gt_angle_local = _global_doa_to_local_az(
                            float(gt_phi_deg), float(gt_theta_deg)
                        )
                    except Exception as e:
                        print(f"    ⚠️  Global→Local conversion failed: {e}")
                        continue

                    # ── Visibility gate: skip targets behind the array ──────
                    if abs(gt_angle_local) > 90.0:
                        n_invisible += 1
                        continue          # not visible to array → skip

                    if est_angle is None:
                        continue

                    diff = float(est_angle) - gt_angle_local
                    diff = (diff + 180) % 360 - 180   # circular wrap
                    angle_errors.append(abs(diff))

                if self.verbose:
                    print(f"    Angle pairs: {len(angle_errors)} visible, "
                          f"{n_invisible} skipped (behind array)")

                if angle_errors:
                    metrics['angle_error_deg']     = float(np.mean(angle_errors))
                    metrics['angle_error_median_deg'] = float(np.median(angle_errors))
                    metrics['angle_rmse_deg']      = float(np.sqrt(np.mean(np.array(angle_errors)**2)))
                    metrics['angle_std_deg']       = float(np.std(angle_errors))
                    metrics['n_angle_pairs']       = len(angle_errors)
                    metrics['n_angle_invisible']   = n_invisible
                else:
                    metrics['angle_error_deg']     = float('nan')
                    metrics['angle_error_median_deg'] = float('nan')
                    metrics['angle_rmse_deg']      = float('nan')
                    metrics['angle_std_deg']       = float('nan')
                    metrics['n_angle_pairs']       = 0
                    metrics['n_angle_invisible']   = n_invisible
                metrics['detection_matched'] = True
                metrics['matched_targets'] = len(matched_pairs)
                metrics['total_gt_targets'] = len(gt_for_evaluation)
                metrics['detection_rate'] = len(matched_pairs) / len(gt_for_evaluation)

                if self.verbose:
                    print(f"\n  [MATCHING RESULTS]")
                    print(f"    Total matches: {len(matched_pairs)}/{len(gt_for_evaluation)} "
                        f"({len(matched_pairs)/len(gt_for_evaluation)*100:.1f}%)")
                    print(f"    Range error (mean): {metrics['range_error_m']:.3f} m")
                    print(f"    Range RMSE: {metrics['range_rmse_m']:.3f} m")
                    print(f"    Velocity error (mean): {metrics['velocity_error_ms']:.3f} m/s")
                    print(f"    Velocity RMSE: {metrics['velocity_rmse_ms']:.3f} m/s")
                    print(f"\n  MATCHED PAIRS (first 5):")
                    for i, (det_idx, gt_idx, r_err, v_err) in enumerate(matched_pairs[:5], 1):
                        det = targets[det_idx]
                        gt = gt_for_evaluation[gt_idx]
                        print(f"    {i}. Det[{det_idx}] ↔ GT[{gt_idx}]:")
                        print(f"       Det: R={det['range_m']:.2f}m, V={det['velocity_ms']:+.2f}m/s")
                        print(f"       GT:  R={gt['range_m']:.2f}m, V={gt.get('velocity_ms', 0.0):+.2f}m/s")
                        print(f"       Err: ΔR={r_err:.3f}m, Δv={v_err:.3f}m/s")
                    if len(matched_pairs) > 5:
                        print(f"    ... and {len(matched_pairs) - 5} more matches")
                    unmatched_gts = len(gt_for_evaluation) - len(matched_pairs)
                    if unmatched_gts > 0:
                        print(f"\n  ⚠️  {unmatched_gts} GT targets NOT detected")
                    print(f"{'─'*80}\n")

                if len(targets) > 0:
                    first_target = targets[0]
                    range_crb = first_target.get('crb_range_m', 1e-12)
                    velocity_crb = first_target.get('crb_velocity_ms', 1e-12)
                    metrics['range_error_normalized'] = float(
                        metrics['range_error_m'] / (range_crb + 1e-12))
                    metrics['velocity_error_normalized'] = float(
                        metrics['velocity_error_ms'] / (velocity_crb + 1e-12))

            else:
                metrics['range_error_m'] = float('inf')
                metrics['velocity_error_ms'] = float('inf')
                metrics['detection_matched'] = False
                metrics['matched_targets'] = 0
                metrics['total_gt_targets'] = len(gt_for_evaluation)
                metrics['detection_rate'] = 0.0
                if self.verbose:
                    print(f"\n  ❌ NO MATCHES FOUND")
                    print(f"    Range threshold: {detection_match_threshold_range_m} m")
                    print(f"    Velocity threshold: {detection_match_threshold_velocity_ms} m/s")

        else:
            if self.verbose and len(targets) == 0:
                print(f"\n  ⚠️  No targets detected — cannot compute GT errors")
            elif self.verbose and len(gt_for_evaluation) == 0:
                print(f"\n  ⚠️  No GT targets available — cannot compute errors")

        metrics['range_resolution_m'] = self.config.range_resolution_m
        metrics['velocity_resolution_ms'] = self.config.doppler_resolution_ms
        metrics['metadata'] = {
            'radar_mode': self.config.radar_mode,
            'n_rx_antennas': self.config.n_radar_rx_antennas,
            'n_tx_antennas': self.config.n_radar_tx_antennas,
            'virtual_array_size': self.config.virtual_array_size,
            'coherent_combining': self.config.coherent_combining,
        }

        # Multi-target scene characterization
        if len(gt_for_evaluation) > 0:
            gt_count = len(gt_for_evaluation)
            metrics['n_gt_targets'] = gt_count
            metrics['scene_type'] = 'multi_target' if gt_count > 1 else 'single_target'

            if gt_count > 0 and len(targets) > 0:
                detected_gts_count = 0
                for gt in gt_for_evaluation:   # ← already FoV-gated
                    gt_range = gt['range_m']
                    gt_velocity = gt.get('velocity_ms', 0.0)
                    has_nearby_detection = any(
                        abs(det['range_m'] - gt_range) < detection_match_threshold_range_m and
                        abs(det['velocity_ms'] - gt_velocity) < detection_match_threshold_velocity_ms
                        for det in targets
                    )
                    if has_nearby_detection:
                        detected_gts_count += 1

                metrics['detection_coverage_estimate'] = detected_gts_count / gt_count
                metrics['n_gts_with_nearby_detections'] = detected_gts_count

                if self.verbose and gt_count > 1:
                    print(f"\n  [MULTI-TARGET SCENE SUMMARY]")
                    print(f"    Total GT targets: {gt_count}")
                    print(f"    Total detections: {len(targets)}")
                    print(f"    GTs with nearby detections: {detected_gts_count}/{gt_count} "
                        f"({detected_gts_count/gt_count*100:.1f}%)")
            else:
                metrics['detection_coverage_estimate'] = 0.0
                metrics['n_gts_with_nearby_detections'] = 0
        else:
            metrics['n_gt_targets'] = 0
            metrics['scene_type'] = 'unknown'
            metrics['detection_coverage_estimate'] = None
            metrics['n_gts_with_nearby_detections'] = None

        # ── Range/velocity efficiency (error vs CRB) — computed here after
        #    range_error_m is guaranteed to be set by the GT matching block ──
        if (len(targets) > 0
                and 'top_crb_range_m' in metrics
                and 'range_error_m' in metrics
                and metrics['range_error_m'] != float('inf')):
            crb_r = metrics['top_crb_range_m']
            crb_v = metrics['top_crb_velocity_ms']
            if crb_r > 1e-12:
                metrics['range_efficiency'] = float(
                    metrics['range_error_m'] / (crb_r + 1e-12))
            if crb_v > 1e-12 and metrics['velocity_error_ms'] != float('inf'):
                metrics['velocity_efficiency'] = float(
                    metrics['velocity_error_ms'] / (crb_v + 1e-12))
                
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"[METRICS COMPUTATION COMPLETE]")
            print(f"{'='*80}")
            print(f"  FINAL METRICS:")
            print(f"    radar_snr_rdm   (post-processing): "
                  f"{metrics.get('top_radar_snr_rdm_db', float('nan')):.2f} dB  "
                  f"← CFAR/ranking")
            print(f"    radar_snr_input (pre-processing):  "
                  f"{metrics.get('top_radar_snr_input_db', float('nan')):.2f} dB  "
                  f"← CRB reference")
            print(f"    SNR source: {metrics.get('radar_snr_input_source', 'unknown')}")
            print(f"    N detections: {metrics.get('n_detections', 0)}")
            print(f"    N GT targets:  {metrics.get('n_total_gt', 0)}")

            if 'range_error_m' in metrics and metrics['range_error_m'] != float('inf'):
                print(f"    Matched targets: {metrics.get('matched_targets', 0)}/"
                      f"{metrics.get('total_gt_targets', 0)}")
                print(f"    Detection rate:  {metrics.get('detection_rate', 0.0)*100:.1f}%")
                print(f"  ")
                print(f"    [ESTIMATION ACCURACY]")
                print(f"    Range error (mean):    {metrics['range_error_m']:.4f} m")
                print(f"    Velocity error (mean): {metrics['velocity_error_ms']:.4f} m/s")
                if 'top_crb_range_m' in metrics:
                    print(f"  ")
                    print(f"    [CRB LOWER BOUNDS — radar_snr_input referenced]")
                    print(f"    CRB_range:    {metrics['top_crb_range_m']:.6f} m")
                    print(f"    CRB_velocity: {metrics['top_crb_velocity_ms']:.8f} m/s")
                    if 'range_efficiency' in metrics:
                        eff_r = metrics['range_efficiency']
                        eff_v = metrics.get('velocity_efficiency', float('nan'))
                        status_r = ("✅ near-optimal" if eff_r < 5
                                    else "⚠️  above CRB" if eff_r < 20
                                    else "❌ far from CRB")
                        print(f"    Range efficiency (error/CRB):    {eff_r:.1f}×  {status_r}")
                        print(f"    Velocity efficiency (error/CRB): {eff_v:.1f}×")
                print(f"    Detection matched: "
                      f"{'✅ YES' if metrics.get('detection_matched', False) else '❌ NO'}")
            else:
                print(f"    Range error: NOT COMPUTED (no matches)")
            print(f"{'='*80}\n")

        return metrics
    
    def _print_summary(
        self,
        metrics: Dict,
        targets: List[Dict],
        ground_truth: Optional[Union[Dict, List[Dict]]]
    ):
        
        print(f"\n{'='*80}")
        print(f"[SENSING METRICS SUMMARY]")
        print(f"{'='*80}")
        
        print(f"\n[SIGNAL QUALITY]")
        print(f"  SNR: {metrics['snr_db']:.2f} dB")
        
        if metrics.get('scnr_db') is not None:
            print(f"  SCNR: {metrics['scnr_db']:.2f} dB")
        
        print(f"\n[DETECTIONS]")
        print(f"  Total detections: {metrics['n_detections']}")
        
        if metrics.get('false_alarm_probability') is not None:
            print(f"  P_fa: {metrics['false_alarm_probability']:.6f} ({metrics['false_alarm_probability']*100:.4f}%)")
            print(f"  False alarms: {metrics['n_false_alarms']}")
        
        if len(targets) > 0:
            t0 = targets[0]
            print(f"\n[TOP TARGET]")
            print(f"  Range:    {t0['range_m']:.4f} ± {t0['range_std_m']:.4f} m")
            print(f"  Velocity: {t0['velocity_ms']:+.4f} ± {t0['velocity_std_ms']:.4f} m/s")
            # ── SNR: two distinct quantities ──────────────────────────────
            rdm_db   = t0.get('radar_snr_rdm_db',   t0.get('snr_db',       float('nan')))
            input_db = t0.get('radar_snr_input_db', t0.get('snr_input_db', float('nan')))
            src      = t0.get('radar_snr_input_source', t0.get('snr_input_source', 'unknown'))
            print(f"  radar_snr_rdm:   {rdm_db:.2f} dB  ← post-processing (CFAR/ranking)")
            print(f"  radar_snr_input: {input_db:.2f} dB  ← pre-processing (CRB reference)")
            print(f"  SNR source:      {src}")
            # ── CRB lower bounds ──────────────────────────────────────────
            print(f"  CRB_range:    {t0['crb_range_m']:.6f} m")
            print(f"  CRB_velocity: {t0['crb_velocity_ms']:.8f} m/s")
            if 'angle_deg' in t0 and t0['angle_deg'] is not None:
                print(f"  Angle: {t0['angle_deg']:.2f} ± {t0['angle_std_deg']:.2f} deg")
                if 'crb_angle_deg' in t0 and t0['crb_angle_deg'] is not None:
                    print(f"  CRB_angle: {t0['crb_angle_deg']:.4f} deg")

        if metrics.get('scnr_db') is not None:
            print(f"\n[SCNR]")
            print(f"  SCNR (signal / clutter+noise): {metrics['scnr_db']:.2f} dB")

        if 'range_error_m' in metrics:
            print(f"\n[GROUND TRUTH COMPARISON]")
            print(f"  Range error (mean):    {metrics['range_error_m']:+.4f} m")
            print(f"  Velocity error (mean): {metrics['velocity_error_ms']:+.4f} m/s")
            if 'top_crb_range_m' in metrics:
                print(f"  CRB_range:    {metrics['top_crb_range_m']:.6f} m  "
                      f"← radar_snr_input referenced")
                print(f"  CRB_velocity: {metrics['top_crb_velocity_ms']:.8f} m/s")
            if 'range_efficiency' in metrics:
                eff_r = metrics['range_efficiency']
                eff_v = metrics.get('velocity_efficiency', float('nan'))
                status_r = ("✅ near-optimal" if eff_r < 5
                            else "⚠️  above CRB" if eff_r < 20
                            else "❌ far from CRB")
                print(f"  Range efficiency (error/CRB):    {eff_r:.1f}×  {status_r}")
                print(f"  Velocity efficiency (error/CRB): {eff_v:.1f}×")
            print(f"  Target detected: {'✅ YES' if metrics['detection_matched'] else '❌ NO'}")
        
        print(f"\n{'='*80}\n")
        
    def _visualize_range_doppler(
        self,
        rd_map: np.ndarray,
        targets: List[Dict],
        ground_truth: Optional[Union[Dict, List[Dict]]] = None,
        max_gt_display: int = 100,
        range_threshold_m: float = 6.3,
        velocity_threshold_ms: float = 0.70
    ) -> None:
        import os
        import datetime
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Ellipse

        # ── Save directory — derived from radar mode ──────────────────────
        if self.config.radar_mode == 'monostatic':
            _topology = 'BC_mono_BS'
        elif self.config.n_radar_rx_antennas == 4:
            _topology = 'BC_bist_BS'
        else:
            _topology = 'BC_bist_UE'
        save_dir = (
            f"/home/mababsa/DeepVerse-6G/logs_28GHz_200MHz/"
            f"{_topology}/sensing/visualization"
        )
        os.makedirs(save_dir, exist_ok=True)

        # ── Mode-aware propagation factors ────────────────────────────────
        _delay_factor   = float(self.param_estimator.delay_factor)
        _doppler_factor = float(self.param_estimator.doppler_factor)

        # ── Power map in dB ──────────────────────────────────────────────
        if np.iscomplexobj(rd_map):
            power = np.abs(rd_map) ** 2
        else:
            power = np.asarray(rd_map, dtype=np.float64)
        power_db = 10 * np.log10(power + 1e-12)
        n_range, n_doppler = power_db.shape

        # ── Physical axes ────────────────────────────────────────────────
        chirp_slope = self.config.fmcw_bandwidth_hz / self.config.fmcw_t_chirp_s
        prf         = 1.0 / self.config.fmcw_t_chirp_s
        dc_bin      = self.config.doppler_fft_size // 2

        range_axis = np.array([
            r * self.config.fmcw_sampling_rate_hz / self.config.range_fft_size
            * LIGHTSPEED / (_delay_factor * chirp_slope)
            for r in range(n_range)
        ])
        velocity_axis = np.array([
            (d - dc_bin) * prf / self.config.doppler_fft_size
            * LIGHTSPEED / (_doppler_factor * self.config.carrier_freq_hz)
            for d in range(n_doppler)
        ])

        # ── Display window — zoom to where targets actually appear ────────
        # Full axis extents kept for GT filtering; display clipped separately
        DISPLAY_RANGE_MAX_M  = 600.0   # m   — adjust if targets go beyond
        DISPLAY_VEL_MAX_MS   = 100.0   # m/s — adjust if targets go beyond

        max_range_visible = float(range_axis[-1])
        max_vel_visible   = float(np.max(np.abs(velocity_axis)))

        # ── Normalise GT list ────────────────────────────────────────────
        if ground_truth is None:
            gt_list = []
        elif isinstance(ground_truth, dict):
            gt_list = [ground_truth]
        else:
            gt_list = list(ground_truth)

        # ── Filter GT to targets visible in this RD map window ───────────
        # Two gates applied in sequence:
        #   Gate 1 — FoV: local azimuth ≤ ±90° (array boresight az=330°)
        #   Gate 2 — Range/velocity axis: within RDM display window
        gt_list_visible = []
        n_fov_excluded_viz = 0
        for gt in gt_list:
            r = gt.get('range_m',     gt.get('range',    None))
            v = gt.get('velocity_ms', gt.get('velocity', 0.0))
            if r is None:
                continue

            # Gate 1: FoV check
            phi   = gt.get('doa_phi_deg',   None)
            theta = gt.get('doa_theta_deg', None)
            if phi is not None and theta is not None:
                try:
                    from veisac.rx.estimation import \
                        _global_doa_to_local_az
                    local_az = _global_doa_to_local_az(float(phi), float(theta))
                    if abs(local_az) > 90.0:
                        n_fov_excluded_viz += 1
                        continue   # behind array — skip
                except Exception:
                    pass   # conversion failed — conservative: keep

            # Gate 2: RDM axis window
            if 0 <= r <= max_range_visible and abs(v) <= max_vel_visible:
                gt_list_visible.append(gt)

        if n_fov_excluded_viz > 0:
            print(f"  [VISUALIZATION] FoV gate excluded {n_fov_excluded_viz} "
                  f"behind-array GT targets from plot")

        # Sort by range and cap at max_gt_display
        gt_list_visible = sorted(
            gt_list_visible,
            key=lambda g: g.get('range_m', g.get('range', 0.0))
        )
        if len(gt_list_visible) > max_gt_display:
            gt_list_visible = gt_list_visible[:max_gt_display]

        print(f"\n  [VISUALIZATION] GT targets in scene: {len(gt_list)}")
        print(f"  [VISUALIZATION] GT targets visible in RD window: {len(gt_list_visible)}")
        print(f"  [VISUALIZATION] Estimated targets available: {len(targets)}")

        # ── Greedy nearest-neighbour matching: one best det per GT ───────
        # For each GT find the single closest unused detection.
        matched_pairs = []   # list of (gt, est, dr, dv)
        used_est      = set()

        for gt in gt_list_visible:
            r_gt      = gt.get('range_m',     gt.get('range',    0.0))
            v_gt      = gt.get('velocity_ms', gt.get('velocity', 0.0))
            best_idx  = None
            best_dist = float('inf')

            for i, t in enumerate(targets):
                if i in used_est:
                    continue
                dr   = abs(t['range_m']     - r_gt)
                dv   = abs(t['velocity_ms'] - v_gt)
                dist = dr + dv * 10.0
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = i          # ← fixed: no longer resets to None

            if best_idx is not None:
                t  = targets[best_idx]
                dr = abs(t['range_m']     - r_gt)
                dv = abs(t['velocity_ms'] - v_gt)
                # ← Apply same thresholds as _compute_metrics so
                #   the plot errors exactly match the CSV values
                if dr < range_threshold_m and dv < velocity_threshold_ms:
                    matched_pairs.append((gt, t, dr, dv))
                    used_est.add(best_idx)

        best_estimates = [p[1] for p in matched_pairs]
        mean_dr = float(np.mean([p[2] for p in matched_pairs])) if matched_pairs else float('nan')
        mean_dv = float(np.mean([p[3] for p in matched_pairs])) if matched_pairs else float('nan')
        n_missed = len(gt_list_visible) - len(matched_pairs)

        print(f"  [VISUALIZATION] Matched pairs: {len(matched_pairs)}")
        print(f"  [VISUALIZATION] Missed GT targets: {n_missed}")
        print(f"  [VISUALIZATION] Mean range error: {mean_dr:.3f} m  ← matches CSV")
        print(f"  [VISUALIZATION] Mean velocity error: {mean_dv:.4f} m/s  ← matches CSV")
        print(f"  [VISUALIZATION] Range threshold used: {range_threshold_m} m")
        print(f"  [VISUALIZATION] Velocity threshold used: {velocity_threshold_ms} m/s")

        # ── Figure ────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(16, 8), facecolor='#0a0a0a')
        fig.subplots_adjust(left=0.07, right=0.93, top=0.91, bottom=0.10)

        vmin = float(np.percentile(power_db, 50))
        vmax = float(np.percentile(power_db, 99.5))
        power_db_clipped = np.clip(power_db, vmin, vmax)

        im = ax.imshow(
            power_db_clipped.T,
            aspect='auto',
            origin='lower',
            extent=[range_axis[0], range_axis[-1],
                    velocity_axis[0], velocity_axis[-1]],
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            interpolation='bilinear',
        )
        cbar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.03, shrink=0.85)
        cbar.set_label('Power (dB)', fontsize=11, color='white', labelpad=8)
        cbar.ax.yaxis.set_tick_params(color='white', labelsize=9)
        cbar.outline.set_edgecolor('#555555')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
        cbar.ax.text(2.8, 0.02, 'noise floor', transform=cbar.ax.transAxes,
                     fontsize=7, color='#aaaaaa', ha='left', va='bottom')
        cbar.ax.text(2.8, 0.98, 'peak signal', transform=cbar.ax.transAxes,
                     fontsize=7, color='#aaaaaa', ha='left', va='top')

        GT_COLOR  = '#FF5555'
        EST_COLOR = '#00DDFF'
        CRB_COLOR = '#FFDD00'
        MISS_COLOR = '#FF9900'

        # ── Plot ALL GT targets ───────────────────────────────────────────
        # Track which GTs were matched
        matched_gt_indices = set(
            gt_list_visible.index(p[0])
            for p in matched_pairs
            if p[0] in gt_list_visible
        )

        for i, gt in enumerate(gt_list_visible):
            if i not in matched_gt_indices:
                continue                          # ← skip missed GT markers
            r_gt = gt.get('range_m',     gt.get('range',    0.0))
            v_gt = gt.get('velocity_ms', gt.get('velocity', 0.0))
            ax.scatter(r_gt, v_gt,
                       marker='x', s=80, linewidths=2.0,
                       color=GT_COLOR, zorder=5)

        # ── Plot best estimated for each matched GT ───────────────────────
        for gt, t, dr, dv in matched_pairs:
            r_e = t['range_m']
            v_e = t['velocity_ms']
            ax.scatter(r_e, v_e,
                       marker='o', s=60, linewidths=1.8,
                       facecolors='none', edgecolors=EST_COLOR,
                       zorder=6)

            # CRB ellipse on each matched pair
            #crb_r = t.get('crb_range_m',     None)
            #crb_v = t.get('crb_velocity_ms', None)
            #if crb_r and crb_v and crb_r > 0 and crb_v > 0:
                #ax.add_patch(Ellipse(
                #    xy=(r_e, v_e),
                #    width=2 * crb_r, height=2 * crb_v,
                #    edgecolor=CRB_COLOR, facecolor='none',
                #    linewidth=1.0, linestyle='--', zorder=7
                #))

        # ── Summary text box ──────────────────────────────────────────────
        summary = (
            #f"GT targets (scene):      {len(gt_list)}\n"
            f"GT targets (visible):    {len(matched_pairs)}\n"
            f"Matched estimates:       {len(matched_pairs)}\n"
            #f"Missed GT targets:       {n_missed}\n"
            #f"Detection rate:          {len(matched_pairs)/max(len(gt_list_visible),1)*100:.1f}%\n"
            f"Mean range error:        {mean_dr:.3f} m\n"
            f"Mean velocity error:     {mean_dv:.4f} m/s"
        )
        ax.text(
            0.01, 0.99, summary,
            transform=ax.transAxes,
            fontsize=8.5, color='white',
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='black',
                      alpha=0.70, edgecolor='#555')
        )

        # ── Legend ────────────────────────────────────────────────────────
        import matplotlib.lines as mlines
        legend_handles = [
            mpatches.Patch(color=GT_COLOR,
                           label=f'GT detected ({len(matched_pairs)})'),
            #mpatches.Patch(color=MISS_COLOR,
                           #label=f'GT missed ({n_missed})'),
            mlines.Line2D([], [], color=EST_COLOR, marker='o',
                          linestyle='None', markersize=7,
                          markerfacecolor='none', markeredgewidth=1.5,
                          label=f'Best estimates ({len(best_estimates)})'),
            #mpatches.Patch(color=CRB_COLOR,
                           #label='CRB 1σ ellipse'),
        ]
        ax.legend(handles=legend_handles, loc='upper right',
                  fontsize=9, framealpha=0.75,
                  facecolor='#111111', labelcolor='white',
                  edgecolor='#444444')

        # ── Zoom axes to display window ───────────────────────────────────
        ax.set_xlim(0.0,               DISPLAY_RANGE_MAX_M)
        ax.set_ylim(-DISPLAY_VEL_MAX_MS, DISPLAY_VEL_MAX_MS)

        # ── Zero-velocity reference line ──────────────────────────────────
        ax.axhline(y=0, color='#ffffff', alpha=0.20,
                   linewidth=0.8, linestyle='--')

        # ── Grid ──────────────────────────────────────────────────────────
        ax.grid(True, which='major', color='white', alpha=0.12,
                linestyle='--', linewidth=0.6)
        ax.minorticks_on()
        ax.grid(True, which='minor', color='white', alpha=0.05,
                linestyle=':', linewidth=0.4)

        # ── Labels, ticks, title ──────────────────────────────────────────
        ax.set_xlabel('Range (m)', fontsize=12, color='white', labelpad=8)
        ax.set_ylabel('Velocity (m/s)', fontsize=12, color='white', labelpad=8)
        ax.tick_params(colors='white', labelsize=10, length=4, width=0.8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
        ax.set_facecolor('#0a0a0a')
        if self.config.radar_mode == 'monostatic':
            _topo_label = 'MONOSTATIC BS'
        elif self.config.n_radar_rx_antennas == 4:
            _topo_label = 'BISTATIC BS'
        else:
            _topo_label = 'BISTATIC UE'
        ax.set_title(
            f'Range-Doppler Map  |  {_topo_label}  '
            f'|  {self.config.carrier_freq_hz/1e9:.1f} GHz  '
            f'|  BW = {self.config.fmcw_bandwidth_hz/1e6:.0f} MHz  '
            f'|  {self.config.n_radar_tx_antennas}×{self.config.n_radar_rx_antennas} MIMO',
            fontsize=11, color='white', pad=10, loc='center'
        )

        # ── Save ──────────────────────────────────────────────────────────
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        fname = os.path.join(save_dir, f'rdmap_{timestamp}.png')
        fig.savefig(fname, dpi=180, bbox_inches='tight',
                    facecolor=fig.get_facecolor(),
                    edgecolor='none')
        plt.close(fig)
        print(f"\n  📊 RD map saved → {fname}")
        print(f"     GT shown: {len(gt_list_visible)} "
              f"(matched: {len(matched_pairs)}, "
              f"missed: {n_missed})")

    
    def generate_synthetic_signal(
        self,
        target_range_m: float,
        target_velocity_ms: float,
        target_rcs: float = 1.0,
        target_snr_db: float = 10.0,
        add_noise: bool = True,
        seed: Optional[int] = None
    ) -> np.ndarray:
        
        xp = self.xp
        
        if seed is not None:
            np.random.seed(seed)
            if self.use_gpu:
                cp.random.seed(seed)
        
        if target_range_m < 0 or target_range_m > self.config.max_range_m:
            warnings.warn(
                f"Target range {target_range_m:.1f} m outside valid range "
                f"[0, {self.config.max_range_m:.1f}] m"
            )
        
        if abs(target_velocity_ms) > self.config.max_velocity_ms:
            warnings.warn(
                f"Target velocity {target_velocity_ms:.1f} m/s exceeds max "
                f"±{self.config.max_velocity_ms:.1f} m/s"
            )
        
        t_fast = np.arange(self.config.fmcw_n_samples_per_chirp) / self.config.fmcw_sampling_rate_hz
        t_slow = np.arange(self.config.fmcw_n_chirps) * self.config.fmcw_t_chirp_s
        
        if self.use_gpu:
            t_fast = cp.asarray(t_fast)
            t_slow = cp.asarray(t_slow)
        
        tau = (2 * target_range_m) / LIGHTSPEED
        f_d = (2 * target_velocity_ms * self.config.carrier_freq_hz) / LIGHTSPEED
        
        chirp_slope = self.config.fmcw_bandwidth_hz / self.config.fmcw_t_chirp_s
        f_beat = chirp_slope * tau
        
        snr_linear = 10 ** (target_snr_db / 10)
        noise_power = self.config.sens_noise_power_w
        n_channels = self.config.n_radar_rx_antennas * self.config.n_radar_tx_antennas
        
        amplitude = xp.sqrt(snr_linear * noise_power / n_channels)
        
        if self.verbose:
            print(f"\n{'─'*80}")
            print(f"[SYNTHETIC SIGNAL GENERATION]")
            print(f"{'─'*80}")
            print(f"  Target range: {target_range_m} m")
            print(f"  Target velocity: {target_velocity_ms:+.2f} m/s")
            print(f"  Delay: {tau*1e6:.3f} µs (round-trip)")
            print(f"  Beat frequency: {f_beat/1e6:.3f} MHz")
            print(f"  Doppler frequency: {f_d:.3f} Hz (two-way)")
            print(f"  Target SNR: {target_snr_db:.1f} dB")
        
        received_signal = xp.zeros((
            self.config.n_radar_rx_antennas,
            self.config.n_radar_tx_antennas,
            self.config.fmcw_n_samples_per_chirp,
            self.config.fmcw_n_chirps
        ), dtype=complex)
        
        for m in range(self.config.fmcw_n_chirps):
            phase_doppler = 2 * xp.pi * f_d * t_slow[m]
            
            phase_chirp = xp.pi * chirp_slope * ((t_fast - tau)**2 - t_fast**2)
            
            target_echo = amplitude * xp.exp(1j * (phase_chirp + phase_doppler))
            
            for rx_idx in range(self.config.n_radar_rx_antennas):
                for tx_idx in range(self.config.n_radar_tx_antennas):
                    received_signal[rx_idx, tx_idx, :, m] = target_echo
        
        if add_noise:
            noise_std = xp.sqrt(noise_power / 2)
            
            if self.use_gpu:
                noise = noise_std * (cp.random.randn(*received_signal.shape) + 
                                   1j * cp.random.randn(*received_signal.shape))
            else:
                noise = noise_std * (np.random.randn(*received_signal.shape) + 
                                   1j * np.random.randn(*received_signal.shape))
            
            received_signal += noise
        
        if self.use_gpu:
            received_signal = cp.asnumpy(received_signal)
        
        return received_signal


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SENSING RECEIVER v4.2 - CORRECTED & PRODUCTION-READY")
    print("="*80)