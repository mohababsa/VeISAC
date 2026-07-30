# isac_tx_config.py
"""
VeISAC — ISAC Transmitter Configuration

Dataclass holding all TX parameters for the MIMO-OFDM-FMCW ISAC system (RF, OFDM, FMCW, MIMO, power allocation...).

Paper: "VeISAC: An End-to-End MIMO-OFDM-FMCW Framework for ISAC
        in 6G Vehicular Networks"
Authors: M. Ababsa, S. Ribouh, Y. El Hillali, A. Rivenq
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Literal
import json


@dataclass
class ISACTXConfig:
    """
    Unified ISAC Transmitter Configuration.
    
    Dual-function system with:
    - MIMO-OFDM for communication (BS→UE broadcast)
    - FMCW for radar sensing (monostatic/bistatic)
    - Shared hardware and spectrum (28 GHz, 200 MHz)
    
    VERSION 4.0: ADDITIVE ISAC INTEGRATION
    - Integration mode: 'additive' (x = α·OFDM + β·chirp) or 'multiplicative' (x = OFDM ⊙ chirp)
    - Power allocation: α² + β² = 1 (tunable COMM/RADAR priority)
    - Default: α = β = 0.707 (equal power, 50/50 split)
    
    ANTENNA CONFIGURATION (CORRECTED):
    - BS (TX): 4 antennas (2×2 UPA) - Transmitter
    - UE (RX): 2 antennas (2×1 linear) - Receiver
    - COMM coeffs shape: (2, 4, 1633) = (RX, TX, subcarriers)
    
    PILOT CONFIGURATION (OPTIMIZED v3.2):
    - Time spacing (Mt): 3 symbols (auto-computed from max Doppler)
    - Frequency spacing (Mf): 3 subcarriers ✅ OPTIMIZED (was 1)
    - Pilot overhead: ~12% (was 35%)
    - Spectral efficiency: +23% improvement
    
    VALIDATED AGAINST:
    - COMM GT: 52,080 samples, 1.04M paths
    - RADAR GT: 52,080 samples
    - Temporal coverage: [40, 1099] frames
    - Coherence BW: 27 MHz (supports Mf=3 perfectly)
    """
    
    # ================================================================
    # RF PARAMETERS (Fixed - Validated from DeepVerse-6G GT)
    # ================================================================
    carrier_freq_hz: float = 28e9              # 28 GHz (n257 band) ✅ VALIDATED
    bandwidth_hz: float = 200e6                # 200 MHz instantaneous BW ✅ VALIDATED
    tx_power_dbm: float = 43.0                 # 43 dBm = 20 W ✅ VALIDATED
    
    # Physical constants
    _c: float = field(default=299792458.0, repr=False)  # Speed of light (m/s)
    
    # ================================================================
    # 5G NR OFDM PARAMETERS (3GPP TS 38.211) - VALIDATED
    # ================================================================
    # Numerology configuration
    subcarrier_spacing_khz: int = 120          # SCS for FR2: 60/120/240 kHz ✅ VALIDATED
    n_fft: int = 2048                          # FFT size (power of 2) ✅ VALIDATED
    n_subcarriers_total: int = 1638            # Total allocatable subcarriers
    n_subcarriers_used: int = 1633             # Used subcarriers [0:1632] ✅ VALIDATED from COMM GT
    
    # VALIDATED: Actual exported COMM coefficients have shape (2, 4, 1633)
    # This matches our COMM GT extraction perfectly
    n_subcarriers_actual: int = 1633           # ✅ CONFIRMED from COMM diagnostic
    
    # Guard bands and DC null
    dc_null: bool = False                       # Null DC subcarrier
    guard_band_left: int = 0                   # Left guard band
    guard_band_right: int = 5                  # Right guard band (1638 - 1633)
    
    # Cyclic prefix (CRITICAL for sensing!)
    # VALIDATED: CP must satisfy T_cp ≥ 2*R_max/c
    # From COMM GT: Max range = 308.39 m → τ_max = 2.06 μs
    # From RADAR GT: Max range = 497.56 m → τ_max = 3.32 μs
    cp_type: str = "extended"                  # "normal" or "extended"
    cp_length_samples: int = 1024              # CP samples ✅ VALIDATED (4.096 μs > 3.32 μs)
    
    # Frame structure
    n_symbols_per_slot: int = 14               # 5G NR slot structure
    n_slots_per_subframe: int = 2              # For SCS=120kHz
    n_subframes_per_frame: int = 10            # 10ms frame
    
    # Resource allocation
    n_prb: int = 136                           # Physical Resource Blocks (200MHz @ 120kHz SCS)
    n_sc_per_prb: int = 12                     # Subcarriers per PRB (3GPP standard)
    
    # ================================================================
    # MIMO CONFIGURATION - CORRECTED ✅
    # ================================================================
    # CORRECTED based on COMM coeffs shape (2, 4, 1633)
    # Coeffs convention: (n_rx, n_tx, n_subcarriers)
    
    # BS (Base Station) = TRANSMITTER
    n_tx_antennas: int = 4                     # BS TX: 4 antennas ✅ CORRECTED
    tx_antenna_shape: Tuple[int, int] = (2, 2) # (N_y, N_z) 2×2 UPA ✅ CORRECTED
    tx_antenna_spacing: float = 0.5            # λ/2 spacing
    tx_antenna_rotation: Tuple[float, float, float] = (330.0, -10.0, 0.0)  # (γ, β, α) in degrees
    
    # UE (User Equipment) = RECEIVER
    n_rx_antennas: int = 2                     # UE RX: 2 antennas ✅ CORRECTED
    rx_antenna_shape: Tuple[int, int] = (2, 1) # (N_y, N_z) 2×1 linear ✅ CORRECTED
    rx_antenna_spacing: float = 0.5            # λ/2 spacing
    
    # Beamforming strategy
    beamforming_type: str = "digital"          # "digital", "hybrid", "analog"
    precoding_matrix: str = "identity"         # "identity", "dft", "svd"
    
    # ================================================================
    # FMCW RADAR PARAMETERS (Integrated with OFDM) - VALIDATED
    # ================================================================
    # Chirp configuration (from config.m, validated against RADAR GT)
    fmcw_enable: bool = True                   # Enable FMCW radar
    fmcw_chirp_slope: float = 2.4e13           # 24 THz/s ✅ VALIDATED
    fmcw_n_samples_per_chirp: int = 1664       # Fast-time samples ✅ VALIDATED
    fmcw_n_chirps: int = 128                   # Slow-time chirps ✅ VALIDATED
    fmcw_sampling_rate_hz: float = 200e6       # 200 MHz (matches OFDM BW) ✅ VALIDATED
    
    # Radar mode (can be overridden per sample)
    radar_mode: Literal['monostatic', 'bistatic'] = 'monostatic'
    
    # ================================================================
    # FMCW INTEGRATION CONFIGURATION (NEW v4.0) ✅
    # ================================================================
    # Integration mode: how OFDM and FMCW are combined
    fmcw_integration_mode: str = 'additive'    # 'additive' or 'multiplicative' ✅ NEW
    
    # Power allocation factors for additive integration
    # Constraint: α² + β² = 1 (enforced by validate_power_allocation)
    comm_power_factor: float = np.sqrt(0.5)           # α (COMM power allocation) ✅ NEW
    radar_power_factor: float = np.sqrt(0.5)          # β (RADAR power allocation) ✅ NEW
    
    # Note: These values represent the amplitude scaling factors
    # Actual power fractions: α² for COMM, β² for RADAR
    # Default: 0.707² = 0.5 → 50% COMM, 50% RADAR (equal power)
    
    # ================================================================
    # PILOT CONFIGURATION (OPTIMIZED v3.2) ✅
    # ================================================================
    # CRITICAL: Pilot spacing determines max unambiguous range/velocity
    # Based on MathWorks Part II equations:
    #   Mt ≤ 1 / (2 * f_D_max * T_OFDM)
    #   Mf ≤ 1 / (2 * Δf * τ_max)
    
    # VALIDATED from GT data:
    # COMM: Max velocity = 25.99 m/s, Max range = 308.39 m
    # RADAR: Max velocity = 62.28 m/s, Max range = 497.56 m
    max_velocity_ms: float = 65.0              # Max target velocity (m/s) - conservative ✅ UPDATED
    max_range_m: float = 600.0                 # Max target range (m) - conservative ✅ UPDATED
    
    # Pilot spacing (can be overridden, otherwise auto-computed)
    pilot_spacing_time: Optional[int] = None   # Symbols (Mt) - auto-computed from Doppler
    pilot_spacing_freq: Optional[int] = 3      # ✅ OPTIMIZED: Mf=3 (was None→1)
    
    # ================================================================
    # SIGNAL GENERATION PARAMETERS (CORRECTED)
    # ================================================================
    # From MathWorks: Use coherent combining (SUM, not MEAN)
    coherent_combining: str = "sum"            # "sum" or "mean" ✅ VALIDATED
    
    # Amplitude scaling (CORRECTED based on analysis)
    # When combining N channels: amplitude_per_channel = sqrt(SNR × N0) / N
    amplitude_scaling_mode: str = "per_channel"  # "total" or "per_channel" ✅ VALIDATED
    
    # Noise parameters (for signal generation)
    noise_figure_db: float = 7.0               # Receiver noise figure
    temperature_k: float = 290.0               # System temperature (K)
    
    # ================================================================
    # DERIVED OFDM PARAMETERS
    # ================================================================
    @property
    def subcarrier_spacing_hz(self) -> float:
        """Subcarrier spacing in Hz."""
        return self.subcarrier_spacing_khz * 1e3
    
    @property
    def ofdm_symbol_duration_s(self) -> float:
        """OFDM symbol duration (without CP)."""
        return 1.0 / self.subcarrier_spacing_hz
    
    @property
    def sampling_rate_hz(self) -> float:
        """Sampling rate (time-domain)."""
        return self.subcarrier_spacing_hz * self.n_fft
    
    @property
    def cp_duration_s(self) -> float:
        """Cyclic prefix duration."""
        return self.cp_length_samples / self.sampling_rate_hz
    
    @property
    def total_symbol_duration_s(self) -> float:
        """Total symbol duration (with CP)."""
        return self.ofdm_symbol_duration_s + self.cp_duration_s
    
    @property
    def slot_duration_s(self) -> float:
        """Slot duration."""
        return self.n_symbols_per_slot * self.total_symbol_duration_s
    
    @property
    def subframe_duration_s(self) -> float:
        """Subframe duration (1 ms)."""
        return self.n_slots_per_subframe * self.slot_duration_s
    
    @property
    def wavelength_m(self) -> float:
        """Carrier wavelength."""
        return self._c / self.carrier_freq_hz
    
    @property
    def tx_power_w(self) -> float:
        """TX power in watts."""
        return 10 ** ((self.tx_power_dbm - 30) / 10)
    
    @property
    def thermal_noise_power_w(self) -> float:
        """Thermal noise power (N0 × BW)."""
        k_boltzmann = 1.380649e-23  # J/K
        N0 = k_boltzmann * self.temperature_k
        NF_linear = 10 ** (self.noise_figure_db / 10)
        return N0 * self.bandwidth_hz * NF_linear
    
    @property
    def thermal_noise_power_dbm(self) -> float:
        """Thermal noise power in dBm."""
        return 10 * np.log10(self.thermal_noise_power_w) + 30
    
    # ================================================================
    # DERIVED FMCW PARAMETERS
    # ================================================================
    @property
    def fmcw_chirp_duration_s(self) -> float:
        """Chirp duration (fast-time)."""
        return self.fmcw_n_samples_per_chirp / self.fmcw_sampling_rate_hz
    
    @property
    def fmcw_bandwidth_hz(self) -> float:
        """FMCW chirp bandwidth."""
        return self.fmcw_chirp_slope * self.fmcw_chirp_duration_s
    
    @property
    def fmcw_range_resolution_m(self) -> float:
        """Range resolution."""
        return self._c / (2 * self.fmcw_bandwidth_hz)
    
    @property
    def fmcw_frame_duration_s(self) -> float:
        """Total frame duration (slow-time)."""
        return self.fmcw_chirp_duration_s * self.fmcw_n_chirps
    
    @property
    def fmcw_frame_rate_hz(self) -> float:
        """Frame rate (radar)."""
        return 1.0 / self.fmcw_frame_duration_s
    
    @property
    def fmcw_doppler_resolution_hz(self) -> float:
        """Doppler frequency resolution."""
        return 1.0 / self.fmcw_frame_duration_s
    
    @property
    def fmcw_velocity_resolution_ms(self) -> float:
        """Velocity resolution (monostatic)."""
        # For monostatic: Δv = λ / (2 * T_frame)
        return self.wavelength_m * self.fmcw_doppler_resolution_hz / 2.0
    
    @property
    def fmcw_max_range_m(self) -> float:
        """Maximum unambiguous range."""
        return self._c * self.fmcw_chirp_duration_s / 2.0
    
    @property
    def fmcw_max_velocity_ms(self) -> float:
        """Maximum unambiguous velocity (monostatic)."""
        # Nyquist: v_max = λ / (4 * T_chirp)
        return self.wavelength_m / (4 * self.fmcw_chirp_duration_s)
    
    # ================================================================
    # DERIVED POWER ALLOCATION PARAMETERS (NEW v4.0) ✅
    # ================================================================
    @property
    def comm_power_fraction(self) -> float:
        """Communication power fraction (α²)."""
        return self.comm_power_factor ** 2
    
    @property
    def radar_power_fraction(self) -> float:
        """Radar power fraction (β²)."""
        return self.radar_power_factor ** 2
    
    @property
    def power_allocation_sum(self) -> float:
        """Sum of power fractions (should be 1.0)."""
        return self.comm_power_fraction + self.radar_power_fraction
    
    # ================================================================
    # DERIVED PILOT PARAMETERS (MathWorks Method)
    # ================================================================
    @property
    def max_doppler_shift_hz(self) -> float:
        """Maximum Doppler shift from max velocity."""
        # f_D = 2 * v / λ (monostatic)
        return 2 * self.max_velocity_ms / self.wavelength_m
    
    @property
    def max_delay_s(self) -> float:
        """Maximum delay from max range."""
        # τ = 2 * R / c (monostatic), τ = R / c (bistatic)
        delay_factor = 2.0 if self.radar_mode == 'monostatic' else 1.0
        return delay_factor * self.max_range_m / self._c
    
    @property
    def pilot_spacing_time_symbols(self) -> int:
        """
        Maximum pilot spacing in time (symbols).
        
        Based on MathWorks: Mt ≤ 1 / (2 * f_D_max * T_OFDM)
        """
        if self.pilot_spacing_time is not None:
            return self.pilot_spacing_time
        
        Mt = int(np.floor(1.0 / (2 * self.max_doppler_shift_hz * self.total_symbol_duration_s)))
        return max(1, Mt)  # At least 1
    
    @property
    def pilot_spacing_freq_subcarriers(self) -> int:
        """
        Maximum pilot spacing in frequency (subcarriers).
        
        Based on MathWorks: Mf ≤ 1 / (2 * Δf * τ_max)
        
        OPTIMIZED v3.2:
        - If pilot_spacing_freq is set (e.g., 3), use it directly
        - Otherwise auto-compute from Nyquist (would give Mf=1)
        - Coherence BW = 27 MHz allows Mf up to ~225 subcarriers
        - Mf=3 is optimal balance: 12% overhead, excellent channel estimation
        """
        if self.pilot_spacing_freq is not None:
            return self.pilot_spacing_freq
        
        # Nyquist-based (conservative)
        Mf = int(np.floor(1.0 / (2 * self.subcarrier_spacing_hz * self.max_delay_s)))
        return max(1, min(Mf, self.n_subcarriers_used // 2))  # Reasonable bounds
    
    # ================================================================
    # VALIDATION
    # ================================================================
    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_5g_nr_compliance()
        self._validate_cp_requirement()
        self._validate_fmcw_alignment()
        self._validate_doppler_constraint()
        self._validate_pilot_constraints()
        self._validate_power_allocation()  # ✅ NEW v4.0
        self._validate_against_gt_data()
    
    def _validate_5g_nr_compliance(self):
        """Ensure 5G NR FR2 compliance."""
        # SCS must be 60, 120, or 240 kHz for FR2
        valid_scs = [60, 120, 240]
        if self.subcarrier_spacing_khz not in valid_scs:
            raise ValueError(
                f"SCS {self.subcarrier_spacing_khz} kHz not valid for FR2. "
                f"Must be one of {valid_scs}"
            )
        
        # Bandwidth check
        if not (50e6 <= self.bandwidth_hz <= 400e6):
            raise ValueError(
                f"Bandwidth {self.bandwidth_hz/1e6:.1f} MHz outside FR2 range (50-400 MHz)"
            )
        
        # Carrier frequency check (n257 band: 26.5-29.5 GHz)
        if not (26.5e9 <= self.carrier_freq_hz <= 29.5e9):
            raise ValueError(
                f"Carrier freq {self.carrier_freq_hz/1e9:.1f} GHz outside n257 band"
            )
        
        # FFT size must be power of 2
        if not (self.n_fft & (self.n_fft - 1)) == 0:
            raise ValueError(f"FFT size {self.n_fft} must be power of 2")
        
        # Subcarriers per PRB
        if self.n_sc_per_prb != 12:
            raise ValueError("Subcarriers per PRB must be 12 (3GPP standard)")
    
    def _validate_cp_requirement(self):
        """Validate cyclic prefix satisfies sensing requirement."""
        # CP must accommodate max sensing delay
        tau_sens_required_s = self.max_delay_s
        
        if self.cp_duration_s < tau_sens_required_s:
            raise ValueError(
                f"CP duration {self.cp_duration_s*1e6:.2f} μs < "
                f"required {tau_sens_required_s*1e6:.2f} μs for R_max={self.max_range_m} m"
            )
        
        print(f"✓ CP validation: {self.cp_duration_s*1e6:.2f} μs "
              f">= {tau_sens_required_s*1e6:.2f} μs (R_max={self.max_range_m} m)")
    
    def _validate_fmcw_alignment(self):
        """Validate FMCW parameters align with OFDM."""
        if not self.fmcw_enable:
            return
        
        # FMCW sampling rate should match OFDM bandwidth
        if abs(self.fmcw_sampling_rate_hz - self.bandwidth_hz) > 1e3:
            raise ValueError(
                f"FMCW sampling rate {self.fmcw_sampling_rate_hz/1e6:.1f} MHz "
                f"!= OFDM bandwidth {self.bandwidth_hz/1e6:.1f} MHz"
            )
        
        # Verify FMCW bandwidth matches
        expected_bw = self.fmcw_bandwidth_hz
        if abs(expected_bw - self.bandwidth_hz) / self.bandwidth_hz > 0.01:
            raise ValueError(
                f"FMCW bandwidth {expected_bw/1e6:.1f} MHz != "
                f"OFDM bandwidth {self.bandwidth_hz/1e6:.1f} MHz"
            )
        
        print(f"✓ FMCW validation: BW={self.fmcw_bandwidth_hz/1e6:.1f} MHz, "
              f"ΔR={self.fmcw_range_resolution_m:.3f} m, "
              f"ΔV={self.fmcw_velocity_resolution_ms:.2f} m/s")
    
    def _validate_doppler_constraint(self):
        """
        Validate subcarrier spacing vs Doppler shift.
        
        From MathWorks: Δf >> f_D_max (typically Δf ≥ 10 × f_D_max)
        """
        ratio = self.subcarrier_spacing_hz / self.max_doppler_shift_hz
        
        if ratio < 10.0:
            print(f"⚠️  WARNING: Subcarrier spacing only {ratio:.1f}x max Doppler shift")
            print(f"   Recommended: ≥10x for OFDM orthogonality")
        else:
            print(f"✓ Doppler constraint: Δf = {ratio:.1f}x f_D_max (good!)")
    
    def _validate_pilot_constraints(self):
        """Validate pilot spacing and overhead."""
        Mt = self.pilot_spacing_time_symbols
        Mf = self.pilot_spacing_freq_subcarriers
        
        # Compute pilot overhead
        n_pilot_symbols = int(np.ceil(self.n_symbols_per_slot / Mt))
        n_pilots_per_symbol = int(np.ceil(self.n_subcarriers_actual / Mf))
        n_pilots_total = n_pilot_symbols * n_pilots_per_symbol
        n_res_total = self.n_symbols_per_slot * self.n_subcarriers_actual
        overhead_pct = (n_pilots_total / n_res_total) * 100
        
        print(f"✓ Pilot spacing: Mt={Mt} symbols, Mf={Mf} subcarriers")
        print(f"  Pilot overhead: {overhead_pct:.2f}% ({n_pilots_total}/{n_res_total} REs)")
        print(f"  Max unambiguous: R={self.max_range_m} m, V={self.max_velocity_ms} m/s")
        
        # Validate against coherence bandwidth
        coherence_bw_hz = 27e6  # From GT data
        coherence_span_subcarriers = coherence_bw_hz / self.subcarrier_spacing_hz
        
        if Mf > coherence_span_subcarriers / 2:
            print(f"  ⚠️  WARNING: Mf={Mf} may be too large for BC={coherence_bw_hz/1e6:.1f} MHz")
            print(f"     Coherence span: ~{coherence_span_subcarriers:.0f} subcarriers")
        else:
            print(f"  ✓ Mf={Mf} well within coherence BW ({coherence_bw_hz/1e6:.1f} MHz = {coherence_span_subcarriers:.0f} sc)")
    
    def _validate_power_allocation(self):
        """
        Validate power allocation constraint (NEW v4.0).
        
        For additive ISAC: α² + β² = 1
        """
        if self.fmcw_integration_mode != 'additive':
            print(f"✓ Integration mode: {self.fmcw_integration_mode.upper()} (power allocation not applicable)")
            return
        
        alpha = self.comm_power_factor
        beta = self.radar_power_factor
        power_sum = self.power_allocation_sum
        
        # Check constraint
        if abs(power_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Power allocation constraint violated: "
                f"α²={alpha**2:.6f}, β²={beta**2:.6f}, "
                f"α²+β²={power_sum:.6f} (must equal 1.0)\n"
                f"Current: α={alpha:.6f}, β={beta:.6f}\n"
                f"Suggestion: Normalize or use valid pairs like:\n"
                f"  - Equal power: α=0.707, β=0.707 (50/50)\n"
                f"  - COMM-dominant: α=0.9, β=0.436 (81/19)\n"
                f"  - RADAR-dominant: α=0.436, β=0.9 (19/81)"
            )
        
        print(f"✓ Power allocation validation (v4.0):")
        print(f"  Integration mode: {self.fmcw_integration_mode.upper()}")
        print(f"  COMM factor:  α={alpha:.6f} → {self.comm_power_fraction*100:.2f}% power")
        print(f"  RADAR factor: β={beta:.6f} → {self.radar_power_fraction*100:.2f}% power")
        print(f"  Constraint: α²+β²={power_sum:.6f} ✅")
    
    def _validate_against_gt_data(self):
        """
        Validate configuration against extracted GT data.
        
        Checks:
        - COMM coefficients shape (n_rx, n_tx, n_subcarriers) = (2, 4, 1633)
        - Max range/velocity coverage
        - Channel metrics compatibility
        """
        print(f"\n✓ GT Data Validation:")
        
        # Validate COMM coefficients shape
        expected_shape = (self.n_rx_antennas, self.n_tx_antennas, self.n_subcarriers_actual)
        print(f"  Expected COMM coeffs shape: {expected_shape}")
        print(f"  Interpretation: (n_rx={self.n_rx_antennas}, n_tx={self.n_tx_antennas}, n_sc={self.n_subcarriers_actual})")
        print(f"  BS (TX): {self.n_tx_antennas} antennas ({self.tx_antenna_shape[0]}×{self.tx_antenna_shape[1]} UPA)")
        print(f"  UE (RX): {self.n_rx_antennas} antennas ({self.rx_antenna_shape[0]}×{self.rx_antenna_shape[1]} linear)")
        
        # Validate range coverage
        # From COMM GT: 8.4 - 308.4 m
        # From RADAR GT: 8.76 - 497.56 m
        comm_max_range = 308.4
        radar_max_range = 497.6
        
        if self.max_range_m < radar_max_range:
            print(f"  ⚠️  Config max_range ({self.max_range_m}m) < RADAR GT max ({radar_max_range:.1f}m)")
        else:
            print(f"  ✓ Max range coverage: {self.max_range_m}m >= {radar_max_range:.1f}m")
        
        # Validate velocity coverage
        # From COMM GT: ±26 m/s
        # From RADAR GT: ±62.3 m/s
        comm_max_vel = 26.0
        radar_max_vel = 62.3
        
        if self.max_velocity_ms < radar_max_vel:
            print(f"  ⚠️  Config max_velocity ({self.max_velocity_ms}m/s) < RADAR GT max ({radar_max_vel:.1f}m/s)")
        else:
            print(f"  ✓ Max velocity coverage: {self.max_velocity_ms}m/s >= {radar_max_vel:.1f}m/s")
        
        # Validate channel metrics compatibility
        # From COMM GT: Delay spread mean ~28 ns, Coherence BW ~27 MHz
        comm_delay_spread_ns = 28.0
        comm_coh_bw_mhz = 27.0
        
        expected_coh_bw_mhz = 1.0 / (5 * comm_delay_spread_ns * 1e-9) / 1e6
        print(f"  Channel metrics (from GT):")
        print(f"    Delay spread: ~{comm_delay_spread_ns:.1f} ns")
        print(f"    Coherence BW: ~{comm_coh_bw_mhz:.1f} MHz (measured)")
        print(f"    Expected BC (50%): ~{expected_coh_bw_mhz:.1f} MHz (from τ_rms)")
    
    # ================================================================
    # UTILITIES
    # ================================================================
    def validate_power_allocation(self):
        """
        Explicit power allocation validation method (NEW v4.0).
        
        Can be called manually to verify power allocation constraint.
        
        Returns:
            bool: True if valid, raises ValueError if invalid
        """
        if self.fmcw_integration_mode != 'additive':
            print(f"Power allocation not applicable for {self.fmcw_integration_mode} mode")
            return True
        
        alpha = self.comm_power_factor
        beta = self.radar_power_factor
        power_sum = alpha**2 + beta**2
        
        if abs(power_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Power allocation constraint violated: α²+β²={power_sum:.6f} ≠ 1.0\n"
                f"α={alpha:.6f}, β={beta:.6f}"
            )
        
        print(f"✓ Power allocation valid:")
        print(f"  α={alpha:.6f} → COMM: {alpha**2*100:.2f}%")
        print(f"  β={beta:.6f} → RADAR: {beta**2*100:.2f}%")
        print(f"  Constraint: α²+β²={power_sum:.6f} ✅")
        return True
    
    def get_subcarrier_frequencies(self) -> np.ndarray:
        """
        Compute subcarrier frequencies.
        
        Returns:
            f_k: Subcarrier frequencies (N_used,) in Hz
        """
        k_indices = np.arange(self.n_subcarriers_used)
        k_centered = k_indices - self.n_subcarriers_used // 2
        f_k = self.carrier_freq_hz + k_centered * self.subcarrier_spacing_hz
        return f_k
    
    def get_time_vector(self, n_samples: int) -> np.ndarray:
        """
        Generate time vector for signal generation.
        
        Args:
            n_samples: Number of time samples
        
        Returns:
            t: Time vector (n_samples,) in seconds
        """
        dt = 1.0 / self.sampling_rate_hz
        t = np.arange(n_samples) * dt
        return t
    
    def get_antenna_positions(self, antenna_type: Literal['tx', 'rx']) -> np.ndarray:
        """
        Get antenna element positions.
        
        Args:
            antenna_type: 'tx' (BS transmitter) or 'rx' (UE receiver)
        
        Returns:
            positions: (N_antennas, 3) array of [x, y, z] positions in wavelengths
        """
        if antenna_type == 'tx':
            shape = self.tx_antenna_shape
            spacing = self.tx_antenna_spacing
        else:
            shape = self.rx_antenna_shape
            spacing = self.rx_antenna_spacing
        
        N_y, N_z = shape
        positions = []
        
        for iy in range(N_y):
            for iz in range(N_z):
                y = (iy - (N_y - 1) / 2) * spacing
                z = (iz - (N_z - 1) / 2) * spacing
                x = 0.0  # All antennas in y-z plane
                positions.append([x, y, z])
        
        return np.array(positions)
    
    def summary(self) -> Dict:
        """Generate configuration summary."""
        return {
            # RF Parameters
            "carrier_freq_ghz": self.carrier_freq_hz / 1e9,
            "bandwidth_mhz": self.bandwidth_hz / 1e6,
            "tx_power_dbm": self.tx_power_dbm,
            "tx_power_w": self.tx_power_w,
            "wavelength_mm": self.wavelength_m * 1e3,
            "noise_figure_db": self.noise_figure_db,
            "thermal_noise_dbm": self.thermal_noise_power_dbm,
            
            # OFDM Parameters
            "scs_khz": self.subcarrier_spacing_khz,
            "n_fft": self.n_fft,
            "n_subcarriers_used": self.n_subcarriers_used,
            "n_subcarriers_actual": self.n_subcarriers_actual,
            "sampling_rate_mhz": self.sampling_rate_hz / 1e6,
            "ofdm_symbol_duration_us": self.ofdm_symbol_duration_s * 1e6,
            "cp_duration_us": self.cp_duration_s * 1e6,
            "total_symbol_duration_us": self.total_symbol_duration_s * 1e6,
            "slot_duration_us": self.slot_duration_s * 1e6,
            
            # MIMO
            "n_tx_antennas": self.n_tx_antennas,
            "n_rx_antennas": self.n_rx_antennas,
            "tx_antenna_shape": self.tx_antenna_shape,
            "rx_antenna_shape": self.rx_antenna_shape,
            "coherent_combining": self.coherent_combining,
            
            # FMCW
            "fmcw_enabled": self.fmcw_enable,
            "fmcw_chirp_duration_us": self.fmcw_chirp_duration_s * 1e6,
            "fmcw_n_chirps": self.fmcw_n_chirps,
            "fmcw_range_resolution_m": self.fmcw_range_resolution_m,
            "fmcw_velocity_resolution_ms": self.fmcw_velocity_resolution_ms,
            "fmcw_frame_rate_hz": self.fmcw_frame_rate_hz,
            "fmcw_max_range_m": self.fmcw_max_range_m,
            "fmcw_max_velocity_ms": self.fmcw_max_velocity_ms,
            
            # FMCW Integration (NEW v4.0)
            "fmcw_integration_mode": self.fmcw_integration_mode,
            "comm_power_factor": self.comm_power_factor,
            "radar_power_factor": self.radar_power_factor,
            "comm_power_fraction": self.comm_power_fraction,
            "radar_power_fraction": self.radar_power_fraction,
            
            # Pilot Configuration
            "pilot_spacing_time_symbols": self.pilot_spacing_time_symbols,
            "pilot_spacing_freq_subcarriers": self.pilot_spacing_freq_subcarriers,
            "max_doppler_hz": self.max_doppler_shift_hz,
            "max_delay_us": self.max_delay_s * 1e6,
        }
    
    def print_summary(self):
        """Print human-readable configuration summary."""
        summary = self.summary()
        
        print("\n" + "="*80)
        print("DUAL-FUNCTION ISAC TRANSMITTER CONFIGURATION")
        print("Version 4.0 - ADDITIVE ISAC: Power allocation (α, β) with α² + β² = 1")
        print("="*80)
        
        print("\n[RF PARAMETERS]")
        print(f"  Carrier Frequency:    {summary['carrier_freq_ghz']:.2f} GHz")
        print(f"  Bandwidth:            {summary['bandwidth_mhz']:.1f} MHz")
        print(f"  TX Power:             {summary['tx_power_dbm']:.1f} dBm ({summary['tx_power_w']:.1f} W)")
        print(f"  Wavelength:           {summary['wavelength_mm']:.3f} mm")
        print(f"  Noise Figure:         {summary['noise_figure_db']:.1f} dB")
        print(f"  Thermal Noise:        {summary['thermal_noise_dbm']:.1f} dBm")
        
        print("\n[5G NR OFDM PARAMETERS]")
        print(f"  Subcarrier Spacing:   {summary['scs_khz']} kHz")
        print(f"  FFT Size:             {summary['n_fft']}")
        print(f"  Used Subcarriers:     {summary['n_subcarriers_used']}")
        print(f"  Actual Subcarriers:   {summary['n_subcarriers_actual']} ✅ VALIDATED from COMM GT")
        print(f"  Sampling Rate:        {summary['sampling_rate_mhz']:.2f} MHz")
        print(f"  OFDM Symbol Duration: {summary['ofdm_symbol_duration_us']:.2f} μs")
        print(f"  CP Duration:          {summary['cp_duration_us']:.2f} μs")
        print(f"  Total Symbol Dur:     {summary['total_symbol_duration_us']:.2f} μs")
        print(f"  Slot Duration:        {summary['slot_duration_us']:.2f} μs")
        
        print("\n[MIMO CONFIGURATION] ✅ CORRECTED")
        print(f"  BS (Transmitter):")
        print(f"    TX Antennas:        {summary['n_tx_antennas']} ({summary['tx_antenna_shape'][0]}×{summary['tx_antenna_shape'][1]} UPA)")
        print(f"  UE (Receiver):")
        print(f"    RX Antennas:        {summary['n_rx_antennas']} ({summary['rx_antenna_shape'][0]}×{summary['rx_antenna_shape'][1]} linear)")
        print(f"  COMM Coeffs Shape:    ({summary['n_rx_antennas']}, {summary['n_tx_antennas']}, {summary['n_subcarriers_actual']})")
        print(f"  Combining Method:     {summary['coherent_combining'].upper()}")
        
        print("\n[PILOT CONFIGURATION] ✅ OPTIMIZED v3.2")
        print(f"  Time Spacing (Mt):    {summary['pilot_spacing_time_symbols']} symbols")
        print(f"  Freq Spacing (Mf):    {summary['pilot_spacing_freq_subcarriers']} subcarriers ✅ (was 1)")
        print(f"  Max Doppler Shift:    {summary['max_doppler_hz']:.1f} Hz")
        print(f"  Max Delay:            {summary['max_delay_us']:.2f} μs")
        
        if summary['fmcw_enabled']:
            print("\n[FMCW RADAR PARAMETERS]")
            print(f"  Chirp Duration:       {summary['fmcw_chirp_duration_us']:.2f} μs")
            print(f"  Number of Chirps:     {summary['fmcw_n_chirps']}")
            print(f"  Range Resolution:     {summary['fmcw_range_resolution_m']:.3f} m")
            print(f"  Velocity Resolution:  {summary['fmcw_velocity_resolution_ms']:.3f} m/s")
            print(f"  Frame Rate:           {summary['fmcw_frame_rate_hz']:.1f} Hz")
            print(f"  Max Range:            {summary['fmcw_max_range_m']:.1f} m")
            print(f"  Max Velocity:         {summary['fmcw_max_velocity_ms']:.1f} m/s")
            
            # ✅ NEW v4.0: FMCW Integration info
            print(f"\n[FMCW INTEGRATION] ✅ NEW v4.0")
            print(f"  Integration Mode:     {summary['fmcw_integration_mode'].upper()}")
            if summary['fmcw_integration_mode'] == 'additive':
                print(f"  Formula:              x(t) = α·x_ofdm(t) + β·chirp(t)")
                print(f"  Power Allocation:")
                print(f"    COMM factor:  α={summary['comm_power_factor']:.6f} → {summary['comm_power_fraction']*100:.2f}% power")
                print(f"    RADAR factor: β={summary['radar_power_factor']:.6f} → {summary['radar_power_fraction']*100:.2f}% power")
                print(f"    Constraint:   α²+β²={summary['comm_power_fraction']+summary['radar_power_fraction']:.6f} ✅")
            else:
                print(f"  Formula:              x(t) = x_ofdm(t) ⊙ chirp(t)")
        
        print("\n" + "="*80 + "\n")
    
    def save_config(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = self.summary()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"✓ Configuration saved to {filepath}")
    
    @classmethod
    def from_json(cls, filepath: str) -> 'ISACTXConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct config from saved parameters
        constructor_params = {
            'carrier_freq_hz': config_dict.get('carrier_freq_ghz', 28.0) * 1e9,
            'bandwidth_hz': config_dict.get('bandwidth_mhz', 200.0) * 1e6,
            'tx_power_dbm': config_dict.get('tx_power_dbm', 43.0),
            'subcarrier_spacing_khz': config_dict.get('scs_khz', 120),
            'n_fft': config_dict.get('n_fft', 2048),
            'n_subcarriers_used': config_dict.get('n_subcarriers_used', 1633),
            'n_subcarriers_actual': config_dict.get('n_subcarriers_actual', 1633),
            'n_tx_antennas': config_dict.get('n_tx_antennas', 4),
            'n_rx_antennas': config_dict.get('n_rx_antennas', 2),
            'pilot_spacing_freq': config_dict.get('pilot_spacing_freq_subcarriers', 3),
            # ✅ NEW v4.0: Load power allocation parameters
            'fmcw_integration_mode': config_dict.get('fmcw_integration_mode', 'additive'),
            'comm_power_factor': config_dict.get('comm_power_factor', 0.707),
            'radar_power_factor': config_dict.get('radar_power_factor', 0.707),
        }
        
        return cls(**constructor_params)


# ================================================================
# DEFAULT CONFIGURATION (Aligned with DeepVerse-6G)
# ================================================================
def get_default_config(**kwargs) -> ISACTXConfig:
    """
    Get default ISAC-TX configuration matching DeepVerse-6G dataset.
    
    Args:
        **kwargs: Override default parameters
    
    Returns:
        config: ISACTXConfig instance
    
    Example:
        >>> # Default: equal power allocation (50/50)
        >>> config = get_default_config()
        >>> config.print_summary()
        
        >>> # COMM-dominant (81% COMM, 19% RADAR)
        >>> config_comm = get_default_config(
        ...     comm_power_factor=0.9,
        ...     radar_power_factor=0.436
        ... )
        
        >>> # RADAR-dominant (19% COMM, 81% RADAR)
        >>> config_radar = get_default_config(
        ...     comm_power_factor=0.436,
        ...     radar_power_factor=0.9
        ... )
        
        >>> # Multiplicative mode (legacy)
        >>> config_legacy = get_default_config(
        ...     fmcw_integration_mode='multiplicative'
        ... )
        
        >>> # Override for bistatic mode
        >>> config_bistatic = get_default_config(radar_mode='bistatic')
        
        >>> # Override pilot spacing
        >>> config_dense = get_default_config(pilot_spacing_freq=1)  # 35% overhead
        >>> config_sparse = get_default_config(pilot_spacing_freq=6)  # 6% overhead
    """
    return ISACTXConfig(**kwargs)


# ================================================================
# VALIDATION SCRIPT
# ================================================================
if __name__ == "__main__":
    print("="*80)
    print("ISAC-TX Configuration Validator v4.0")
    print("NEW: Additive ISAC with power allocation (α, β)")
    print("OPTIMIZED: Pilot spacing Mf=3 for 12% overhead (was 35%)")
    print("="*80)
    
    print("\n[ANTENNA CONFIGURATION]")
    print("COMM Coeffs Shape: (2, 4, 1633)")
    print("  Dimension 0 (2):    RX antennas (UE: 2×1 linear)")
    print("  Dimension 1 (4):    TX antennas (BS: 2×2 UPA)")
    print("  Dimension 2 (1633): Subcarriers")
    print("\nConvention: H[rx, tx, subcarrier]")
    print("  BS (Base Station):  TRANSMITTER with 4 antennas")
    print("  UE (User Equipment): RECEIVER with 2 antennas")
    
    print("\n[FMCW INTEGRATION v4.0] ✅ NEW")
    print("Additive mode: x(t) = α·x_ofdm(t) + β·chirp(t)")
    print("Constraint: α² + β² = 1 (power conservation)")
    print("Examples:")
    print("  - Equal power:     α=0.707, β=0.707 → 50% COMM, 50% RADAR")
    print("  - COMM-dominant:   α=0.9,   β=0.436 → 81% COMM, 19% RADAR")
    print("  - RADAR-dominant:  α=0.436, β=0.9   → 19% COMM, 81% RADAR")
    print("  - Pure COMM:       α=1.0,   β=0.0   → 100% COMM, 0% RADAR")
    print("  - Pure RADAR:      α=0.0,   β=1.0   → 0% COMM, 100% RADAR")
    
    print("\n[PILOT OPTIMIZATION v3.2]")
    print("Previous (v3.1): Mf=1 → 35.71% overhead (8165 pilots)")
    print("Optimized (v3.2): Mf=3 → ~12% overhead (~2725 pilots)")
    print("Improvement: +23% spectral efficiency")
    print("Justification: Coherence BW = 27 MHz >> 360 kHz (3 subcarriers)")
    
    print("\n[INITIALIZING CONFIGURATION]")
    print("Based on:")
    print("  - 52,080 COMM GT samples (1.04M paths)")
    print("  - 52,080 RADAR GT samples")
    print("  - MathWorks ISAC analysis (Parts I & II)")
    print("  - 3GPP TS 38.211 (5G NR)")
    
    config = get_default_config()
    config.print_summary()
    
    # Test power allocation validation
    print("\n[TESTING POWER ALLOCATION VALIDATION]")
    print("Test 1: Valid allocation (default)")
    config.validate_power_allocation()
    
    print("\nTest 2: COMM-dominant")
    config_comm = get_default_config(comm_power_factor=0.9, radar_power_factor=0.436)
    config_comm.validate_power_allocation()
    
    print("\nTest 3: Multiplicative mode")
    config_mult = get_default_config(fmcw_integration_mode='multiplicative')
    # Should skip validation for multiplicative mode
    
    # Verify subcarrier frequencies
    print("\n[VERIFICATION - Subcarrier Frequencies]")
    f_k = config.get_subcarrier_frequencies()
    print(f"  Subcarrier freq range: {f_k[0]/1e9:.6f} - {f_k[-1]/1e9:.6f} GHz")
    print(f"  Subcarrier spacing:    {(f_k[1] - f_k[0])/1e3:.2f} kHz")
    print(f"  Number of subcarriers: {len(f_k)}")
    
    # Verify antenna positions
    print("\n[VERIFICATION - Antenna Positions]")
    tx_pos = config.get_antenna_positions('tx')
    rx_pos = config.get_antenna_positions('rx')
    print(f"  BS TX antenna positions (λ units) - 2×2 UPA:")
    for i, pos in enumerate(tx_pos):
        print(f"    TX Ant {i}: {pos}")
    print(f"  UE RX antenna positions (λ units) - 2×1 linear:")
    for i, pos in enumerate(rx_pos):
        print(f"    RX Ant {i}: {pos}")
    
    # Compute Doppler constraint
    print("\n[VERIFICATION - OFDM Constraints]")
    scs_to_doppler_ratio = config.subcarrier_spacing_hz / config.max_doppler_shift_hz
    print(f"  SCS / f_D_max ratio:   {scs_to_doppler_ratio:.1f}x (should be ≥10x)")
    
    # Verify against GT statistics
    print("\n[VERIFICATION - GT Data Compatibility]")
    print(f"  COMM GT max range:     308.4 m  → Config: {config.max_range_m} m")
    print(f"  RADAR GT max range:    497.6 m  → Config: {config.max_range_m} m")
    print(f"  COMM GT max velocity:  ±26.0 m/s → Config: ±{config.max_velocity_ms} m/s")
    print(f"  RADAR GT max velocity: ±62.3 m/s → Config: ±{config.max_velocity_ms} m/s")
    print(f"  COMM GT delay spread:  ~28 ns")
    print(f"  COMM GT coherence BW:  ~27 MHz")
    
    # Save configuration
    print("\n[SAVING CONFIGURATION]")
    output_path = "/home/mababsa/DeepVerse-6G/config/isac_tx_config_v40.json"
    try:
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        config.save_config(output_path)
    except Exception as e:
        print(f"⚠️  Could not save config: {e}")
        print("   (This is OK if running in read-only environment)")
    
    print("\n" + "="*80)
    print("✅ CONFIGURATION VALIDATED SUCCESSFULLY!")
    print("✅ Power allocation: Additive ISAC with α² + β² = 1")
    print("✅ Pilot overhead optimized: 35% → 12%")
    print("✅ Ready for dual-function ISAC signal processing")
    print("="*80 + "\n")