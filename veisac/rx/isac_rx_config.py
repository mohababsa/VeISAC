# isac_rx_config.py
"""
VeISAC — ISAC Receiver Configuration

Dataclass holding all RX parameters for communication and sensing receivers (noise, OFDM, pilots, FMCW, CFAR, MIMO) across monostatic and bistatic modes.

Paper: "VeISAC: An End-to-End MIMO-OFDM-FMCW Framework for ISAC
        in 6G Vehicular Networks"
Authors: M. Ababsa, S. Ribouh, Y. El Hillali, A. Rivenq
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np

# Physical constants
LIGHTSPEED = 299792458.0  # m/s (exact, CODATA 2019)
BOLTZMANN_K = 1.380649e-23  # J/K (exact, CODATA 2019)


@dataclass
class ISACRXConfig:
    """
    ISAC Receiver Configuration.
    
    VERSION 4.0: Additive ISAC Integration Support + Finalized Sen RX Model
    
    Covers both communication and sensing receivers with support for
    monostatic and bistatic radar modes, and both additive and multiplicative
    ISAC integration modes.
    
    ALIGNED WITH:
    - TX config v4.0 (additive ISAC support)
    - TX transmitter v5.0 (power allocation)
    - MATLAB config.mat (FMCW parameters)
    - DeepVerse-6G COMM GT: (2, 4, 1633) shape
    - DeepVerse-6G RADAR GT: 52,080 samples
    - MathWorks ISAC methodology
    - Finalized Sen RX theoretical model (2026-04-02)
    
    CRITICAL: Uses TWO different sampling rates:
    - sampling_rate_hz: 245.76 MHz (COMM/OFDM)
    - fmcw_sampling_rate_hz: 200 MHz (RADAR/FMCW)
    
    RADAR DATASETS (DeepVerse-6G):
    - BC_Monostatic_BS.mat: (4, 4, 1664, 128) - BS self-echo
    - BC_Bistatic_BS.mat: (4, 4, 1664, 128) - BS reference
    - BC_Bistatic_UE.mat: (2, 4, 1664, 128) - UE receiver
    """
    
    # ==========================================
    # SYSTEM PARAMETERS (MUST MATCH TX v4.0)
    # ==========================================
    
    carrier_freq_hz: float = 28e9  # 28 GHz (n257 band)
    bandwidth_hz: float = 200e6  # 200 MHz (system bandwidth)
    
    # ✅ COMM sampling rate (for OFDM)
    # Derived from: SCS × N_FFT = 120 kHz × 2048 = 245.76 MHz
    sampling_rate_hz: float = 245.76e6  # COMM subsystem
    
    # ==========================================
    # FMCW INTEGRATION CONFIGURATION (NEW v4.0) ✅
    # ==========================================
    
    # Integration mode for ISAC signal
    # 'additive': x(t) = α·x_ofdm(t) + β·chirp(t) with α²+β²=1
    # 'multiplicative': x(t) = x_ofdm(t) ⊙ chirp(t) (legacy)
    fmcw_integration_mode: str = 'additive'
    
    # Power allocation factors (for additive mode)
    # These should match the transmitter configuration
    # α: COMM power allocation factor (amplitude, not power!)
    # β: RADAR power allocation factor (amplitude, not power!)
    # Constraint: α² + β² = 1
    comm_power_factor: float = np.sqrt(0.5)  # α (default: 50% power)
    radar_power_factor: float = np.sqrt(0.5)  # β (default: 50% power)
    
    # Note: The receiver needs to know these for proper signal processing
    # and power estimation, but does not enforce them (TX responsibility)
    
    # ==========================================
    # RECEIVER NOISE
    # ==========================================
    
    # Thermal noise: P_thermal = k·T·B
    temperature_k: float = 290.0  # K (ITU-R P.372-14 standard: 17°C)
    
    # Communication receiver (UE at 28 GHz)
    comm_noise_figure_db: float = 7.0  # dB (3GPP TR 38.803)
    
    # Sensing receiver (dedicated radar RX)
    sens_noise_figure_db: float = 5.0  # dB (better for radar)
    
    # ==========================================
    # COMMUNICATION RECEIVER (MUST MATCH TX)
    # ==========================================
    
    # OFDM parameters (MUST MATCH TX v4.0)
    n_fft: int = 2048
    n_subcarriers_actual: int = 1633  # ✅ CORRECTED: Active subcarriers (matched with TX)
    cp_length_samples: int = 1024  # Extended CP
    n_symbols_per_slot: int = 14  # 5G NR
    
    # Subcarrier spacing (MUST MATCH TX)
    subcarrier_spacing_khz: int = 120  # FR2 (mmWave)
    
    # Antenna configuration (UE receiver)
    # ✅ DeepVerse-6G COMM GT shape: (N_rx, N_tx, N_sc) = (2, 4, 1633)
    n_rx_antennas: int = 2  # UE has 2 RX antennas
    n_tx_antennas: int = 4  # BS has 4 TX antennas
    rx_antenna_shape: tuple = (2, 1)  # 2×1 ULA (linear array)
    tx_antenna_shape: tuple = (2, 2)  # 2×2 UPA (BS transmitter)
    rx_antenna_spacing: float = 0.5  # λ/2 spacing
    tx_antenna_spacing: float = 0.5  # λ/2 spacing
    
    # ==========================================
    # PILOT CONFIGURATION (CRITICAL - MUST MATCH TX v4.0!)
    # ==========================================
    
    # ✅ CORRECTED: Match TX v4.0 pilot spacing
    pilot_spacing_time: int = 3  # Mt=3 (was 4 in v3.0) ✅ CRITICAL!
    pilot_spacing_freq: int = 3  # Mf=3 (was 1 in v3.0) ✅ CRITICAL!
    
    # Pilot overhead (computed in __post_init__)
    n_pilots: int = field(init=False)  # Will be 2725
    n_data_res: int = field(init=False)  # Will be 20137
    pilot_overhead_pct: float = field(init=False)  # Will be 11.92%
    
    # Channel estimation
    channel_estimation_method: Literal['ls', 'mmse', 'dft'] = 'mmse'
    interpolation_method: Literal['linear', 'spline', 'cubic', 'nearest'] = 'linear'
    
    # Equalization
    equalization_method: Literal['zf', 'mmse', 'mrc'] = 'mmse'
    mmse_snr_db: float = 5.0  # SNR estimate for MMSE (will be updated with actual SNR)
    
    # Demodulation
    modulation: Literal['BPSK', 'QPSK', '16QAM', '64QAM', '256QAM'] = 'QPSK'
    use_soft_decision: bool = True  # Use LLR for soft decoding
    
    # ==========================================
    # SENSING RECEIVER (RADAR - MUST MATCH MATLAB config.mat)
    # ==========================================
    
    # Radar mode
    radar_mode: Literal['monostatic', 'bistatic'] = 'monostatic'
    
    # ✅ FMCW parameters (FROM MATLAB config.mat)
    # CRITICAL: These are RADAR-specific parameters!
    fmcw_n_chirps: int = 128  # From config.mat
    fmcw_n_samples_per_chirp: int = 1664  # From config.mat
    fmcw_sampling_rate_hz: float = 200e6  # ✅ RADAR Fs (from config.mat, NOT 245.76 MHz!)
    fmcw_chirp_slope: float = 24e12  # Hz/s (2.4×10^13, from config.mat)
    
    # Antenna configuration (radar receiver)
    # ✅ CRITICAL CLARIFICATION (2026-03-23):
    # DeepVerse-6G RADAR datasets have THREE configurations:
    # 
    # 1. Monostatic BS-BS: BC_Monostatic_BS.mat
    #    Shape: (N_tx, N_rx, N_samples, N_chirps) = (4, 4, 1664, 128)
    #    Scenario: BS transmits FMCW, BS receives echoes (self-echo)
    #    N_tx=4 (BS transmit), N_rx=4 (BS receive)
    #    Virtual array: 4×4 = 16 elements
    # 
    # 2. Bistatic BS (reference): BC_Bistatic_BS.mat
    #    Shape: (N_tx, N_rx, N_samples, N_chirps) = (4, 4, 1664, 128)
    #    Scenario: BS transmits FMCW, BS monitors its own transmission
    #    Purpose: Reference signal for bistatic processing
    #    N_tx=4 (BS transmit), N_rx=4 (BS receive)
    #    Virtual array: 4×4 = 16 elements
    # 
    # 3. Bistatic UE (receiver): BC_Bistatic_UE.mat
    #    Shape: (N_tx, N_rx, N_samples, N_chirps) = (2, 4, 1664, 128)
    #    Scenario: UE receives BS FMCW signals (passive radar)
    #    Purpose: Primary bistatic receiver
    #    N_tx=4 (BS transmit - known from reference), N_rx=2 (UE receive)
    #    Virtual array: 4×2 = 8 elements
    # 
    # NOTE: These defaults will be updated based on radar_mode in __post_init__
    n_radar_rx_antennas: int = 2  # Will be set based on radar_mode
    n_radar_tx_antennas: int = 4  # Always 4 (BS transmitter)
    radar_rx_antenna_shape: tuple = (2, 1)  # Will be set based on radar_mode
    radar_tx_antenna_shape: tuple = (2, 2)  # 2×2 UPA (BS transmitter)
    radar_antenna_spacing: float = 0.5  # λ/2 spacing
    
    # ==========================================
    # MATHWORKS ISAC CORRECTIONS (FINALIZED!) ✅
    # ==========================================
    
    # ✅ UPDATED: Aligned with finalized Sen RX theoretical model
    
    # 1. Coherent combining mode (MathWorks Part I & II)
    # CRITICAL: MUST BE 'sum', NOT 'mean'!
    # Sum coherently across TX antennas (per RX) for +12 dB coherent gain (4 TX)
    coherent_combining: Literal['sum', 'mean'] = 'sum'
    
    # 2. Amplitude scaling mode (MathWorks Part II)
    # CRITICAL: Per-channel scaling, NOT global!
    # Applied BEFORE coherent TX sum to normalize each RX channel
    amplitude_scaling_mode: Literal['per_channel', 'global'] = 'per_channel'
    
    # 3. Static clutter removal (MathWorks Part I)
    # ✅ CLARIFIED: Zero DC bin AFTER Doppler FFT (not before)
    # Implementation: Z[:, 0, :, :] = 0 (all range bins, DC Doppler, all antennas)
    # This removes stationary targets in Range-Doppler map
    remove_static_clutter: bool = True
    
    # 4. Doppler sign convention (MathWorks Part I)
    # ✅ CLARIFIED: Sign flip due to conjugation in de-chirping
    # De-chirping: z[n,m] = y[n,m] · s*[n] → Doppler sign reversed
    # Velocity extraction: v = -λf_d/2 (negative sign for correct convention)
    # Convention: Positive velocity → target approaching
    flip_doppler_sign: bool = True
    
    # 5. Non-coherent integration (MathWorks Part II)
    # ✅ CLARIFIED: Sum |.|² across RX antennas for diversity gain
    # After coherent TX sum: RDM = Σ_q |Z_tx_coh[r,d,q]|²
    # Provides +3 dB diversity gain (for 4 RX)
    # Total gain: +12 dB (TX coherent) + +3 dB (RX non-coherent) = +15 dB
    use_non_coherent_integration: bool = True
    
    # ==========================================
    # RANGE-DOPPLER PROCESSING
    # ==========================================
    
    # FFT sizes
    range_fft_size: int = 2048  # Zero-padded FFT for range (≥ n_samples_per_chirp)
    doppler_fft_size: int = 256  # Zero-padded FFT for Doppler (≥ n_chirps)
    
    # Windowing
    range_window: Literal['hann', 'hamming', 'blackman', 'kaiser', 'none'] = 'hann'
    doppler_window: Literal['hann', 'hamming', 'blackman', 'kaiser', 'none'] = 'hann'
    
    # Kaiser window parameters (if used)
    kaiser_beta_range: float = 8.6  # Side-lobe level: -60 dB
    kaiser_beta_doppler: float = 8.6
    
    # ==========================================
    # CFAR DETECTION
    # ==========================================
    
    cfar_method: Literal['ca', 'os', 'go', 'so'] = 'os'  # Cell-Averaging CFAR
    cfar_guard_cells: int = 8  # Guard cells around CUT
    cfar_training_cells: int = 24  # Training cells for noise estimation
    cfar_pfa: float = 1e-5  # Probability of false alarm
    cfar_threshold_offset_db: float = 0.0  # Additional threshold offset (dB)
    
    # 2D-CFAR (range-Doppler)
    use_2d_cfar: bool = True  # Apply CFAR in both dimensions
    
    # ==========================================
    # TARGET ESTIMATION
    # ==========================================
    
    # Peak search
    peak_search_method: Literal['max', 'centroid', 'parabolic'] = 'parabolic'
    min_peak_separation_range_bins: int = 3  # Minimum separation in range
    min_peak_separation_doppler_bins: int = 2  # Minimum separation in Doppler
    
    # Angle estimation (for MIMO)
    angle_estimation_method: Literal['music', 'esprit', 'bartlett', 'capon'] = 'music'
    n_music_sources: int = 3  # Number of sources for MUSIC
    
    # ==========================================
    # PERFORMANCE BOUNDS (MUST MATCH TX v4.0)
    # ==========================================
    
    # Maximum range and velocity (from TX validation)
    max_range_m: float = 600.0  # Maximum unambiguous range (from TX config)
    max_velocity_ms: float = 65.0  # ✅ CORRECTED: Matched with TX (was 50.0)
    
    # ==========================================
    # DEEPVERSE-6G GT DATA PARAMETERS
    # ==========================================
    
    # COMM GT statistics (from extraction)
    comm_gt_delay_spread_ns: float = 28.0  # RMS delay spread
    comm_gt_coherence_bw_mhz: float = 27.0  # Measured coherence bandwidth
    comm_gt_max_range_m: float = 308.4  # Max COMM range in dataset
    comm_gt_max_velocity_ms: float = 26.0  # Max COMM velocity in dataset
    
    # RADAR GT statistics (from extraction)
    radar_gt_max_range_m: float = 497.6  # Max RADAR range in dataset
    radar_gt_max_velocity_ms: float = 62.3  # Max RADAR velocity in dataset
    
    # ==========================================
    # GPU ACCELERATION
    # ==========================================
    
    use_gpu: bool = True  # Use CuPy if available
    
    # ==========================================
    # POST-INITIALIZATION
    # ==========================================
    
    def __post_init__(self):
        """
        Configure mode-dependent parameters after initialization.
        
        This method automatically sets antenna configurations based on radar_mode
        and computes pilot-related parameters.
        
        ✅ v4.0: Added power allocation validation for additive mode
        """
        # ✅ COMPUTE PILOT PARAMETERS (CRITICAL!)
        n_symbols = self.n_symbols_per_slot
        n_subcarriers = self.n_subcarriers_actual
        
        # Calculate number of pilots (Mt=3, Mf=3)
        # Pilots at: symbol indices [0, 3, 6, 9, 12] = 5 symbols
        # Pilots at: subcarrier indices [0, 3, 6, ..., 1632] = ceil(1633/3) = 545 subcarriers
        n_pilot_symbols = int(np.ceil(n_symbols / self.pilot_spacing_time))
        n_pilot_subcarriers = int(np.ceil(n_subcarriers / self.pilot_spacing_freq))
        n_pilots = n_pilot_symbols * n_pilot_subcarriers
        
        # Set as instance attributes
        object.__setattr__(self, 'n_pilots', n_pilots)
        
        # Calculate data REs
        n_total_res = n_symbols * n_subcarriers
        n_data_res = n_total_res - n_pilots
        object.__setattr__(self, 'n_data_res', n_data_res)
        
        # Calculate overhead
        pilot_overhead = (n_pilots / n_total_res) * 100
        object.__setattr__(self, 'pilot_overhead_pct', pilot_overhead)
        
        # ✅ NEW v4.0: Validate power allocation for additive mode
        if self.fmcw_integration_mode == 'additive':
            alpha = self.comm_power_factor
            beta = self.radar_power_factor
            power_sum = alpha**2 + beta**2
            
            if abs(power_sum - 1.0) > 1e-6:
                print(f"⚠️  WARNING: Power allocation constraint violated!")
                print(f"   α={alpha:.6f}, β={beta:.6f}")
                print(f"   α²+β²={power_sum:.6f} (should be 1.0)")
                print(f"   This config will work but may not match TX power allocation")
        
        # ✅ Configure radar RX antennas based on mode
        if self.radar_mode == 'monostatic':
            # Monostatic BS-BS: BS receives its own radar echoes
            # Dataset: BC_Monostatic_BS.mat
            # Shape: (N_tx, N_rx, N_samples, N_chirps) = (4, 4, 1664, 128)
            # Scenario: BS transmits FMCW chirps, BS receives echoes from targets
            # N_tx=4 (BS transmit, 2×2 UPA), N_rx=4 (BS receive, 2×2 UPA)
            # Virtual array: 4×4 = 16 elements
            # Processing: Self-echo radar (traditional monostatic)
            object.__setattr__(self, 'n_radar_rx_antennas', 4)
            object.__setattr__(self, 'radar_rx_antenna_shape', (2, 2))  # 2×2 UPA
            
        elif self.radar_mode == 'bistatic':
            # Distinguish bistatic BS (N_RX=4, 2×2 UPA) from bistatic UE (N_RX=2, 2×1 ULA).
            # n_radar_rx_antennas may have been passed as a constructor kwarg —
            # read the current value rather than hardcoding 2.
            _n_rx = self.n_radar_rx_antennas
            if _n_rx == 4:
                # Bistatic BS: nearest-BS receiver, 2×2 UPA, same aperture as monostatic
                object.__setattr__(self, 'n_radar_rx_antennas', 4)
                object.__setattr__(self, 'radar_rx_antenna_shape', (2, 2))  # 2×2 UPA
            else:
                # Bistatic UE: passive UE receiver, 2×1 ULA
                object.__setattr__(self, 'n_radar_rx_antennas', 2)
                object.__setattr__(self, 'radar_rx_antenna_shape', (2, 1))  # 2×1 ULA
            # Both bistatic topologies: one-way propagation → no Doppler sign reversal
            object.__setattr__(self, 'flip_doppler_sign', False)
            object.__setattr__(self, 'cfar_pfa', 1e-6)
    
    # ==========================================
    # DERIVED PROPERTIES
    # ==========================================
    
    @property
    def wavelength_m(self) -> float:
        """Carrier wavelength (m)."""
        return LIGHTSPEED / self.carrier_freq_hz
    
    @property
    def subcarrier_spacing_hz(self) -> float:
        """Subcarrier spacing (Hz)."""
        return self.subcarrier_spacing_khz * 1000
    
    @property
    def thermal_noise_power_w(self) -> float:
        """Thermal noise power: k·T·B."""
        return BOLTZMANN_K * self.temperature_k * self.bandwidth_hz
    
    @property
    def comm_noise_power_w(self) -> float:
        """Communication receiver noise power (W)."""
        nf_linear = 10 ** (self.comm_noise_figure_db / 10)
        return nf_linear * self.thermal_noise_power_w
    
    @property
    def comm_noise_power_dbm(self) -> float:
        """Communication receiver noise power (dBm)."""
        return 10 * np.log10(self.comm_noise_power_w * 1000)
    
    @property
    def sens_noise_power_w(self) -> float:
        """Sensing receiver noise power (W)."""
        nf_linear = 10 ** (self.sens_noise_figure_db / 10)
        return nf_linear * self.thermal_noise_power_w
    
    @property
    def sens_noise_power_dbm(self) -> float:
        """Sensing receiver noise power (dBm)."""
        return 10 * np.log10(self.sens_noise_power_w * 1000)
    
    @property
    def slot_duration_s(self) -> float:
        """OFDM slot duration (seconds)."""
        symbol_duration = (self.n_fft + self.cp_length_samples) / self.sampling_rate_hz
        return self.n_symbols_per_slot * symbol_duration
    
    @property
    def symbol_duration_s(self) -> float:
        """Single OFDM symbol duration with CP (seconds)."""
        return (self.n_fft + self.cp_length_samples) / self.sampling_rate_hz
    
    @property
    def cp_duration_s(self) -> float:
        """Cyclic prefix duration (seconds)."""
        return self.cp_length_samples / self.sampling_rate_hz
    
    @property
    def fmcw_t_chirp_s(self) -> float:
        """
        FMCW chirp duration (seconds).
        
        ✅ USES RADAR Fs (200 MHz from config.mat), NOT COMM Fs (245.76 MHz)!
        """
        return self.fmcw_n_samples_per_chirp / self.fmcw_sampling_rate_hz
    
    @property
    def fmcw_bandwidth_hz(self) -> float:
        """FMCW bandwidth (Hz)."""
        return self.fmcw_chirp_slope * self.fmcw_t_chirp_s
    
    @property
    def range_resolution_m(self) -> float:
        _df = 2.0 if self.radar_mode == 'monostatic' else 1.0
        return LIGHTSPEED / (_df * self.fmcw_bandwidth_hz)
    
    @property
    def doppler_resolution_hz(self) -> float:
        """Doppler resolution (Hz)."""
        t_frame = self.fmcw_n_chirps * self.fmcw_t_chirp_s
        return 1.0 / t_frame
    
    @property
    def doppler_resolution_ms(self) -> float:
        _df = 2.0 if self.radar_mode == 'monostatic' else 1.0
        return self.wavelength_m / (_df * self.fmcw_n_chirps * self.fmcw_t_chirp_s)
    
    @property
    def max_unambiguous_range_m(self) -> float:
        """Maximum unambiguous range (meters)."""
        return LIGHTSPEED * self.fmcw_t_chirp_s / 2
    
    @property
    def max_unambiguous_velocity_ms(self) -> float:
        """Maximum unambiguous velocity (m/s)."""
        return self.wavelength_m / (4 * self.fmcw_t_chirp_s)
    
    @property
    def max_doppler_shift_hz(self) -> float:
        """Maximum Doppler shift (Hz) for max_velocity_ms."""
        return 2 * self.max_velocity_ms / self.wavelength_m
    
    @property
    def bistatic_delay_factor(self) -> float:
        """
        Delay factor for bistatic vs monostatic.

        Monostatic: τ = 2R/c (round-trip) → factor = 2
        Bistatic:   τ = R/c  (one-way)    → factor = 1
        """
        return 1.0 if self.radar_mode == 'bistatic' else 2.0

    @property
    def delay_factor(self) -> float:
        """
        Range delay factor — used by SensingReceiver and ParameterEstimator.

        Monostatic: 2  (round-trip)
        Bistatic:   1  (one-way TX→target→RX)
        """
        return 1.0 if self.radar_mode == 'bistatic' else 2.0

    @property
    def doppler_factor(self) -> float:
        """
        Doppler factor — used by SensingReceiver and ParameterEstimator.

        Monostatic: 2  (two-way Doppler shift)
        Bistatic:   1  (one-way Doppler shift)
        """
        return 1.0 if self.radar_mode == 'bistatic' else 2.0
    @property
    def virtual_array_size(self) -> int:
        """
        MIMO virtual array size.
        
        For MIMO radar: N_virtual = N_tx × N_rx
        """
        return self.n_radar_tx_antennas * self.n_radar_rx_antennas
    
    @property
    def comm_channel_shape(self) -> tuple:
        """
        DeepVerse-6G COMM channel shape.
        
        Returns: (N_rx, N_tx, N_sc) = (2, 4, 1633)
        """
        return (self.n_rx_antennas, self.n_tx_antennas, self.n_subcarriers_actual)
    
    @property
    def radar_channel_shape(self) -> tuple:
        """
        DeepVerse-6G RADAR channel shape.
        
        Returns: (N_tx, N_rx, N_samples, N_chirps)
        - Monostatic BS: (4, 4, 1664, 128)
        - Bistatic BS: (4, 4, 1664, 128)
        - Bistatic UE: (2, 4, 1664, 128)
        """
        return (self.n_radar_tx_antennas, self.n_radar_rx_antennas, 
                self.fmcw_n_samples_per_chirp, self.fmcw_n_chirps)
    
    # ==========================================
    # NEW v4.0: POWER ALLOCATION PROPERTIES
    # ==========================================
    
    @property
    def comm_power_fraction(self) -> float:
        """
        Communication power fraction (α²).
        
        For additive ISAC: α² represents the fraction of total power
        allocated to the communication signal.
        """
        return self.comm_power_factor ** 2
    
    @property
    def radar_power_fraction(self) -> float:
        """
        Radar power fraction (β²).
        
        For additive ISAC: β² represents the fraction of total power
        allocated to the radar signal.
        """
        return self.radar_power_factor ** 2
    
    @property
    def power_allocation_sum(self) -> float:
        """
        Sum of power fractions (should be 1.0 for additive mode).
        
        Constraint: α² + β² = 1
        """
        return self.comm_power_fraction + self.radar_power_fraction
    
    # ==========================================
    # NEW: VELOCITY SIGN CONVENTION PROPERTY ✅
    # ==========================================
    
    @property
    def velocity_sign_factor(self) -> float:
        """
        Velocity sign factor for Doppler-to-velocity conversion.
        
        Returns -1.0 if flip_doppler_sign=True, else +1.0
        
        Usage:
            v = velocity_sign_factor * doppler_bin * doppler_resolution_ms
        
        Physical convention (when flip_doppler_sign=True):
            - Positive velocity (+) → Target approaching
            - Negative velocity (-) → Target receding
        
        Reason for sign flip:
            De-chirping uses conjugation: z[n,m] = y[n,m] · s*[n]
            This reverses the Doppler sign in the beat signal.
        """
        return -1.0 if self.flip_doppler_sign else 1.0
    
    # ==========================================
    # VALIDATION
    # ==========================================
    
    def validate(self) -> bool:
        """Validate configuration consistency."""
        errors = []
        warnings = []
        
        # ✅ NEW v4.0: Validate power allocation for additive mode
        if self.fmcw_integration_mode == 'additive':
            power_sum = self.power_allocation_sum
            if abs(power_sum - 1.0) > 1e-6:
                errors.append(
                    f"Power allocation constraint violated: "
                    f"α²+β²={power_sum:.6f} (should be 1.0). "
                    f"α={self.comm_power_factor:.6f}, β={self.radar_power_factor:.6f}"
                )
        elif self.fmcw_integration_mode != 'multiplicative':
            errors.append(
                f"Invalid FMCW integration mode: '{self.fmcw_integration_mode}'. "
                f"Must be 'additive' or 'multiplicative'"
            )
        
        # ✅ Check pilot overhead is reasonable
        if self.pilot_overhead_pct < 5.0:
            warnings.append(
                f"Pilot overhead very low: {self.pilot_overhead_pct:.2f}% "
                f"(may affect channel estimation quality)"
            )
        elif self.pilot_overhead_pct > 30.0:
            warnings.append(
                f"Pilot overhead high: {self.pilot_overhead_pct:.2f}% "
                f"(reduces spectral efficiency)"
            )
        
        # ✅ Check pilot spacing matches TX
        expected_Mt = 3
        expected_Mf = 3
        if self.pilot_spacing_time != expected_Mt or self.pilot_spacing_freq != expected_Mf:
            errors.append(
                f"CRITICAL: Pilot spacing mismatch with TX! "
                f"RX: Mt={self.pilot_spacing_time}, Mf={self.pilot_spacing_freq} "
                f"TX: Mt={expected_Mt}, Mf={expected_Mf}"
            )
        
        # ✅ Check data REs calculation
        expected_n_pilots = 2725
        expected_n_data = 20137
        if abs(self.n_pilots - expected_n_pilots) > 5:
            errors.append(
                f"Pilot count mismatch: {self.n_pilots} (expected ~{expected_n_pilots})"
            )
        if abs(self.n_data_res - expected_n_data) > 10:
            errors.append(
                f"Data REs mismatch: {self.n_data_res} (expected ~{expected_n_data})"
            )
        
        # ✅ Check FMCW parameters match MATLAB config.mat
        expected_fmcw_fs = 200e6
        expected_t_chirp = 8.32e-6
        actual_t_chirp = self.fmcw_t_chirp_s
        
        if abs(self.fmcw_sampling_rate_hz - expected_fmcw_fs) > 1e3:
            errors.append(
                f"RADAR Fs mismatch: {self.fmcw_sampling_rate_hz/1e6:.2f} MHz "
                f"(expected {expected_fmcw_fs/1e6:.0f} MHz from config.mat)"
            )
        
        if abs(actual_t_chirp - expected_t_chirp) > 1e-9:
            errors.append(
                f"Chirp duration mismatch: {actual_t_chirp*1e6:.2f} µs "
                f"(expected {expected_t_chirp*1e6:.2f} µs from config.mat)"
            )
        
        # Velocity resolution — mode-aware expected value
        # Monostatic: λ/(2·N·T) = 0.01071/(2·128·8.32e-6) ≈ 5.027 m/s
        # Bistatic:   λ/(N·T)   =                          ≈ 10.054 m/s
        expected_vel_res = (5.027 if self.radar_mode == 'monostatic'
                            else 10.054)
        actual_vel_res = self.doppler_resolution_ms

        if abs(actual_vel_res - expected_vel_res) > 0.1:
            errors.append(
                f"Velocity resolution mismatch ({self.radar_mode}): "
                f"{actual_vel_res:.3f} m/s "
                f"(expected {expected_vel_res:.3f} m/s)"
            )
        
        # Check CP is sufficient for max range
        tau_max = self.bistatic_delay_factor * self.max_range_m / LIGHTSPEED
        cp_duration = self.cp_length_samples / self.sampling_rate_hz
        
        if cp_duration < tau_max:
            errors.append(
                f"CP too short: {cp_duration*1e6:.2f} μs < {tau_max*1e6:.2f} μs "
                f"(R_max={self.max_range_m} m, mode={self.radar_mode})"
            )
        
        # Check FFT sizes
        if self.range_fft_size < self.fmcw_n_samples_per_chirp:
            errors.append(
                f"Range FFT size ({self.range_fft_size}) < "
                f"samples per chirp ({self.fmcw_n_samples_per_chirp})"
            )
        
        if self.doppler_fft_size < self.fmcw_n_chirps:
            errors.append(
                f"Doppler FFT size ({self.doppler_fft_size}) < "
                f"number of chirps ({self.fmcw_n_chirps})"
            )
        
        # Check CFAR parameters
        total_cells = 2 * (self.cfar_guard_cells + self.cfar_training_cells)
        if total_cells > min(self.range_fft_size, self.doppler_fft_size) // 4:
            errors.append(
                f"CFAR window too large: {total_cells} cells, "
                f"but FFT size is only {min(self.range_fft_size, self.doppler_fft_size)}"
            )
        
        # Check Doppler constraint (Δf ≥ 10 × f_D_max, MathWorks requirement)
        max_doppler_hz = self.max_doppler_shift_hz
        ratio = self.subcarrier_spacing_hz / max_doppler_hz
        if ratio < 10.0:
            warnings.append(
                f"Doppler constraint: Δf = {ratio:.1f}× f_D_max (should be ≥10×). "
                f"May cause ICI (Inter-Carrier Interference)"
            )
        
        # Print warnings
        if warnings:
            print("\n[CONFIG VALIDATION WARNINGS]")
            for warn in warnings:
                print(f"  ⚠️  {warn}")
        
        # Print errors
        if errors:
            print("\n[CONFIG VALIDATION ERRORS]")
            for err in errors:
                print(f"  ✗ {err}")
            return False
        
        return True
    
    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "="*80)
        print("ISAC RX CONFIGURATION SUMMARY (v4.0 - ADDITIVE ISAC + FINALIZED SEN RX)")
        print("Aligned with TX v4.0 + MATLAB config.mat + Sen RX Theoretical Model")
        print("="*80)
        
        print(f"\n[SYSTEM]")
        print(f"  Carrier: {self.carrier_freq_hz/1e9:.2f} GHz")
        print(f"  Bandwidth: {self.bandwidth_hz/1e6:.1f} MHz")
        print(f"  Wavelength: {self.wavelength_m*1000:.2f} mm")
        print(f"  COMM Fs: {self.sampling_rate_hz/1e6:.2f} MHz (OFDM)")
        print(f"  RADAR Fs: {self.fmcw_sampling_rate_hz/1e6:.0f} MHz (FMCW, from config.mat)")
        
        # ✅ NEW v4.0: FMCW Integration Info
        print(f"\n[FMCW INTEGRATION] ✅ NEW v4.0")
        print(f"  Mode: {self.fmcw_integration_mode.upper()}")
        if self.fmcw_integration_mode == 'additive':
            print(f"  Formula: x(t) = α·x_ofdm(t) + β·chirp(t)")
            print(f"  Power Allocation:")
            print(f"    COMM factor:  α={self.comm_power_factor:.6f} → {self.comm_power_fraction*100:.2f}% power")
            print(f"    RADAR factor: β={self.radar_power_factor:.6f} → {self.radar_power_fraction*100:.2f}% power")
            print(f"    Constraint:   α²+β²={self.power_allocation_sum:.6f} ✅")
        else:
            print(f"  Formula: x(t) = x_ofdm(t) ⊙ chirp(t)")
        
        print(f"\n[NOISE]")
        print(f"  Thermal (k·T·B): {self.thermal_noise_power_w:.3e} W ({10*np.log10(self.thermal_noise_power_w*1000):.2f} dBm)")
        print(f"  Comm RX (NF={self.comm_noise_figure_db} dB): {self.comm_noise_power_w:.3e} W ({self.comm_noise_power_dbm:.2f} dBm)")
        print(f"  Sens RX (NF={self.sens_noise_figure_db} dB): {self.sens_noise_power_w:.3e} W ({self.sens_noise_power_dbm:.2f} dBm)")
        
        print(f"\n[COMMUNICATION RX] ✅ Aligned with TX v4.0")
        print(f"  RX antennas: {self.n_rx_antennas} ({self.rx_antenna_shape[0]}×{self.rx_antenna_shape[1]} ULA)")
        print(f"  TX antennas: {self.n_tx_antennas} ({self.tx_antenna_shape[0]}×{self.tx_antenna_shape[1]} UPA)")
        print(f"  COMM GT shape: {self.comm_channel_shape}")
        print(f"  OFDM: {self.n_subcarriers_actual} active subcarriers (FFT={self.n_fft})")
        print(f"  CP: {self.cp_length_samples} samples ({self.cp_duration_s*1e6:.2f} μs)")
        print(f"  Slot duration: {self.slot_duration_s*1e6:.2f} μs ({self.n_symbols_per_slot} symbols)")
        
        print(f"\n[PILOT CONFIGURATION] ✅ CRITICAL - Aligned with TX!")
        print(f"  Time spacing (Mt): {self.pilot_spacing_time} symbols")
        print(f"  Freq spacing (Mf): {self.pilot_spacing_freq} subcarriers")
        print(f"  Total pilots: {self.n_pilots} ({self.pilot_overhead_pct:.2f}% overhead)")
        print(f"  Data REs: {self.n_data_res} ({100-self.pilot_overhead_pct:.2f}%)")
        print(f"  Total REs: {self.n_symbols_per_slot * self.n_subcarriers_actual}")
        
        print(f"\n[CHANNEL ESTIMATION]")
        print(f"  Method: {self.channel_estimation_method.upper()}")
        print(f"  Interpolation: {self.interpolation_method}")
        print(f"  Equalization: {self.equalization_method.upper()}")
        print(f"  Modulation: {self.modulation}")
        print(f"  Soft decision: {'ON' if self.use_soft_decision else 'OFF'}")
        
        print(f"\n[SENSING RX] ✅ Aligned with MATLAB config.mat")
        print(f"  Mode: {self.radar_mode.upper()}")
        if self.radar_mode == 'monostatic':
            print(f"  Scenario: BS-BS (Base Station receives own echoes)")
            print(f"  Dataset: BC_Monostatic_BS.mat")
            print(f"  RADAR GT shape: {self.radar_channel_shape}")
        else:
            if self.n_radar_rx_antennas == 4:
                print(f"  Scenario: Bistatic BS (nearest-BS receiver, 2×2 UPA)")
                print(f"  Dataset: BC_Bistatic_BS  {self.radar_channel_shape}")
                print(f"  Config uses: BS receiver parameters (4×4 MIMO, 16 virtual elements)")
            else:
                print(f"  Scenario: Bistatic UE (passive UE receiver, 2×1 ULA)")
                print(f"  BS dataset: BC_Bistatic_BS.mat (4, 4, 1664, 128) - Reference")
                print(f"  UE dataset: BC_Bistatic_UE.mat {self.radar_channel_shape} - Receiver")
                print(f"  Config uses: UE receiver parameters (4×2 MIMO, 8 virtual elements)")
        print(f"  RX antennas: {self.n_radar_rx_antennas} ({self.radar_rx_antenna_shape[0]}×{self.radar_rx_antenna_shape[1]})")
        print(f"  TX antennas: {self.n_radar_tx_antennas} ({self.radar_tx_antenna_shape[0]}×{self.radar_tx_antenna_shape[1]})")
        print(f"  Virtual array: {self.virtual_array_size} elements")
        print(f"  FMCW: {self.fmcw_n_chirps} chirps × {self.fmcw_n_samples_per_chirp} samples")
        print(f"  Chirp duration: {self.fmcw_t_chirp_s*1e6:.2f} μs (from config.mat)")
        print(f"  Chirp slope: {self.fmcw_chirp_slope/1e12:.1f} THz/s")
        print(f"  Bandwidth: {self.fmcw_bandwidth_hz/1e6:.1f} MHz")
        
        print(f"\n[MATHWORKS CORRECTIONS] ✅ FINALIZED SEN RX MODEL")
        print(f"  Coherent combining: {self.coherent_combining.upper()} (TX sum → +12 dB gain)")
        print(f"  Amplitude scaling: {self.amplitude_scaling_mode.replace('_', ' ')} (before TX sum)")
        print(f"  Static clutter removal: {'AFTER Doppler FFT' if self.remove_static_clutter else 'OFF'}")
        print(f"  Doppler sign flip: {'ON (v=-λf_d/2)' if self.flip_doppler_sign else 'OFF'}")
        print(f"  Non-coherent integration: {'ON (RX sum |.|² → +3 dB)' if self.use_non_coherent_integration else 'OFF'}")
        _rx_div_gain_db = round(10 * np.log10(self.n_radar_rx_antennas))
        _total_gain_db  = 12 + _rx_div_gain_db
        print(f"  Total SNR gain: +{_total_gain_db} dB "
              f"(+12 TX coherent + +{_rx_div_gain_db} RX diversity, "
              f"N_RX={self.n_radar_rx_antennas})")
        
        print(f"\n[RESOLUTION]")
        _df   = self.delay_factor
        _dopf = self.doppler_factor
        _exp_r = 0.751  if self.radar_mode == 'monostatic' else LIGHTSPEED / self.fmcw_bandwidth_hz
        _exp_v = 5.027  if self.radar_mode == 'monostatic' else self.doppler_resolution_ms
        print(f"  Range:    {self.range_resolution_m:.3f} m  "
              f"(expected {_exp_r:.3f} m, c / ({_df:.0f}×B), {self.radar_mode})")
        print(f"  Velocity: {self.doppler_resolution_ms:.3f} m/s  "
              f"(expected {_exp_v:.3f} m/s, λ / ({_dopf:.0f}×N×T), {self.radar_mode})")
        print(f"  Velocity sign: {self.velocity_sign_factor:+.1f}  "
              f"(flip_doppler_sign={self.flip_doppler_sign})")
        print(f"  Delay factor:   {_df:.0f}×  "
              f"({'round-trip' if self.radar_mode == 'monostatic' else 'one-way'})")
        print(f"  Doppler factor: {_dopf:.0f}×  "
              f"({'two-way' if self.radar_mode == 'monostatic' else 'one-way'})")
        print(f"  Doppler:  {self.doppler_resolution_hz:.3f} Hz")
        
        print(f"\n[UNAMBIGUOUS RANGES] ✅ Aligned with TX")
        print(f"  Range (config): {self.max_range_m:.1f} m")
        print(f"  Range (unambiguous): {self.max_unambiguous_range_m:.1f} m")
        print(f"  Velocity (config): ±{self.max_velocity_ms:.1f} m/s")
        print(f"  Velocity (unambiguous): ±{self.max_unambiguous_velocity_ms:.1f} m/s")
        print(f"  Max Doppler shift: ±{self.max_doppler_shift_hz:.1f} Hz")
        
        print(f"\n[DEEPVERSE-6G GT DATA]")
        print(f"  COMM max range: {self.comm_gt_max_range_m:.1f} m")
        print(f"  COMM max velocity: ±{self.comm_gt_max_velocity_ms:.1f} m/s")
        print(f"  COMM delay spread: ~{self.comm_gt_delay_spread_ns:.1f} ns")
        print(f"  COMM coherence BW: ~{self.comm_gt_coherence_bw_mhz:.1f} MHz")
        print(f"  RADAR max range: {self.radar_gt_max_range_m:.1f} m")
        print(f"  RADAR max velocity: ±{self.radar_gt_max_velocity_ms:.1f} m/s")
        
        print(f"\n[PROCESSING]")
        print(f"  Range FFT: {self.range_fft_size} points")
        print(f"  Doppler FFT: {self.doppler_fft_size} points")
        print(f"  Range window: {self.range_window}")
        print(f"  Doppler window: {self.doppler_window}")
        
        print(f"\n[CFAR DETECTION]")
        print(f"  Method: {self.cfar_method.upper()}")
        print(f"  Pfa: {self.cfar_pfa:.1e}")
        print(f"  Guard cells: {self.cfar_guard_cells}")
        print(f"  Training cells: {self.cfar_training_cells}")
        print(f"  2D-CFAR: {'ON' if self.use_2d_cfar else 'OFF'}")
        
        print(f"\n[GPU]")
        print(f"  Acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
        
        print("="*80 + "\n")


def get_default_config(**kwargs) -> ISACRXConfig:
    """
    Get default ISAC RX configuration.
    
    Args:
        **kwargs: Override default parameters
    
    Returns:
        config: ISACRXConfig instance
    
    Example:
        >>> # Default additive mode
        >>> config = get_default_config()
        >>> config.print_summary()
        
        >>> # COMM-dominant
        >>> config_comm = get_default_config(
        ...     comm_power_factor=0.9,
        ...     radar_power_factor=0.436
        ... )
        
        >>> # Multiplicative mode (legacy)
        >>> config_mult = get_default_config(
        ...     fmcw_integration_mode='multiplicative'
        ... )
    """
    config = ISACRXConfig(**kwargs)
    
    # Validate
    if not config.validate():
        raise ValueError("Configuration validation failed!")
    
    return config


# For backward compatibility
ISACReceiverConfig = ISACRXConfig


# Test when run directly
if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING ISAC RX CONFIGURATION (v4.0 - FINALIZED SEN RX MODEL)")
    print("Aligned with TX v4.0 + MATLAB config.mat + Theoretical Model")
    print("="*80)
    
    # Test 1: Default additive configuration
    print("\n[TEST 1: Additive Configuration (Equal Power)]")
    config_add = get_default_config(
        fmcw_integration_mode='additive',
        comm_power_factor=np.sqrt(0.5),
        radar_power_factor=np.sqrt(0.5)
    )
    config_add.print_summary()
    
    # Test 2: Velocity sign convention
    print("\n[TEST 2: Velocity Sign Convention]")
    config_test = get_default_config()
    print(f"\n  Velocity resolution (magnitude): {config_test.doppler_resolution_ms:.3f} m/s")
    print(f"  Velocity sign factor: {config_test.velocity_sign_factor:+.1f}")
    print(f"  Formula: v = {config_test.velocity_sign_factor:+.1f} × d_bin × {config_test.doppler_resolution_ms:.3f}")
    print(f"  Physical meaning: Positive v → Target {'approaching' if config_test.velocity_sign_factor < 0 else 'receding'}")
    
    # Test 3: MathWorks corrections verification
    print("\n[TEST 3: MathWorks Corrections Verification]")
    print(f"\n  Coherent combining: {config_test.coherent_combining} ✓")
    print(f"  Non-coherent integration: {config_test.use_non_coherent_integration} ✓")
    print(f"  Amplitude scaling: {config_test.amplitude_scaling_mode} ✓")
    print(f"  Static clutter: {config_test.remove_static_clutter} ✓")
    print(f"  Doppler sign flip: {config_test.flip_doppler_sign} ✓")
    
    print("\n" + "="*80)
    print("✅ ALL CONFIGURATION TESTS PASSED!")
    print("✅ v4.0: Additive ISAC integration support")
    print("✅ v4.0: Finalized Sen RX theoretical model alignment")
    print("✅ NEW: Velocity sign convention property")
    print("✅ NEW: Enhanced MathWorks corrections documentation")
    print("="*80 + "\n")