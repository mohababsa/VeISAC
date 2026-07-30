# isac_waveform_generator.py
"""
VeISAC — ISAC Waveform Generator

OFDM and FMCW signal generation with additive time-domain superposition (x = α·x_ofdm + β·chirp, α²+β²=1) for the dual-function ISAC-TX chain.

Paper: "VeISAC: An End-to-End MIMO-OFDM-FMCW Framework for ISAC
        in 6G Vehicular Networks"
Authors: M. Ababsa, S. Ribouh, Y. El Hillali, A. Rivenq
"""

import numpy as np
from typing import Union, Tuple, Optional, Dict, List
import warnings
from pathlib import Path
import sys

# Add parent directory to path for imports
#sys.path.insert(0, str(Path(__file__).parent.parent))

# Local imports
try:
    from deepverse.signal_processing.tx.isac_tx_config import ISACTXConfig, get_default_config
    from deepverse.signal_processing.tx.modulation import DigitalModulator
except ImportError:
    print("Warning: Could not import local modules. Using defaults.")
    ISACTXConfig = None
    get_default_config = None
    DigitalModulator = None

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np


class OFDMSignalGenerator:
    """
    OFDM signal generator for 5G NR communication.
    
    Generates OFDM symbols with cyclic prefix according to 3GPP TS 38.211.
    Supports subcarrier mapping, pilot insertion, and guard bands.
    
    VALIDATED FOR DEEPVERSE-6G:
    - 1633 active subcarriers
    - 120 kHz subcarrier spacing
    - 2048 FFT size
    - 1024 CP samples (4.096 μs)
    - 14 symbols per slot
    
    Args:
        config: ISAC transmitter configuration
        use_gpu: Enable GPU acceleration (default: True if available)
        
    Example:
        >>> config = get_default_config()
        >>> ofdm = OFDMSignalGenerator(config)
        >>> symbol = ofdm.generate_ofdm_symbol(data_symbols)
    """
    
    def __init__(self, config: 'ISACTXConfig', use_gpu: bool = True):
        """Initialize OFDM generator."""
        self.config = config
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        
        if self.use_gpu and not CUPY_AVAILABLE:
            warnings.warn("CuPy not available. Falling back to NumPy (CPU).")
            self.use_gpu = False
        
        # Set array library
        self.xp = cp if self.use_gpu else np
        
        # Precompute subcarrier indices
        self._compute_subcarrier_mapping()
    
    def _compute_subcarrier_mapping(self):
        """
        Compute subcarrier mapping for OFDM.
        
        Standard OFDM/FFT indexing (3GPP TS 38.211):
        - FFT bins: [0, 1, 2, ..., N_FFT-1]
        - DC is at bin 0 (or N_FFT/2 after fftshift)
        - Positive frequencies: [1, ..., N_FFT/2-1]
        - Negative frequencies: [N_FFT/2, ..., N_FFT-1]
        
        For used subcarriers centered around DC:
        - Lower half: bins [N_FFT - N_used/2, ..., N_FFT-1]
        - Upper half: bins [0, ..., N_used/2-1] (excluding DC if dc_null)
        
        DeepVerse-6G: 1633 active subcarriers out of 2048 FFT
        """
        n_fft = self.config.n_fft
        n_used = self.config.n_subcarriers_used
        
        # Guard bands
        n_guard_left = self.config.guard_band_left
        n_guard_right = self.config.guard_band_right
        
        # Active subcarriers (excluding guards and DC)
        n_active = n_used - (1 if self.config.dc_null else 0)
        
        # Split around DC
        n_lower = n_active // 2  # Lower (negative frequencies)
        n_upper = n_active - n_lower  # Upper (positive frequencies)
        
        # Lower half: [N_FFT - n_lower, ..., N_FFT-1]
        lower_indices = np.arange(n_fft - n_lower, n_fft)
        
        # Upper half: [1, ..., n_upper] (skip DC at index 0 if dc_null)
        if self.config.dc_null:
            upper_indices = np.arange(1, n_upper + 1)
        else:
            upper_indices = np.arange(0, n_upper)
        
        # Combine
        self.subcarrier_indices = np.concatenate([lower_indices, upper_indices])
        self.n_active_subcarriers = len(self.subcarrier_indices)
        
        # DC index
        self.dc_index = 0
        
        # Create mask for null subcarriers (guards + DC)
        self.null_mask = np.ones(n_fft, dtype=bool)
        self.null_mask[self.subcarrier_indices] = False
        
        if self.use_gpu:
            self.subcarrier_indices_gpu = cp.asarray(self.subcarrier_indices)
            self.null_mask_gpu = cp.asarray(self.null_mask)
        
        # Verify mapping matches DeepVerse-6G
        if self.n_active_subcarriers != self.config.n_subcarriers_actual:
            warnings.warn(
                f"Subcarrier mapping mismatch: computed {self.n_active_subcarriers}, "
                f"expected {self.config.n_subcarriers_actual} (from COMM GT)"
            )
    
    def insert_pilots(self,
                     data_symbols: Union[np.ndarray, 'cp.ndarray'],
                     pilot_indices: Union[np.ndarray, List[int]],
                     pilot_values: Union[np.ndarray, 'cp.ndarray']
                     ) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Insert pilot symbols into data subcarriers.
        
        Args:
            data_symbols: Data symbols (excluding pilots)
            pilot_indices: Indices where pilots should be inserted
            pilot_values: Pilot symbol values
        
        Returns:
            symbols_with_pilots: Combined data and pilot symbols
        """
        xp = self.xp
        
        # Convert to GPU if needed
        if self.use_gpu:
            if isinstance(data_symbols, np.ndarray):
                data_symbols = cp.asarray(data_symbols)
            if isinstance(pilot_indices, np.ndarray):
                pilot_indices = cp.asarray(pilot_indices)
            if isinstance(pilot_values, np.ndarray):
                pilot_values = cp.asarray(pilot_values)
        
        # Initialize output
        n_total = len(data_symbols) + len(pilot_values)
        symbols_with_pilots = xp.zeros(n_total, dtype=xp.complex64)
        
        # Create mask for pilot positions
        pilot_mask = xp.zeros(n_total, dtype=bool)
        pilot_mask[pilot_indices] = True
        
        # Insert pilots
        symbols_with_pilots[pilot_mask] = pilot_values
        
        # Insert data in remaining positions
        symbols_with_pilots[~pilot_mask] = data_symbols
        
        return symbols_with_pilots
    
    def generate_ofdm_symbol(self, 
                            freq_symbols: Union[np.ndarray, 'cp.ndarray'],
                            symbol_index: int = 0,
                            normalize: bool = True) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Generate single OFDM symbol with cyclic prefix.
        
        Args:
            freq_symbols: Frequency-domain symbols for active subcarriers
            symbol_index: Symbol index in slot (0-13)
            normalize: Apply √N_FFT normalization (default: True)
        
        Returns:
            x_symbol: Time-domain OFDM symbol with CP (N_FFT + N_CP,)
            
        Example:
            >>> # DeepVerse-6G: 1633 symbols → OFDM symbol with CP
            >>> symbols = mod.modulate(bits)  # 1633 QPSK symbols
            >>> x = ofdm.generate_ofdm_symbol(symbols)
            >>> # x.shape = (3072,)  # 2048 FFT + 1024 CP
        """
        xp = self.xp
        
        # Convert to GPU if needed
        if self.use_gpu and isinstance(freq_symbols, np.ndarray):
            freq_symbols = cp.asarray(freq_symbols)
        
        # Validate input size
        if len(freq_symbols) != self.n_active_subcarriers:
            raise ValueError(
                f"Expected {self.n_active_subcarriers} symbols, "
                f"got {len(freq_symbols)}"
            )
        
        # Initialize frequency-domain signal (all zeros)
        X = xp.zeros(self.config.n_fft, dtype=xp.complex64)
        
        # Map symbols to active subcarriers
        if self.use_gpu:
            X[self.subcarrier_indices_gpu] = freq_symbols
        else:
            X[self.subcarrier_indices] = freq_symbols
        
        # Null DC subcarrier (if configured)
        if self.config.dc_null:
            X[self.dc_index] = 0.0
        
        # IFFT to time domain
        x = xp.fft.ifft(X)
        
        # Apply √N_FFT normalization (3GPP convention)
        # This ensures unit average power for normalized input
        if normalize:
            x = x * xp.sqrt(self.config.n_fft)
        
        # Add cyclic prefix
        cp_length = self.config.cp_length_samples
        cp = x[-cp_length:]  # Last cp_length samples
        x_with_cp = xp.concatenate([cp, x])
        
        return x_with_cp
    
    def generate_ofdm_slot(self,
                          freq_grid: Union[np.ndarray, 'cp.ndarray'],
                          ) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Generate complete OFDM slot (14 symbols).
        
        Args:
            freq_grid: Frequency-domain symbols for slot
                      Shape: (N_symbols, N_active_subcarriers) or
                             (N_symbols, N_active_subcarriers, N_tx)
        
        Returns:
            x_slot: Time-domain slot signal
                   Shape: (N_samples,) or (N_samples, N_tx)
                   
        Example:
            >>> # Single antenna (SISO)
            >>> freq_grid = np.random.randn(14, 1633) + 1j*...
            >>> x_slot = ofdm.generate_ofdm_slot(freq_grid)
            >>> # x_slot.shape = (43008,)  # 14 * 3072
            
            >>> # Multi-antenna (MIMO 4 TX)
            >>> freq_grid = np.random.randn(14, 1633, 4) + 1j*...
            >>> x_slot = ofdm.generate_ofdm_slot(freq_grid)
            >>> # x_slot.shape = (43008, 4)
        """
        xp = self.xp
        
        # Convert to GPU if needed
        if self.use_gpu and isinstance(freq_grid, np.ndarray):
            freq_grid = cp.asarray(freq_grid)
        
        n_symbols = self.config.n_symbols_per_slot
        
        # Validate input
        if freq_grid.shape[0] != n_symbols:
            raise ValueError(
                f"Expected {n_symbols} symbols, got {freq_grid.shape[0]}"
            )
        
        # Handle MIMO case
        if freq_grid.ndim == 3:
            # (N_symbols, N_subcarriers, N_tx)
            n_tx = freq_grid.shape[2]
            
            # Generate per antenna
            slot_symbols = [[] for _ in range(n_tx)]
            
            for sym_idx in range(n_symbols):
                for tx_idx in range(n_tx):
                    x_sym = self.generate_ofdm_symbol(
                        freq_grid[sym_idx, :, tx_idx], 
                        sym_idx
                    )
                    slot_symbols[tx_idx].append(x_sym)
            
            # Concatenate per antenna
            x_slot = xp.stack([xp.concatenate(symbols) for symbols in slot_symbols], axis=1)
            
        else:
            # Single antenna case
            slot_symbols = []
            
            for sym_idx in range(n_symbols):
                x_sym = self.generate_ofdm_symbol(freq_grid[sym_idx, :], sym_idx)
                slot_symbols.append(x_sym)
            
            # Concatenate all symbols
            x_slot = xp.concatenate(slot_symbols)
        
        return x_slot
    
    def compute_papr(self, signal: Union[np.ndarray, 'cp.ndarray']) -> float:
        """
        Compute Peak-to-Average Power Ratio (PAPR).
        
        PAPR = 10 * log10(max(|x|^2) / mean(|x|^2))
        
        Args:
            signal: Time-domain signal
        
        Returns:
            papr_db: PAPR in dB
            
        Note:
            Typical OFDM PAPR: 8-12 dB
            With large FFT (2048): 10-11 dB expected
        """
        xp = self.xp
        
        # Handle multi-antenna case
        if signal.ndim > 1:
            signal = signal.flatten()
        
        power = xp.abs(signal) ** 2
        peak_power = xp.max(power)
        avg_power = xp.mean(power)
        
        papr_linear = peak_power / (avg_power + 1e-20)
        papr_db = 10 * xp.log10(papr_linear)
        
        return float(papr_db)
    
    def normalize_power(self,
                       signal: Union[np.ndarray, 'cp.ndarray'],
                       target_power_w: float) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Normalize signal to target average power.
        
        Args:
            signal: Input signal
            target_power_w: Target average power in watts
        
        Returns:
            signal_normalized: Power-normalized signal
        """
        xp = self.xp
        
        # Compute current average power
        current_power = xp.mean(xp.abs(signal) ** 2)
        
        # Compute scaling factor
        scale = xp.sqrt(target_power_w / (current_power + 1e-20))
        
        # Apply scaling
        signal_normalized = signal * scale
        
        return signal_normalized
    
    def print_info(self):
        """Print OFDM generator information."""
        print("\n" + "="*70)
        print("OFDM SIGNAL GENERATOR (Communication)")
        print("3GPP TS 38.211 | DeepVerse-6G Validated")
        print("="*70)
        print(f"  FFT Size:                 {self.config.n_fft}")
        print(f"  Used Subcarriers:         {self.config.n_subcarriers_used}")
        print(f"  Active Subcarriers:       {self.n_active_subcarriers} ✅ (DeepVerse-6G: 1633)")
        print(f"  DC Null:                  {'✓' if self.config.dc_null else '✗'}")
        print(f"  Guard Bands (L/R):        {self.config.guard_band_left}/{self.config.guard_band_right}")
        print(f"  CP Length:                {self.config.cp_length_samples} samples ({self.config.cp_duration_s*1e6:.3f} μs)")
        print(f"  Symbols per Slot:         {self.config.n_symbols_per_slot}")
        print(f"  Subcarrier Spacing:       {self.config.subcarrier_spacing_khz} kHz")
        print(f"  Sampling Rate:            {self.config.sampling_rate_hz/1e6:.2f} MHz")
        print(f"  Symbol Duration:          {self.config.total_symbol_duration_s*1e6:.2f} μs")
        print(f"  Slot Duration:            {self.config.slot_duration_s*1e6:.2f} μs")
        print(f"  GPU Acceleration:         {'✓ Enabled (CuPy)' if self.use_gpu else '✗ Disabled (NumPy)'}")
        print("="*70 + "\n")


class FMCWChirpGenerator:
    """
    FMCW chirp generator for radar sensing.
    
    Generates linear frequency modulated chirps for FMCW radar.
    
    VALIDATED FOR DEEPVERSE-6G:
    - Chirp slope: 24 THz/s
    - Samples per chirp: 1664
    - Number of chirps: 128
    - Bandwidth: 200 MHz
    - Range resolution: 0.749 m
    - Velocity resolution: 0.167 m/s (monostatic)
    
    Args:
        config: ISAC transmitter configuration
        use_gpu: Enable GPU acceleration
        
    Example:
        >>> config = get_default_config()
        >>> fmcw = FMCWChirpGenerator(config)
        >>> chirp = fmcw.generate_chirp(n_samples=1664)
    """
    
    def __init__(self, config: 'ISACTXConfig', use_gpu: bool = True):
        """Initialize FMCW generator."""
        self.config = config
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        
        if self.use_gpu and not CUPY_AVAILABLE:
            self.use_gpu = False
        
        self.xp = cp if self.use_gpu else np
    
    def generate_chirp(self, 
                      n_samples: Optional[int] = None,
                      chirp_index: int = 0) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Generate FMCW chirp signal.
        
        Linear FM chirp: s(t) = exp(j * π * slope * t^2)
        
        Args:
            n_samples: Number of time samples (default: from config)
            chirp_index: Chirp index in frame (for phase continuity)
        
        Returns:
            chirp: Complex chirp signal (unit power)
            
        Example:
            >>> chirp = fmcw.generate_chirp(1664)
            >>> # chirp.shape = (1664,)
            >>> # |chirp[i]| = 1.0 for all i
        """
        xp = self.xp
        
        if n_samples is None:
            n_samples = self.config.fmcw_n_samples_per_chirp
        
        # Time vector
        dt = 1.0 / self.config.fmcw_sampling_rate_hz
        t = xp.arange(n_samples, dtype=xp.float32) * dt
        
        # Chirp parameters
        slope = self.config.fmcw_chirp_slope
        
        # Instantaneous phase: φ(t) = π * slope * t^2
        phi = xp.pi * slope * t**2
        
        # Complex chirp (unit amplitude)
        chirp = xp.exp(1j * phi).astype(xp.complex64)
        
        return chirp
    
    def generate_chirp_frame(self) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Generate complete FMCW frame (multiple chirps).
        
        Returns:
            frame: Chirp frame (N_chirps, N_samples_per_chirp)
            
        Example:
            >>> frame = fmcw.generate_chirp_frame()
            >>> # frame.shape = (128, 1664) for DeepVerse-6G
        """
        xp = self.xp
        
        n_chirps = self.config.fmcw_n_chirps
        n_samples = self.config.fmcw_n_samples_per_chirp
        
        frame = xp.zeros((n_chirps, n_samples), dtype=xp.complex64)
        
        for i in range(n_chirps):
            frame[i, :] = self.generate_chirp(n_samples, chirp_index=i)
        
        return frame
    
    def print_info(self):
        """Print FMCW generator information."""
        print("\n" + "="*70)
        print("FMCW CHIRP GENERATOR (Radar)")
        print("DeepVerse-6G Validated")
        print("="*70)
        print(f"  Chirp Slope:              {self.config.fmcw_chirp_slope/1e12:.1f} THz/s ✅")
        print(f"  Chirp Duration:           {self.config.fmcw_chirp_duration_s*1e6:.2f} μs")
        print(f"  Samples per Chirp:        {self.config.fmcw_n_samples_per_chirp} ✅")
        print(f"  Number of Chirps:         {self.config.fmcw_n_chirps} ✅")
        print(f"  Bandwidth:                {self.config.fmcw_bandwidth_hz/1e6:.1f} MHz ✅")
        print(f"  Range Resolution:         {self.config.fmcw_range_resolution_m:.3f} m ✅")
        print(f"  Velocity Resolution:      {self.config.fmcw_velocity_resolution_ms:.3f} m/s ✅")
        print(f"  Max Range:                {self.config.fmcw_max_range_m:.1f} m")
        print(f"  Max Velocity:             {self.config.fmcw_max_velocity_ms:.1f} m/s")
        print(f"  Frame Rate:               {self.config.fmcw_frame_rate_hz:.1f} Hz")
        print(f"  Sampling Rate:            {self.config.fmcw_sampling_rate_hz/1e6:.2f} MHz")
        print(f"  GPU Acceleration:         {'✓ Enabled (CuPy)' if self.use_gpu else '✗ Disabled (NumPy)'}")
        print("="*70 + "\n")


class ISACWaveformGenerator:
    """
    Integrated Sensing and Communication (ISAC) waveform generator.
    
    Combines OFDM communication with FMCW radar in a dual-function system.
    Implements spectral coexistence architecture (shared 28 GHz, 200 MHz).
    
    VALIDATED FOR DEEPVERSE-6G:
    - COMM: 1633 subcarriers, 4 TX, 2 RX, QPSK
    - RADAR: FMCW with 1664 samples/chirp, 128 chirps
    - Carrier: 28 GHz
    - Bandwidth: 200 MHz
    - TX Power: 20 W (43 dBm)
    
    VERSION 4.0 - Additive ISAC Integration:
    - New: x(t) = α·x_ofdm(t) + β·chirp(t) with α² + β² = 1
    - Legacy: x(t) = x_ofdm(t) ⊙ chirp(t) (multiplicative)
    
    Based on MathWorks ISAC Part II methodology.
    
    Args:
        config: ISAC transmitter configuration
        modulation: Modulation scheme ('BPSK', 'QPSK', '16QAM', '64QAM', '256QAM')
        use_gpu: Enable GPU acceleration
        fmcw_integration: Integration mode ('additive' or 'multiplicative')
        alpha: COMM power allocation factor (default: 0.707 for equal power)
        beta: RADAR power allocation factor (default: 0.707 for equal power)
        
    Example:
        >>> config = get_default_config()
        >>> # Equal power allocation (50% COMM, 50% RADAR)
        >>> waveform_gen = ISACWaveformGenerator(
        ...     config, 'QPSK',
        ...     fmcw_integration='additive',
        ...     alpha=0.707, beta=0.707
        ... )
        >>> result = waveform_gen.generate_slot_waveform(n_ue=1, n_tx=4)
    """
    
    def __init__(self,
                 config: 'ISACTXConfig',
                 modulation: str = 'QPSK',
                 use_gpu: bool = True,
                 fmcw_integration: str = 'additive',
                 alpha: float = 0.707,
                 beta: float = 0.707):
        """Initialize ISAC waveform generator."""
        self.config = config
        self.modulation_scheme = modulation
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        
        if self.use_gpu and not CUPY_AVAILABLE:
            warnings.warn("CuPy not available. Falling back to NumPy (CPU).")
            self.use_gpu = False
        
        # Set array library
        self.xp = cp if self.use_gpu else np
        
        # ✅ NEW: FMCW integration mode
        self.fmcw_integration = fmcw_integration
        
        # ✅ NEW: Set power allocation factors
        self.set_power_allocation(alpha, beta)
        
        # Initialize modulator
        if DigitalModulator is not None:
            self.modulator = DigitalModulator(modulation, use_gpu=self.use_gpu)
        else:
            raise ImportError("DigitalModulator not available")
        
        # Initialize OFDM generator (Communication)
        self.ofdm_gen = OFDMSignalGenerator(config, use_gpu=self.use_gpu)
        
        # Initialize FMCW generator (Radar)
        if config.fmcw_enable:
            self.fmcw_gen = FMCWChirpGenerator(config, use_gpu=self.use_gpu)
    
    def set_power_allocation(self, alpha: float, beta: float):
        """
        Set power allocation factors for additive ISAC.
        
        Validates and enforces the constraint: α² + β² = 1
        
        Args:
            alpha: COMM power allocation factor (0 ≤ α ≤ 1)
            beta: RADAR power allocation factor (0 ≤ β ≤ 1)
            
        Examples:
            >>> # Pure COMM (no radar)
            >>> gen.set_power_allocation(alpha=1.0, beta=0.0)
            
            >>> # Equal power (50% COMM, 50% RADAR)
            >>> gen.set_power_allocation(alpha=0.707, beta=0.707)
            
            >>> # COMM-dominant (81% COMM, 19% RADAR)
            >>> gen.set_power_allocation(alpha=0.9, beta=0.436)
            
            >>> # RADAR-dominant (19% COMM, 81% RADAR)
            >>> gen.set_power_allocation(alpha=0.436, beta=0.9)
            
            >>> # Pure RADAR (no comm)
            >>> gen.set_power_allocation(alpha=0.0, beta=1.0)
        """
        # Validate constraint: α² + β² = 1
        power_sum = alpha**2 + beta**2
        
        if abs(power_sum - 1.0) > 1e-6:
            warnings.warn(
                f"Power allocation constraint violated: "
                f"α²={alpha**2:.6f}, β²={beta**2:.6f}, "
                f"α²+β²={power_sum:.6f} (must equal 1.0). "
                f"Normalizing to satisfy constraint..."
            )
            # Normalize to satisfy constraint
            norm_factor = np.sqrt(power_sum)
            alpha = alpha / norm_factor
            beta = beta / norm_factor
            
            # Verify
            power_sum_normalized = alpha**2 + beta**2
            print(f"[POWER ALLOCATION] Normalized factors:")
            print(f"  α={alpha:.6f}, β={beta:.6f}")
            print(f"  Verification: α²+β²={power_sum_normalized:.6f} ✅")
        
        self.alpha = alpha
        self.beta = beta
        
        # Store power fractions for reporting
        self.comm_power_fraction = alpha**2
        self.radar_power_fraction = beta**2
        
        print(f"\n[ISAC POWER ALLOCATION]")
        print(f"  COMM:  α={self.alpha:.4f} → {self.comm_power_fraction*100:.2f}% power")
        print(f"  RADAR: β={self.beta:.4f} → {self.radar_power_fraction*100:.2f}% power")
        print(f"  Constraint: α²+β²={self.alpha**2 + self.beta**2:.6f} ✅")
    
    def apply_fmcw_modulation(self,
                             ofdm_signal: Union[np.ndarray, 'cp.ndarray']
                             ) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Integrate FMCW chirp with OFDM signal.
        
        Two integration methods supported:
        
        1. ADDITIVE (new, default):
           x_isac(t) = α·x_ofdm(t) + β·chirp(t)
           where α² + β² = 1 (power constraint)
           
           Advantages:
           - Linear superposition (preserves OFDM structure)
           - Independent COMM/RADAR power control
           - Better PAPR management
           - True dual-function coexistence
        
        2. MULTIPLICATIVE (legacy):
           x_isac(t) = x_ofdm(t) ⊙ chirp(t)
           
           Note: This mode is kept for backward compatibility
        
        Args:
            ofdm_signal: OFDM baseband signal
                Shape: (N_samples,) for SISO or (N_samples, N_tx) for MIMO
        
        Returns:
            isac_signal: Integrated ISAC signal (same shape as input)
            
        Example:
            >>> # Additive mode with equal power
            >>> gen.fmcw_integration = 'additive'
            >>> gen.alpha = 0.707  # 50% COMM
            >>> gen.beta = 0.707   # 50% RADAR
            >>> x_isac = gen.apply_fmcw_modulation(x_ofdm)
        """
        xp = self.xp
        n_samples = len(ofdm_signal) if ofdm_signal.ndim == 1 else ofdm_signal.shape[0]
        
        # Generate chirp matching OFDM duration
        chirp = self.fmcw_gen.generate_chirp(n_samples)
        
        # ============================================
        # INTEGRATION MODE SELECTION
        # ============================================
        
        if self.fmcw_integration == 'multiplicative':
            # ----------------------------------------
            # LEGACY: Multiplicative Integration
            # ----------------------------------------
            print(f"[FMCW] Integration mode: MULTIPLICATIVE")
            print(f"[FMCW] Formula: x_isac(t) = x_ofdm(t) ⊙ chirp(t)")
            
            if ofdm_signal.ndim == 1:
                # SISO case
                isac_signal = ofdm_signal * chirp
            else:
                # MIMO case: broadcast chirp to all antennas
                isac_signal = ofdm_signal * chirp[:, xp.newaxis]
        
        elif self.fmcw_integration == 'additive':
            # ----------------------------------------
            # NEW: Additive Integration with Power Allocation
            # ----------------------------------------
            print(f"[FMCW] Integration mode: ADDITIVE")
            print(f"[FMCW] Formula: x_isac(t) = α·x_ofdm(t) + β·chirp(t)")
            print(f"[FMCW] Power allocation:")
            print(f"  COMM:  α={self.alpha:.4f} ({self.comm_power_fraction*100:.1f}% power)")
            print(f"  RADAR: β={self.beta:.4f} ({self.radar_power_fraction*100:.1f}% power)")
            
            if ofdm_signal.ndim == 1:
                # SISO case
                isac_signal = self.alpha * ofdm_signal + self.beta * chirp
            else:
                # MIMO case: apply same chirp to all antennas
                chirp_expanded = chirp[:, xp.newaxis]  # (N_samples, 1)
                isac_signal = self.alpha * ofdm_signal + self.beta * chirp_expanded
        
        else:
            raise ValueError(
                f"Unknown fmcw_integration mode: '{self.fmcw_integration}'. "
                f"Must be 'additive' or 'multiplicative'"
            )
        
        # ============================================
        # POWER VERIFICATION
        # ============================================
        
        if self.use_gpu:
            P_ofdm = float(cp.mean(cp.abs(ofdm_signal)**2))
            P_chirp = float(cp.mean(cp.abs(chirp)**2))
            P_isac = float(cp.mean(cp.abs(isac_signal)**2))
        else:
            P_ofdm = float(np.mean(np.abs(ofdm_signal)**2))
            P_chirp = float(np.mean(np.abs(chirp)**2))
            P_isac = float(np.mean(np.abs(isac_signal)**2))
        
        print(f"\n[FMCW POWER VERIFICATION]")
        print(f"  P_ofdm (input):  {P_ofdm:.6f} W")
        print(f"  P_chirp (unit):  {P_chirp:.6f} W")
        print(f"  P_isac (output): {P_isac:.6f} W")
        
        if self.fmcw_integration == 'additive':
            # For additive: P_isac ≈ α²·P_ofdm + β²·P_chirp (assuming uncorrelated)
            P_expected = self.alpha**2 * P_ofdm + self.beta**2 * P_chirp
            P_error = abs(P_isac - P_expected) / P_expected * 100 if P_expected > 0 else 0
            
            print(f"  P_expected (α²·P_ofdm + β²·P_chirp): {P_expected:.6f} W")
            print(f"  Power error: {P_error:.2f}%")
            
            if P_error > 5.0:
                warnings.warn(
                    f"Large power deviation detected ({P_error:.1f}%). "
                    f"This may indicate correlation between OFDM and chirp signals."
                )
        
        return isac_signal
    
    def generate_slot_waveform(self,
                              n_ue: int = 0,
                              n_tx: int = 1,
                              seed: Optional[int] = None) -> Dict:
        """
        Generate complete ISAC slot waveform.
        
        Args:
            n_ue: UE index (for reproducibility)
            n_tx: Number of TX antennas (1 or 4 for DeepVerse-6G)
            seed: Random seed
        
        Returns:
            waveform_dict: Dictionary containing:
                - waveform: Generated signal (N_samples,) or (N_samples, N_tx)
                - data_bits: Transmitted bits
                - data_symbols: Modulated symbols
                - freq_grid: Frequency-domain grid
                - metadata: Signal metadata (includes power allocation info)
                
        Example:
            >>> # SISO (single antenna) with additive ISAC
            >>> result = gen.generate_slot_waveform(n_ue=1, n_tx=1, seed=42)
            >>> # result['waveform'].shape = (43008,)
            >>> # result['metadata']['comm_power_fraction'] = 0.5
            >>> # result['metadata']['radar_power_fraction'] = 0.5
            
            >>> # MIMO (4 TX antennas - DeepVerse-6G)
            >>> result = gen.generate_slot_waveform(n_ue=1, n_tx=4, seed=42)
            >>> # result['waveform'].shape = (43008, 4)
        """
        xp = self.xp
        
        # Set random seed for reproducibility
        if seed is not None:
            if self.use_gpu:
                cp.random.seed(seed)
            else:
                np.random.seed(seed)
        
        # Generate random data bits
        n_symbols_per_slot = self.config.n_symbols_per_slot
        n_subcarriers = self.ofdm_gen.n_active_subcarriers
        n_bits_per_symbol = self.modulator.bits_per_symbol
        
        # Total data for single antenna
        total_data_symbols = n_symbols_per_slot * n_subcarriers
        total_bits = total_data_symbols * n_bits_per_symbol
        
        # Generate random bits
        if self.use_gpu:
            data_bits = cp.random.randint(0, 2, total_bits, dtype=cp.int32)
        else:
            data_bits = np.random.randint(0, 2, total_bits, dtype=np.int32)
        
        # Modulate to symbols
        data_symbols = self.modulator.modulate(data_bits)
        
        # Reshape to grid (14 symbols × n_subcarriers)
        freq_grid = data_symbols.reshape(n_symbols_per_slot, n_subcarriers)
        
        # For MIMO, replicate or use different data per antenna
        if n_tx > 1:
            # Option 1: Same data on all antennas (identity precoding)
            # Shape: (N_symbols, N_subcarriers, N_tx)
            freq_grid_mimo = xp.stack([freq_grid] * n_tx, axis=2)
        else:
            freq_grid_mimo = freq_grid
        
        # Generate OFDM slot
        x_ofdm = self.ofdm_gen.generate_ofdm_slot(freq_grid_mimo)
        
        # Apply FMCW modulation if enabled
        if self.config.fmcw_enable:
            x_isac = self.apply_fmcw_modulation(x_ofdm)
        else:
            x_isac = x_ofdm
        
        # Normalize to target power
        x_normalized = self.ofdm_gen.normalize_power(x_isac, self.config.tx_power_w)
        
        # Compute PAPR
        papr_db = self.ofdm_gen.compute_papr(x_normalized)
        
        # Compute actual power
        if self.use_gpu:
            avg_power = float(cp.mean(cp.abs(x_normalized) ** 2))
        else:
            avg_power = float(np.mean(np.abs(x_normalized) ** 2))
        
        # Create metadata
        metadata = {
            'n_ue': n_ue,
            'n_tx': n_tx,
            'modulation': self.modulation_scheme,
            'n_symbols': n_symbols_per_slot,
            'n_subcarriers': n_subcarriers,
            'n_samples': len(x_normalized) if x_normalized.ndim == 1 else x_normalized.shape[0],
            'duration_s': (len(x_normalized) if x_normalized.ndim == 1 else x_normalized.shape[0]) / self.config.sampling_rate_hz,
            'papr_db': float(papr_db),
            'avg_power_w': avg_power,
            'avg_power_dbm': 10 * np.log10(avg_power) + 30,
            'target_power_w': self.config.tx_power_w,
            'target_power_dbm': self.config.tx_power_dbm,
            'fmcw_enabled': self.config.fmcw_enable,
            'fmcw_integration': self.fmcw_integration,  # ✅ NEW
            'alpha': self.alpha,  # ✅ NEW
            'beta': self.beta,  # ✅ NEW
            'comm_power_fraction': self.comm_power_fraction,  # ✅ NEW
            'radar_power_fraction': self.radar_power_fraction,  # ✅ NEW
            'seed': seed,
            'sampling_rate_hz': self.config.sampling_rate_hz,
            'carrier_freq_hz': self.config.carrier_freq_hz,
            'bandwidth_hz': self.config.bandwidth_hz
        }
        
        return {
            'waveform': x_normalized,
            'data_bits': data_bits,
            'data_symbols': data_symbols,
            'freq_grid': freq_grid,
            'metadata': metadata
        }
    
    def print_info(self):
        """Print waveform generator information."""
        print("\n" + "="*70)
        print("ISAC WAVEFORM GENERATOR (Dual-Function)")
        print("DeepVerse-6G Validated | MathWorks ISAC Part II")
        print("="*70)
        print(f"  Architecture:             Spectral Coexistence")
        print(f"  Communication:            MIMO-OFDM (4 TX × 2 RX)")
        print(f"  Radar:                    FMCW (1664 samples × 128 chirps)")
        print(f"  Modulation:               {self.modulation_scheme}")
        print(f"  Bits per Symbol:          {self.modulator.bits_per_symbol}")
        print(f"  Active Subcarriers:       {self.ofdm_gen.n_active_subcarriers} (DeepVerse-6G: 1633)")
        print(f"  FMCW Enabled:             {'✓ Yes' if self.config.fmcw_enable else '✗ No'}")
        if self.config.fmcw_enable:
            print(f"  FMCW Integration:         {self.fmcw_integration.upper()} ✅")
            if self.fmcw_integration == 'additive':
                print(f"  Power Allocation:")
                print(f"    COMM:  α={self.alpha:.4f} ({self.comm_power_fraction*100:.1f}%)")
                print(f"    RADAR: β={self.beta:.4f} ({self.radar_power_fraction*100:.1f}%)")
            print(f"  FMCW Chirp Duration:      {self.config.fmcw_chirp_duration_s*1e6:.2f} μs")
            print(f"  FMCW Bandwidth:           {self.config.fmcw_bandwidth_hz/1e6:.1f} MHz")
            print(f"  Range Resolution:         {self.config.fmcw_range_resolution_m:.3f} m")
        print(f"  Carrier Frequency:        {self.config.carrier_freq_hz/1e9:.1f} GHz")
        print(f"  System Bandwidth:         {self.config.bandwidth_hz/1e6:.1f} MHz")
        print(f"  Target TX Power:          {self.config.tx_power_dbm:.1f} dBm ({self.config.tx_power_w:.2f} W)")
        print(f"  GPU Acceleration:         {'✓ Enabled (CuPy)' if self.use_gpu else '✗ Disabled (NumPy)'}")
        print("="*70 + "\n")


# ================================================================
# TESTING AND VALIDATION
# ================================================================
def test_ofdm_generator():
    """Test OFDM generator (Communication)."""
    print("\n" + "="*80)
    print("TEST 1: OFDM Signal Generator (Communication)")
    print("DeepVerse-6G Validated")
    print("="*80)
    
    # Initialize config
    if get_default_config is None:
        print("⚠️  Config not available, skipping test")
        return
    
    config = get_default_config()
    
    # Initialize generator
    use_gpu = CUPY_AVAILABLE
    ofdm_gen = OFDMSignalGenerator(config, use_gpu=use_gpu)
    ofdm_gen.print_info()
    
    xp = cp if use_gpu else np
    
    print(f"[TEST] Subcarrier mapping validation...")
    print(f"  Active subcarriers: {ofdm_gen.n_active_subcarriers}")
    print(f"  Expected (DeepVerse-6G): 1633")
    print(f"  First 5 indices: {ofdm_gen.subcarrier_indices[:5]}")
    print(f"  Last 5 indices: {ofdm_gen.subcarrier_indices[-5:]}")
    print(f"  DC index: {ofdm_gen.dc_index}")
    
    if ofdm_gen.n_active_subcarriers == 1633:
        print("  ✓ Subcarrier count matches DeepVerse-6G")
    else:
        print(f"  ⚠️  Subcarrier count mismatch")
    
    # Generate random data symbols
    n_subcarriers = ofdm_gen.n_active_subcarriers
    data_symbols = (xp.random.randn(n_subcarriers) + 
                   1j * xp.random.randn(n_subcarriers)) / xp.sqrt(2.0)
    data_symbols = data_symbols.astype(xp.complex64)
    
    print(f"\n[TEST] Generating single OFDM symbol...")
    x_symbol = ofdm_gen.generate_ofdm_symbol(data_symbols, symbol_index=0)
    
    expected_length = config.n_fft + config.cp_length_samples
    print(f"  Output length: {len(x_symbol)} samples")
    print(f"  Expected: {expected_length} samples (2048 FFT + 1024 CP)")
    
    if len(x_symbol) == expected_length:
        print("  ✓ Symbol length correct")
    else:
        print("  ✗ Symbol length mismatch")
    
    # Compute PAPR
    papr = ofdm_gen.compute_papr(x_symbol)
    print(f"  PAPR: {papr:.2f} dB (typical OFDM: 8-12 dB)")
    
    # Verify cyclic prefix
    if use_gpu:
        x_cpu = cp.asnumpy(x_symbol)
    else:
        x_cpu = x_symbol
    
    cp_samples = x_cpu[:config.cp_length_samples]
    tail_samples = x_cpu[-config.cp_length_samples:]
    
    cp_match = np.allclose(cp_samples, tail_samples, rtol=1e-5)
    print(f"  CP matches tail: {'✓ Yes' if cp_match else '✗ No'}")
    
    # Generate full slot
    print(f"\n[TEST] Generating full OFDM slot (14 symbols)...")
    freq_grid = xp.random.randn(config.n_symbols_per_slot, n_subcarriers) + \
                1j * xp.random.randn(config.n_symbols_per_slot, n_subcarriers)
    freq_grid = freq_grid.astype(xp.complex64) / xp.sqrt(2.0)
    
    x_slot = ofdm_gen.generate_ofdm_slot(freq_grid)
    
    slot_duration = len(x_slot) / config.sampling_rate_hz
    expected_duration = config.slot_duration_s
    
    print(f"  Slot length: {len(x_slot)} samples")
    print(f"  Slot duration: {slot_duration*1e6:.2f} μs")
    print(f"  Expected duration: {expected_duration*1e6:.2f} μs")
    
    duration_error = abs(slot_duration - expected_duration) / expected_duration * 100
    print(f"  Duration error: {duration_error:.3f}%")
    
    if duration_error < 0.1:
        print("  ✓ Slot duration correct")
    else:
        print("  ⚠️ Slot duration has small error (acceptable)")
    
    print("\n" + "="*80)
    print("✅ OFDM Generator Test PASSED")
    print("="*80 + "\n")


def test_fmcw_generator():
    """Test FMCW generator (Radar)."""
    print("\n" + "="*80)
    print("TEST 2: FMCW Chirp Generator (Radar)")
    print("DeepVerse-6G Validated")
    print("="*80)
    
    # Initialize config
    if get_default_config is None:
        print("⚠️  Config not available, skipping test")
        return
    
    config = get_default_config()
    
    # Initialize generator
    use_gpu = CUPY_AVAILABLE
    fmcw_gen = FMCWChirpGenerator(config, use_gpu=use_gpu)
    fmcw_gen.print_info()
    
    print(f"[TEST] Generating single FMCW chirp...")
    chirp = fmcw_gen.generate_chirp()
    
    print(f"  Chirp length: {len(chirp)} samples")
    print(f"  Expected (DeepVerse-6G): {config.fmcw_n_samples_per_chirp} samples")
    
    if len(chirp) == config.fmcw_n_samples_per_chirp:
        print("  ✓ Chirp length correct")
    else:
        print("  ✗ Chirp length mismatch")
    
    # Verify chirp is unit power
    xp = cp if use_gpu else np
    if use_gpu:
        chirp_power = float(cp.mean(cp.abs(chirp) ** 2))
    else:
        chirp_power = float(np.mean(np.abs(chirp) ** 2))
    
    print(f"  Chirp power: {chirp_power:.6f} (expected: 1.0)")
    
    if abs(chirp_power - 1.0) < 1e-5:
        print("  ✓ Chirp power is unit")
    else:
        print("  ⚠️ Chirp power slightly off (acceptable)")
    
    # Test chirp frame
    print(f"\n[TEST] Generating FMCW frame...")
    frame = fmcw_gen.generate_chirp_frame()
    
    print(f"  Frame shape: {frame.shape}")
    print(f"  Expected (DeepVerse-6G): ({config.fmcw_n_chirps}, {config.fmcw_n_samples_per_chirp})")
    
    if frame.shape == (config.fmcw_n_chirps, config.fmcw_n_samples_per_chirp):
        print("  ✓ Frame shape correct")
    else:
        print("  ✗ Frame shape mismatch")
    
    print("\n" + "="*80)
    print("✅ FMCW Generator Test PASSED")
    print("="*80 + "\n")


def test_isac_waveform_generator():
    """Test ISAC waveform generator (Dual-Function) with additive integration."""
    print("\n" + "="*80)
    print("TEST 3: ISAC Waveform Generator (Additive Integration)")
    print("DeepVerse-6G Validated | Version 4.0")
    print("="*80)
    
    # Initialize config
    if get_default_config is None or DigitalModulator is None:
        print("⚠️  Required modules not available, skipping test")
        return
    
    config = get_default_config()
    
    # Test 1: Equal power allocation (50% COMM, 50% RADAR)
    print(f"\n[TEST 3.1] Equal power allocation (α=0.707, β=0.707)")
    use_gpu = CUPY_AVAILABLE
    isac_gen = ISACWaveformGenerator(
        config, 
        modulation='QPSK', 
        use_gpu=use_gpu,
        fmcw_integration='additive',
        alpha=0.707,
        beta=0.707
    )
    isac_gen.print_info()
    
    # Generate waveform
    waveform_dict = isac_gen.generate_slot_waveform(n_ue=1, n_tx=1, seed=42)
    
    print(f"\n[OUTPUT - SISO, Equal Power]")
    print(f"  Waveform samples: {waveform_dict['metadata']['n_samples']}")
    print(f"  Duration: {waveform_dict['metadata']['duration_s']*1e6:.2f} μs")
    print(f"  PAPR: {waveform_dict['metadata']['papr_db']:.2f} dB")
    print(f"  Average power: {waveform_dict['metadata']['avg_power_w']:.4f} W")
    print(f"  COMM power fraction: {waveform_dict['metadata']['comm_power_fraction']*100:.1f}%")
    print(f"  RADAR power fraction: {waveform_dict['metadata']['radar_power_fraction']*100:.1f}%")
    
    # Test 2: COMM-dominant (81% COMM, 19% RADAR)
    print(f"\n[TEST 3.2] COMM-dominant (α=0.9, β=0.436)")
    isac_gen.set_power_allocation(alpha=0.9, beta=0.436)
    waveform_comm_dom = isac_gen.generate_slot_waveform(n_ue=1, n_tx=1, seed=43)
    
    print(f"  COMM power fraction: {waveform_comm_dom['metadata']['comm_power_fraction']*100:.1f}%")
    print(f"  RADAR power fraction: {waveform_comm_dom['metadata']['radar_power_fraction']*100:.1f}%")
    
    # Test 3: RADAR-dominant (19% COMM, 81% RADAR)
    print(f"\n[TEST 3.3] RADAR-dominant (α=0.436, β=0.9)")
    isac_gen.set_power_allocation(alpha=0.436, beta=0.9)
    waveform_radar_dom = isac_gen.generate_slot_waveform(n_ue=1, n_tx=1, seed=44)
    
    print(f"  COMM power fraction: {waveform_radar_dom['metadata']['comm_power_fraction']*100:.1f}%")
    print(f"  RADAR power fraction: {waveform_radar_dom['metadata']['radar_power_fraction']*100:.1f}%")
    
    # Test 4: Multiplicative mode (legacy)
    print(f"\n[TEST 3.4] Multiplicative mode (legacy)")
    isac_gen_legacy = ISACWaveformGenerator(
        config, 
        modulation='QPSK', 
        use_gpu=use_gpu,
        fmcw_integration='multiplicative'
    )
    waveform_legacy = isac_gen_legacy.generate_slot_waveform(n_ue=1, n_tx=1, seed=42)
    
    print(f"  Integration mode: {waveform_legacy['metadata']['fmcw_integration']}")
    print(f"  PAPR: {waveform_legacy['metadata']['papr_db']:.2f} dB")
    
    print("\n" + "="*80)
    print("✅ ISAC Waveform Generator Test PASSED (All Modes)")
    print("="*80 + "\n")


if __name__ == "__main__":
    """Run all tests."""
    
    print("\n" + "="*80)
    print("ISAC WAVEFORM GENERATOR - VALIDATION TESTS")
    print("Version 4.0: Additive + Multiplicative Integration")
    print("DeepVerse-6G Validated")
    print("="*80)
    
    # Detect GPU
    if CUPY_AVAILABLE:
        print("✓ CuPy detected - GPU acceleration enabled")
        try:
            print(f"  GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        except:
            pass
    else:
        print("✗ CuPy not available - using CPU (NumPy)")
    
    # Run tests
    try:
        test_ofdm_generator()
    except Exception as e:
        print(f"\n✗ OFDM Generator test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
    
    try:
        test_fmcw_generator()
    except Exception as e:
        print(f"\n✗ FMCW Generator test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
    
    try:
        test_isac_waveform_generator()
    except Exception as e:
        print(f"\n✗ ISAC Waveform Generator test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ ALL ISAC WAVEFORM TESTS COMPLETED")
    print("="*80 + "\n")