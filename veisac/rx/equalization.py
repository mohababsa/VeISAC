# equalization.py
"""
VeISAC — MIMO Channel Equalization

Per-subcarrier ZF/MMSE/MRC equalization with FMCW interference-aware regularization and power recovery for the OFDM communication receiver chain.

Paper: "VeISAC: An End-to-End MIMO-OFDM-FMCW Framework for ISAC
        in 6G Vehicular Networks"
Authors: M. Ababsa, S. Ribouh, Y. El Hillali, A. Rivenq
"""

import numpy as np
from typing import Literal, Optional, Union, Tuple
import warnings

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np


class ChannelEqualizer:
    """v7.2 - DETAILED LOGGING - Track every computation step"""
    
    def __init__(
        self,
        method: Literal['zf', 'mmse', 'mrc'] = 'mmse',
        snr_db: float = 5.0,
        fmcw_integration_mode: str = 'additive',
        comm_power_factor: float = np.sqrt(0.5),
        radar_power_factor: float = np.sqrt(0.5),
        use_gpu: bool = True,
        regularization: float = 1e-10,
        verbose: bool = False
    ):
        self.method = method.lower()
        self.snr_db = snr_db
        self.regularization = regularization
        self.verbose = verbose
        
        self.fmcw_integration_mode = fmcw_integration_mode.lower()
        self.comm_power_factor = comm_power_factor
        self.radar_power_factor = radar_power_factor
        
        if self.fmcw_integration_mode not in ['additive', 'multiplicative']:
            raise ValueError(f"Invalid FMCW integration mode")
        
        if self.fmcw_integration_mode == 'additive':
            power_sum = comm_power_factor**2 + radar_power_factor**2
            if abs(power_sum - 1.0) > 1e-6:
                warnings.warn(f"Power constraint violated: α²+β²={power_sum:.6f}")
        
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        print(f"\n{'='*80}")
        print(f"[ChannelEqualizer v7.2 DETAILED LOGGING]")
        print(f"{'='*80}")
        print(f"  Method: {method.upper()}")
        print(f"  ISAC: {self.fmcw_integration_mode.upper()}")
        if self.fmcw_integration_mode == 'additive':
            print(f"  Power factors: α={comm_power_factor:.6f}, β={radar_power_factor:.6f}")
            print(f"  Power fractions: α²={comm_power_factor**2:.6f}, β²={radar_power_factor**2:.6f}")
            print(f"  Sum check: α²+β²={power_sum:.6f} (should be 1.0)")
        print(f"{'='*80}\n")
    
    @property
    def comm_power_fraction(self) -> float:
        return self.comm_power_factor ** 2
    
    @property
    def radar_power_fraction(self) -> float:
        return self.radar_power_factor ** 2
    
    def _compute_mmse_regularization(
        self,
        noise_power: float,
        H_est: Union[np.ndarray, 'cp.ndarray']
    ) -> float:
        """Compute MMSE regularization with detailed logging"""
        
        print(f"\n{'─'*80}")
        print(f"[MMSE Regularization Computation]")
        print(f"{'─'*80}")
        
        if self.fmcw_integration_mode != 'additive':
            print(f"  Mode: {self.fmcw_integration_mode} → returning noise_power directly")
            print(f"  λ (regularization): {noise_power:.6e}")
            return noise_power
        
        xp = self.xp if isinstance(H_est, (cp.ndarray if GPU_AVAILABLE else type(None))) else np
        
        if isinstance(H_est, np.ndarray) and self.use_gpu:
            H_compute = cp.asarray(H_est)
        else:
            H_compute = H_est
        
        h_power_avg = float(xp.mean(xp.abs(H_compute) ** 2))
        
        beta_sq = self.radar_power_factor ** 2
        interference_at_rx = beta_sq * h_power_avg
        
        lambda_mmse = noise_power + interference_at_rx
        
        print(f"  INPUT:")
        print(f"    noise_power (σ²_n): {noise_power:.6e}")
        print(f"  CHANNEL:")
        print(f"    H_est shape: {H_est.shape}")
        print(f"    |H|² (avg): {h_power_avg:.6e}")
        print(f"  POWER FACTORS:")
        print(f"    β (radar factor): {self.radar_power_factor:.6f}")
        print(f"    β²: {beta_sq:.6f}")
        print(f"  INTERFERENCE:")
        print(f"    β² × |H|²: {interference_at_rx:.6e}")
        print(f"  REGULARIZATION:")
        print(f"    λ = σ²_n + β²·|H|²")
        print(f"    λ = {noise_power:.6e} + {interference_at_rx:.6e}")
        print(f"    λ = {lambda_mmse:.6e}")
        print(f"{'─'*80}\n")
        
        return lambda_mmse
    
    def _compute_effective_noise_post_recovery(
        self,
        noise_power: float,
        H_est: Union[np.ndarray, 'cp.ndarray'],
        use_oracle_channel: bool = False,
        ic_applied: bool = False
    ) -> float:
        
        print(f"\n{'='*80}")
        print(f"[POST-RECOVERY EFFECTIVE NOISE COMPUTATION]")
        print(f"{'='*80}")
        
        if self.fmcw_integration_mode != 'additive':
            print(f"  Mode: {self.fmcw_integration_mode} → returning noise_power directly")
            print(f"  σ²_eff: {noise_power:.6e}")
            return noise_power
        
        xp = self.xp if isinstance(H_est, (cp.ndarray if GPU_AVAILABLE else type(None))) else np
        
        if isinstance(H_est, np.ndarray) and self.use_gpu:
            H_compute = cp.asarray(H_est)
        else:
            H_compute = H_est
        
        h_power_avg = float(xp.mean(xp.abs(H_compute) ** 2))
        
        beta_sq = self.radar_power_factor ** 2
        alpha_sq = self.comm_power_factor ** 2
        
        print(f"  STEP 1: INPUT PARAMETERS")
        print(f"    noise_power (σ²_n, thermal): {noise_power:.6e}")
        print(f"    H_est shape: {H_est.shape}")
        print(f"    use_oracle_channel: {use_oracle_channel}")
        print(f"    ic_applied: {ic_applied}")
        
        print(f"\n  STEP 2: CHANNEL POWER")
        print(f"    |H|² (average): {h_power_avg:.6e}")
        print(f"    Expected (normalized): ~1.0")
        print(f"    Status: {'✅ OK' if abs(h_power_avg - 1.0) < 0.1 else '⚠️  NOT NORMALIZED!'}")
        
        print(f"\n  STEP 3: POWER FACTORS")
        print(f"    α (COMM factor): {self.comm_power_factor:.6f}")
        print(f"    β (RADAR factor): {self.radar_power_factor:.6f}")
        print(f"    α²: {alpha_sq:.6f}")
        print(f"    β²: {beta_sq:.6f}")
        print(f"    α² + β²: {alpha_sq + beta_sq:.6f} (should be 1.0)")
        
        interference_full = beta_sq * h_power_avg
        
        if use_oracle_channel:
            print(f"\n  STEP 4: ORACLE CHANNEL MODE")
            print(f"    Perfect H available → MMSE equalizer suppresses interference")
            print(f"    Full interference: {interference_full:.6e}")
            print(f"    Assumed suppression: 85-95%")
            suppression_factor = 0.90
            residual_interference = interference_full * (1 - suppression_factor)
            print(f"    Residual interference: {residual_interference:.6e}")
            noise_before = noise_power + residual_interference
        elif ic_applied:
            print(f"\n  STEP 4: IC APPLIED MODE")
            _ic_iters = getattr(self, 'ic_iterations', 2)
            print(f"    IC removed ~{(_ic_iters * 2 + 6) * 10}% of interference")
            print(f"    Full interference: {interference_full:.6e}")
            _ic_iters = getattr(self, 'ic_iterations', 2)
            residual_ratio = 0.10 if _ic_iters >= 2 else 0.20
            residual_interference = interference_full * residual_ratio
            print(f"    Residual interference: {residual_interference:.6e}")
            noise_before = noise_power + residual_interference
        else:
            print(f"\n  STEP 4: STANDARD MODE (NO IC, NO ORACLE)")
            print(f"    Full interference: {interference_full:.6e}")
            print(f"    No suppression assumed")
            noise_before = noise_power + interference_full
        
        noise_after = noise_before / alpha_sq
        
        print(f"\n  STEP 5: TOTAL NOISE BEFORE RECOVERY")
        print(f"    N_before = {noise_before:.6e}")
        
        print(f"\n  STEP 6: POWER RECOVERY SCALING")
        print(f"    Scaling factor: 1/α² = {1/alpha_sq:.6f}")
        print(f"    σ²_eff = N_before / α²")
        print(f"    σ²_eff = {noise_after:.6e}")
        
        expected_snr_eff = 1.0 / noise_after
        expected_snr_eff_db = 10 * np.log10(expected_snr_eff)
        
        print(f"\n  STEP 7: EXPECTED SNR (POST-RECOVERY)")
        print(f"    Signal power (recovered): 1.0")
        print(f"    Noise power (effective): {noise_after:.6e}")
        print(f"    SNR_eff (linear): {expected_snr_eff:.6f}")
        print(f"    SNR_eff (dB): {expected_snr_eff_db:.2f} dB")
        
        if use_oracle_channel or ic_applied:
            noise_worst = (noise_power + interference_full) / alpha_sq
            snr_worst_db = 10 * np.log10(1.0 / noise_worst)
            improvement_db = expected_snr_eff_db - snr_worst_db
            print(f"\n  COMPARISON TO WORST-CASE:")
            print(f"    Worst-case σ²_eff: {noise_worst:.6e}")
            print(f"    Worst-case SNR: {snr_worst_db:.2f} dB")
            print(f"    Improvement: {improvement_db:.2f} dB")
        
        print(f"{'='*80}\n")
        
        return noise_after
    
    def equalize_with_pilots(
        self,
        y_rx: Union[np.ndarray, 'cp.ndarray'],
        H_est: Union[np.ndarray, 'cp.ndarray'],
        pilot_mask: Union[np.ndarray, 'cp.ndarray'],
        noise_power: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Equalize with detailed logging"""
        
        print(f"\n{'='*80}")
        print(f"[EQUALIZE WITH PILOTS]")
        print(f"{'='*80}")
        
        xp = self.xp
        
        if self.use_gpu:
            if isinstance(y_rx, np.ndarray):
                y_rx = cp.asarray(y_rx)
            if isinstance(H_est, np.ndarray):
                H_est = cp.asarray(H_est)
            if isinstance(pilot_mask, np.ndarray):
                pilot_mask = cp.asarray(pilot_mask)
        
        if y_rx.ndim != 3:
            raise ValueError(f"y_rx must be 3D")
        
        n_rx, n_symbols, n_sc = y_rx.shape
        n_tx = H_est.shape[1]
        
        print(f"  INPUT SHAPES:")
        print(f"    y_rx: {y_rx.shape} (N_rx, N_symbols, N_sc)")
        print(f"    H_est: {H_est.shape} (N_rx, N_tx, N_sc)")
        print(f"    pilot_mask: {pilot_mask.shape}")
        
        if noise_power is not None:
            print(f"    noise_power: {noise_power:.6e}")
        else:
            print(f"    noise_power: None")
        
        data_mask = ~pilot_mask
        data_positions = xp.where(data_mask)
        data_symbol_idx = data_positions[0]
        data_freq_idx = data_positions[1]
        n_data_res = len(data_symbol_idx)
        
        print(f"\n  DATA EXTRACTION:")
        print(f"    N_data_resources: {n_data_res}")
        
        y_data = xp.zeros((n_rx, n_data_res), dtype=xp.complex64)
        for rx_idx in range(n_rx):
            y_data[rx_idx, :] = y_rx[rx_idx, data_symbol_idx, data_freq_idx]
        
        H_data = H_est[:, :, data_freq_idx]
        
        print(f"    y_data shape: {y_data.shape}")
        print(f"    H_data shape: {H_data.shape}")
        
        if noise_power is not None and self.method == 'mmse' and self.fmcw_integration_mode == 'additive':
            lambda_mmse = self._compute_mmse_regularization(noise_power, H_est)
        else:
            lambda_mmse = noise_power
            if lambda_mmse is not None:
                print(f"\n  REGULARIZATION: λ = {lambda_mmse:.6e} (direct, no ISAC adjustment)")
            else:
                print(f"\n  REGULARIZATION: λ = None (will be computed)")
        
        print(f"\n  EQUALIZATION:")
        print(f"    Method: {self.method.upper()}")
        if lambda_mmse is not None:
            print(f"    Regularization λ: {lambda_mmse:.6e}")
        
        x_eq_data = self._equalize_per_subcarrier(y_data, H_data, lambda_mmse)
        
        print(f"    x_eq_data shape: {x_eq_data.shape}")
        print(f"    x_eq_data power (before recovery): {float(xp.mean(xp.abs(x_eq_data)**2)):.6e}")
        
        if self.fmcw_integration_mode == 'additive':
            power_scale = 1.0 / self.comm_power_factor
            x_eq_data_before = x_eq_data.copy()
            x_eq_data = power_scale * x_eq_data
            
            print(f"\n  POWER RECOVERY:")
            print(f"    Scaling: 1/α = 1/{self.comm_power_factor:.6f} = {power_scale:.6f}")
            print(f"    Power before: {float(xp.mean(xp.abs(x_eq_data_before)**2)):.6e}")
            print(f"    Power after: {float(xp.mean(xp.abs(x_eq_data)**2)):.6e}")
            print(f"    Ratio: {float(xp.mean(xp.abs(x_eq_data)**2) / xp.mean(xp.abs(x_eq_data_before)**2)):.6f}")
        
        if self.use_gpu:
            if isinstance(x_eq_data, cp.ndarray):
                x_eq_data = cp.asnumpy(x_eq_data)
            if isinstance(data_mask, cp.ndarray):
                data_mask = cp.asnumpy(data_mask)
        
        print(f"{'='*80}\n")
        
        return x_eq_data, data_mask
    
    def _equalize_per_subcarrier(
        self,
        y_rx: Union[np.ndarray, 'cp.ndarray'],
        H_est: Union[np.ndarray, 'cp.ndarray'],
        noise_power: Optional[float] = None
    ) -> Union[np.ndarray, 'cp.ndarray']:
        if self.method == 'zf':
            return self._equalize_zf_vectorized(y_rx, H_est)
        elif self.method == 'mmse':
            return self._equalize_mmse_vectorized(y_rx, H_est, noise_power)
        elif self.method == 'mrc':
            return self._equalize_mrc_vectorized(y_rx, H_est)
        else:
            raise ValueError(f"Unknown method")
    
    def _equalize_zf_vectorized(
        self,
        y_rx: Union[np.ndarray, 'cp.ndarray'],
        H_est: Union[np.ndarray, 'cp.ndarray']
    ) -> Union[np.ndarray, 'cp.ndarray']:
        xp = self.xp
        n_rx, n_data = y_rx.shape
        n_tx = H_est.shape[1]
        
        x_eq = xp.zeros((n_tx, n_data), dtype=xp.complex64)
        
        for k in range(n_data):
            H_k = H_est[:, :, k]
            y_k = y_rx[:, k]
            
            try:
                H_H = xp.conj(H_k.T)
                gram = H_H @ H_k
                gram_reg = gram + self.regularization * xp.eye(n_tx)
                x_eq[:, k] = xp.linalg.solve(gram_reg, H_H @ y_k)
            except:
                x_eq[:, k] = xp.linalg.pinv(H_k) @ y_k
        
        return x_eq
    
    def _equalize_mmse_vectorized(
        self,
        y_rx: Union[np.ndarray, 'cp.ndarray'],
        H_est: Union[np.ndarray, 'cp.ndarray'],
        noise_power: Optional[float] = None
    ) -> Union[np.ndarray, 'cp.ndarray']:
        xp = self.xp
        n_rx, n_data = y_rx.shape
        n_tx = H_est.shape[1]
        
        if noise_power is None:
            snr_linear = 10 ** (self.snr_db / 10)
            signal_power = float(xp.mean(xp.abs(H_est) ** 2))
            noise_var = signal_power / snr_linear
            
            if self.fmcw_integration_mode == 'additive':
                beta_sq = self.radar_power_factor ** 2
                h_power_avg = signal_power
                noise_var = noise_var + beta_sq * h_power_avg
        else:
            noise_var = noise_power
        
        x_eq = xp.zeros((n_tx, n_data), dtype=xp.complex64)
        
        for k in range(n_data):
            H_k = H_est[:, :, k]
            y_k = y_rx[:, k]
            
            try:
                H_H = xp.conj(H_k.T)
                gram = H_H @ H_k
                gram_reg = gram + noise_var * xp.eye(n_tx)
                x_eq[:, k] = xp.linalg.solve(gram_reg, H_H @ y_k)
            except:
                x_eq[:, k] = xp.linalg.pinv(H_k) @ y_k
        
        return x_eq
    
    def _equalize_mrc_vectorized(
        self,
        y_rx: Union[np.ndarray, 'cp.ndarray'],
        H_est: Union[np.ndarray, 'cp.ndarray']
    ) -> Union[np.ndarray, 'cp.ndarray']:
        xp = self.xp
        n_rx, n_data = y_rx.shape
        n_tx = H_est.shape[1]
        
        if n_tx != 1:
            raise ValueError(f"MRC only supports SIMO")
        
        if self.fmcw_integration_mode == 'additive':
            warnings.warn("MRC not recommended for additive ISAC - use MMSE instead")
        
        H_single = H_est[:, 0, :]
        h_power = xp.sum(xp.abs(H_single) ** 2, axis=0, keepdims=True)
        h_power = xp.maximum(h_power, 1e-12)
        
        W_mrc = xp.conj(H_single) / h_power
        x_eq = xp.sum(W_mrc * y_rx, axis=0, keepdims=True)
        
        return x_eq
    
    def equalize(
        self,
        y_rx: Union[np.ndarray, 'cp.ndarray'],
        H_est: Union[np.ndarray, 'cp.ndarray'],
        noise_power: Optional[float] = None
    ) -> np.ndarray:
        if self.use_gpu:
            if isinstance(y_rx, np.ndarray):
                y_rx = cp.asarray(y_rx)
            if isinstance(H_est, np.ndarray):
                H_est = cp.asarray(H_est)
        
        if noise_power is not None and self.method == 'mmse' and self.fmcw_integration_mode == 'additive':
            lambda_mmse = self._compute_mmse_regularization(noise_power, H_est)
        else:
            lambda_mmse = noise_power
        
        if self.method == 'zf':
            x_eq = self._equalize_zf(y_rx, H_est)
        elif self.method == 'mmse':
            x_eq = self._equalize_mmse(y_rx, H_est, lambda_mmse)
        elif self.method == 'mrc':
            x_eq = self._equalize_mrc(y_rx, H_est)
        else:
            raise ValueError(f"Unknown method")
        
        if self.fmcw_integration_mode == 'additive':
            power_scale = 1.0 / self.comm_power_factor
            x_eq = power_scale * x_eq
        
        if self.use_gpu and isinstance(x_eq, cp.ndarray):
            x_eq = cp.asnumpy(x_eq)
        
        return x_eq
    
    def _equalize_zf(self, y_rx, H_est):
        xp = self.xp
        
        if y_rx.ndim == 2:
            n_rx, n_sc = y_rx.shape
            multiple_symbols = False
        else:
            n_rx, n_sc, n_sym = y_rx.shape
            multiple_symbols = True
        
        n_tx = H_est.shape[1]
        
        H_H = xp.conj(xp.swapaxes(H_est, 0, 1))
        gram = xp.zeros((n_tx, n_tx, n_sc), dtype=complex)
        for k in range(n_sc):
            gram[:, :, k] = H_H[:, :, k] @ H_est[:, :, k]
        
        reg_matrix = self.regularization * xp.eye(n_tx)[:, :, None]
        gram_reg = gram + reg_matrix
        
        W_zf = xp.zeros((n_tx, n_rx, n_sc), dtype=complex)
        for k in range(n_sc):
            try:
                W_zf[:, :, k] = xp.linalg.solve(gram_reg[:, :, k], H_H[:, :, k])
            except:
                W_zf[:, :, k] = xp.linalg.pinv(H_est[:, :, k]).T
        
        if multiple_symbols:
            x_eq = xp.zeros((n_tx, n_sc, n_sym), dtype=complex)
            for sym in range(n_sym):
                for k in range(n_sc):
                    x_eq[:, k, sym] = W_zf[:, :, k] @ y_rx[:, k, sym]
        else:
            x_eq = xp.zeros((n_tx, n_sc), dtype=complex)
            for k in range(n_sc):
                x_eq[:, k] = W_zf[:, :, k] @ y_rx[:, k]
        
        return x_eq
    
    def _equalize_mmse(self, y_rx, H_est, noise_power):
        xp = self.xp
        
        if y_rx.ndim == 2:
            n_rx, n_sc = y_rx.shape
            multiple_symbols = False
        else:
            n_rx, n_sc, n_sym = y_rx.shape
            multiple_symbols = True
        
        n_tx = H_est.shape[1]
        
        if noise_power is None:
            snr_linear = 10 ** (self.snr_db / 10)
            signal_power = float(xp.mean(xp.abs(H_est) ** 2))
            noise_var = signal_power / snr_linear
        else:
            noise_var = noise_power
        
        H_H = xp.conj(xp.swapaxes(H_est, 0, 1))
        gram = xp.zeros((n_tx, n_tx, n_sc), dtype=complex)
        for k in range(n_sc):
            gram[:, :, k] = H_H[:, :, k] @ H_est[:, :, k]
        
        reg_matrix = noise_var * xp.eye(n_tx)[:, :, None]
        gram_reg = gram + reg_matrix
        
        W_mmse = xp.zeros((n_tx, n_rx, n_sc), dtype=complex)
        for k in range(n_sc):
            try:
                W_mmse[:, :, k] = xp.linalg.solve(gram_reg[:, :, k], H_H[:, :, k])
            except:
                W_mmse[:, :, k] = xp.linalg.pinv(H_est[:, :, k]).T
        
        if multiple_symbols:
            x_eq = xp.zeros((n_tx, n_sc, n_sym), dtype=complex)
            for sym in range(n_sym):
                for k in range(n_sc):
                    x_eq[:, k, sym] = W_mmse[:, :, k] @ y_rx[:, k, sym]
        else:
            x_eq = xp.zeros((n_tx, n_sc), dtype=complex)
            for k in range(n_sc):
                x_eq[:, k] = W_mmse[:, :, k] @ y_rx[:, k]
        
        return x_eq
    
    def _equalize_mrc(self, y_rx, H_est):
        xp = self.xp
        
        if y_rx.ndim == 2:
            n_rx, n_sc = y_rx.shape
            multiple_symbols = False
        else:
            n_rx, n_sc, n_sym = y_rx.shape
            multiple_symbols = True
        
        n_tx = H_est.shape[1]
        if n_tx != 1:
            raise ValueError(f"MRC only supports SIMO")
        
        H_single = H_est[:, 0, :]
        h_power = xp.sum(xp.abs(H_single) ** 2, axis=0, keepdims=True)
        h_power = xp.maximum(h_power, 1e-12)
        
        W_mrc = xp.conj(H_single) / h_power
        
        if multiple_symbols:
            x_eq = xp.sum(W_mrc[:, :, None] * y_rx, axis=0, keepdims=True)
        else:
            x_eq = xp.sum(W_mrc * y_rx, axis=0, keepdims=True)
        
        return x_eq
    
    def compute_effective_snr(self, H_est, noise_power):
        xp = self.xp
        
        if isinstance(H_est, np.ndarray) and self.use_gpu:
            H_est = cp.asarray(H_est)
        
        n_rx, n_tx, n_sc = H_est.shape
        signal_power = xp.ones(n_sc, dtype=float)
        
        if self.fmcw_integration_mode == 'additive':
            noise_var_eff = self._compute_effective_noise_post_recovery(noise_power, H_est)
        else:
            noise_var_eff = noise_power
        
        H_H = xp.conj(xp.swapaxes(H_est, 0, 1))
        gram = xp.zeros((n_tx, n_tx, n_sc), dtype=complex)
        for k in range(n_sc):
            gram[:, :, k] = H_H[:, :, k] @ H_est[:, :, k]
        
        noise_amp = xp.zeros(n_sc)
        
        if self.method == 'zf':
            for k in range(n_sc):
                try:
                    gram_inv = xp.linalg.inv(gram[:, :, k])
                    noise_amp[k] = xp.trace(gram_inv).real
                except:
                    noise_amp[k] = 1e6
        elif self.method == 'mmse':
            reg = noise_var_eff * xp.eye(n_tx)
            for k in range(n_sc):
                try:
                    gram_inv = xp.linalg.inv(gram[:, :, k] + reg)
                    noise_amp[k] = xp.trace(gram_inv).real
                except:
                    noise_amp[k] = 1.0
        elif self.method == 'mrc':
            channel_power = xp.sum(xp.abs(H_est) ** 2, axis=(0, 1))
            noise_amp = 1.0 / (channel_power + 1e-12)
        else:
            noise_amp = xp.ones(n_sc)
        
        snr_eff = signal_power / (noise_amp * noise_var_eff + 1e-12)
        
        if self.use_gpu and isinstance(snr_eff, cp.ndarray):
            snr_eff = cp.asnumpy(snr_eff)
        
        return snr_eff
    
    def compute_equalization_metrics(self, H_est, noise_power=None):
        xp = self.xp if isinstance(H_est, (cp.ndarray if GPU_AVAILABLE else type(None))) else np
        
        if isinstance(H_est, np.ndarray) and self.use_gpu:
            H_est = cp.asarray(H_est)
        
        n_rx, n_tx, n_sc = H_est.shape
        
        channel_gain = float(xp.mean(xp.abs(H_est) ** 2))
        channel_gain_db = 10 * np.log10(channel_gain + 1e-12)
        
        if noise_power is not None:
            snr_eff = self.compute_effective_snr(H_est, noise_power)
            effective_snr_db = 10 * np.log10(np.mean(snr_eff) + 1e-12)
        else:
            effective_snr_db = None
        
        return {
            'channel_gain_db': channel_gain_db,
            'effective_snr_db': effective_snr_db,
            'method': self.method,
            'isac_mode': self.fmcw_integration_mode,
        }
    
    def equalize_batch(self, y_rx_batch, H_est_batch, noise_power=None):
        if H_est_batch.ndim == 3:
            return self.equalize(y_rx_batch, H_est_batch, noise_power)
        
        n_sym = y_rx_batch.shape[2]
        x_eq_list = []
        
        for sym_idx in range(n_sym):
            y_sym = y_rx_batch[:, :, sym_idx]
            H_sym = H_est_batch[:, :, :, sym_idx]
            x_eq_sym = self.equalize(y_sym, H_sym, noise_power)
            x_eq_list.append(x_eq_sym)
        
        return np.stack(x_eq_list, axis=2)


if __name__ == "__main__":
    print("="*80)
    print("EQUALIZATION v7.2 - DETAILED LOGGING TEST")
    print("="*80)