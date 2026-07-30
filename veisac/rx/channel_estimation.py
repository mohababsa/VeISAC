# channel_estimation.py
"""
VeISAC — MIMO Channel Estimation

Pilot-based LS/MMSE/DFT channel estimation with iterative FMCW interference cancellation for the OFDM communication receiver chain.

Paper: "VeISAC: An End-to-End MIMO-OFDM-FMCW Framework for ISAC
        in 6G Vehicular Networks"
Authors: M. Ababsa, S. Ribouh, Y. El Hillali, A. Rivenq
"""

import numpy as np
from typing import Optional, Literal, Union, Tuple
import warnings

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np


class ChannelEstimator:
    
    def __init__(
        self,
        method: Literal['ls', 'mmse', 'dft'] = 'mmse',
        interpolation: Literal['linear', 'spline', 'cubic', 'nearest'] = 'linear',
        pilot_spacing_time: int = 3,
        pilot_spacing_freq: int = 3,
        n_fft: int = 2048,
        snr_db: float = 5.0,
        max_delay_taps: Optional[int] = None,
        fmcw_integration_mode: str = 'additive',
        comm_power_factor: float = np.sqrt(0.5),
        radar_power_factor: float = np.sqrt(0.5),
        enable_interference_cancellation: bool = True,
        ic_iterations: int = 2,
        use_gpu: bool = True,
        verbose: bool = False
    ):
        self.method = method.lower()
        self.interpolation = interpolation
        self.pilot_spacing_time = pilot_spacing_time
        self.pilot_spacing_freq = pilot_spacing_freq
        self.n_fft = n_fft
        self.snr_db = snr_db
        self.max_delay_taps = max_delay_taps if max_delay_taps is not None else n_fft // 16
        self.verbose = verbose
    
        self.fmcw_integration_mode = fmcw_integration_mode.lower()
        self.comm_power_factor = comm_power_factor
        self.radar_power_factor = radar_power_factor
        self.enable_ic = enable_interference_cancellation
        self.ic_iterations = max(1, min(ic_iterations, 3))
    
        # NEW: IC reconstruction scale to prevent over-cancellation
        self.ic_reconstruction_scale = 0.90
    
        if self.fmcw_integration_mode == 'additive':
            power_sum = comm_power_factor**2 + radar_power_factor**2
            if abs(power_sum - 1.0) > 1e-6:
                warnings.warn(
                    f"Power allocation constraint violated: α²+β²={power_sum:.6f} ≠ 1.0"
                )
    
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
    
        snr_linear = 10 ** (self.snr_db / 10)
        self.sigma_n_sq = 1.0 / snr_linear
    
        if self.fmcw_integration_mode == 'additive':
            self.epsilon_base = (self.radar_power_factor ** 2) + self.sigma_n_sq
        else:
            self.epsilon_base = self.sigma_n_sq
    
        print(f"\n{'='*80}")
        print(f"[ChannelEstimator v8.2 - STABILIZED IC]")
        print(f"{'='*80}")
        print(f"  Method: {method.upper()}")
        print(f"  ISAC Mode: {self.fmcw_integration_mode.upper()}")
        print(f"  Interference Cancellation: {'ENABLED' if self.enable_ic else 'DISABLED'}")
        if self.enable_ic and self.fmcw_integration_mode == 'additive':
            print(f"  IC Iterations: {self.ic_iterations}")
            print(f"  IC Reconstruction Scale: {self.ic_reconstruction_scale:.2f} ✅")
        if self.fmcw_integration_mode == 'additive':
            print(f"  Power Allocation:")
            print(f"    α (COMM): {comm_power_factor:.6f}, α²: {comm_power_factor**2:.6f}")
            print(f"    β (RADAR): {radar_power_factor:.6f}, β²: {radar_power_factor**2:.6f}")
            print(f"    α²+β²: {power_sum:.6f} (should be 1.0)")
            print(f"  Noise Parameters:")
            print(f"    SNR: {snr_db:.1f} dB")
            print(f"    σ²_n: {self.sigma_n_sq:.6e}")
            print(f"    β²: {self.radar_power_factor**2:.6f}")
            print(f"    ε_base: {self.epsilon_base:.6f}")
        else:
            print(f"  SNR: {snr_db:.1f} dB → σ²_n = {self.sigma_n_sq:.6e}")
        print(f"{'='*80}\n")
    
    @property
    def comm_power_fraction(self):
        return self.comm_power_factor ** 2
    
    # ========================================================================
    # v8.1 CORRECTED: PILOT-BASED RADAR INTERFERENCE CANCELLATION
    # ========================================================================
    
    def cancel_radar_interference(
        self,
        y_pilot: Union[np.ndarray, 'cp.ndarray'],
        x_pilot: Union[np.ndarray, 'cp.ndarray'],
        c_pilot: Union[np.ndarray, 'cp.ndarray'],
        n_iterations: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[Union[np.ndarray, 'cp.ndarray'], Union[np.ndarray, 'cp.ndarray']]:
        
        xp = self.xp
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"[PILOT-BASED IC v8.1 - ITERATIVE WITH CORRECT MATH]")
            print(f"{'='*80}")
        
        # Convert to GPU if needed
        if self.use_gpu:
            if isinstance(y_pilot, np.ndarray):
                y_pilot = cp.asarray(y_pilot)
            if isinstance(x_pilot, np.ndarray):
                x_pilot = cp.asarray(x_pilot)
            if isinstance(c_pilot, np.ndarray):
                c_pilot = cp.asarray(c_pilot)
        
        N_rx, N_pilots = y_pilot.shape
        alpha = self.comm_power_factor
        beta = self.radar_power_factor
        alpha_sq = alpha ** 2
        beta_sq = beta ** 2
        
        n_iter = n_iterations if n_iterations is not None else self.ic_iterations
        
        if verbose:
            print(f"\n  CONFIGURATION:")
            print(f"    N_rx: {N_rx}, N_pilots: {N_pilots}")
            print(f"    α: {alpha:.6f}, α²: {alpha_sq:.6f}")
            print(f"    β: {beta:.6f}, β²: {beta_sq:.6f}")
            print(f"    Iterations: {n_iter}")
            print(f"    σ²_n: {self.sigma_n_sq:.6e}")
        
        # Track performance across iterations
        y_current = y_pilot.copy()
        H_rough = None
        
        for iteration in range(n_iter):
            if verbose:
                print(f"\n  {'━'*76}")
                print(f"  ITERATION {iteration + 1}/{n_iter}")
                print(f"  {'━'*76}")
            
            # ================================================================
            # STEP 1: Rough Channel Estimate (CORRECTED in v8.1)
            # ================================================================
            if verbose:
                print(f"\n    STEP 1: Rough Channel Estimate (CORRECTED)")
                print(f"      Formula: Ĥ[k] = (y[k]·x*[k]) / (α²·|x[k]|² + ε[k])")
                print(f"      ε[k] = β² + σ²_n = {beta_sq:.6f} + {self.sigma_n_sq:.6e}")
            
            H_rough = self._rough_ls_estimate_corrected(
                y_current, x_pilot, alpha_sq, beta_sq, verbose
            )
            
            # ================================================================
            # STEP 2: Reconstruct Radar Interference
            # ================================================================
            if verbose:
                print(f"\n    STEP 2: Reconstruct Interference")
                print(f"      Formula: ŷ_radar[k] = β·Ĥ[k]·c[k]")
            
            y_radar_est = self._reconstruct_interference(
                H_rough, c_pilot, beta, verbose
            )
            
            # ================================================================
            # STEP 3: Cancel Interference + Power Normalization
            # ================================================================
            if verbose:
                print(f"\n    STEP 3: Cancel Interference")
            
            y_clean = y_current - y_radar_est
            
            # POWER NORMALIZATION (STABILIZER)
            power_after = float(xp.mean(xp.abs(y_clean)**2))
            target_power = alpha_sq
            if power_after > 1e-12:
                y_clean = y_clean * xp.sqrt(target_power / power_after)
            
            # ================================================================
            # STEP 4: Log performance (FIXED CALL)
            # ================================================================
            if verbose:
                self._log_cancellation_performance(
                    y_pilot,          # y_before
                    y_current,        # y_before_this_iter
                    y_radar_est,      # y_interference
                    y_clean,          # y_after
                    alpha_sq,
                    beta_sq,
                    iteration + 1,
                    n_iter
                )
            
            # Update for next iteration
            y_current = y_clean
        
        if verbose:
            print(f"\n  {'='*76}")
            print(f"  FINAL RESULTS AFTER {n_iter} ITERATION(S)")
            print(f"  {'='*76}")
            
            original_power = float(xp.mean(xp.abs(y_pilot)**2))
            final_power = float(xp.mean(xp.abs(y_clean)**2))
            total_reduction = original_power - final_power
            total_gain_db = 10 * np.log10(original_power / (final_power + 1e-12))
            
            expected_final = alpha_sq + self.sigma_n_sq
            residual_interference = final_power - expected_final
            residual_ratio = residual_interference / beta_sq if beta_sq > 0 else 0.0
            
            print(f"    Original power: {original_power:.6e}")
            print(f"    Final power: {final_power:.6e}")
            print(f"    Total reduction: {total_reduction:.6e}")
            print(f"    Total gain: {total_gain_db:.2f} dB")
            print(f"    Expected final: {expected_final:.6e}")
            print(f"    Residual interference: {residual_interference:.6e}")
            print(f"    Residual ratio: {residual_ratio:.2%}")
            
            if n_iter == 1:
                print(f"\n    Expected gain (1 iter): ~8-9 dB")
            elif n_iter == 2:
                print(f"\n    Expected gain (2 iter): ~10-11 dB")
            else:
                print(f"\n    Expected gain (3 iter): ~11-12 dB")
            
            print(f"{'='*80}\n")
        
        return y_clean, H_rough
    
    def _rough_ls_estimate_corrected(
        self,
        y_pilot: Union[np.ndarray, 'cp.ndarray'],
        x_pilot: Union[np.ndarray, 'cp.ndarray'],
        alpha_sq: float,
        beta_sq: float,
        verbose: bool = False
    ) -> Union[np.ndarray, 'cp.ndarray']:

        xp = self.xp
        N_rx, N_pilots = y_pilot.shape
        
        # Compute per-pilot regularization: ε = β² + σ²_n
        epsilon_per_pilot = beta_sq + self.sigma_n_sq
        
        # Pilot power: |x_pilot[k]|²
        x_pilot_power = xp.abs(x_pilot) ** 2
        
        # CORRECTED denominator: α²·|x_pilot|² + ε
        denominator = alpha_sq * x_pilot_power + epsilon_per_pilot
        
        # Prevent division by zero
        denominator = xp.maximum(denominator, 1e-12)
        
        # LS estimate: H_rough = (y_pilot ⊙ conj(x_pilot)) / denominator
        H_rough = y_pilot * xp.conj(x_pilot)[xp.newaxis, :] / denominator[xp.newaxis, :]
        
        if verbose:
            h_power = float(xp.mean(xp.abs(H_rough)**2))
            x_power = float(xp.mean(x_pilot_power))
            denom_avg = float(xp.mean(denominator))
            
            print(f"        Input:")
            print(f"          y_pilot power: {float(xp.mean(xp.abs(y_pilot)**2)):.6e}")
            print(f"          x_pilot power: {x_power:.6e}")
            print(f"        Denominator:")
            print(f"          α²·|x|²: {float(xp.mean(alpha_sq * x_pilot_power)):.6e}")
            print(f"          ε (per-pilot): {epsilon_per_pilot:.6e}")
            print(f"          Total (avg): {denom_avg:.6e}")
            print(f"        Output:")
            print(f"          Ĥ_rough power: {h_power:.6e}")
            print(f"          Expected: ~1.0 ✅" if abs(h_power - 1.0) < 0.2 else f"          ⚠️  Power {h_power:.3f} deviates from 1.0")
        
        return H_rough  # (N_rx, N_pilots)
    
    def _reconstruct_interference(
        self,
        H_rough: Union[np.ndarray, 'cp.ndarray'],
        c_pilot: Union[np.ndarray, 'cp.ndarray'],
        beta: float,
        verbose: bool = False
    ) -> Union[np.ndarray, 'cp.ndarray']:
    
        xp = self.xp
    
        # Safe reconstruction with scaling to prevent over-cancellation
        # This prevents removing part of the desired communication signal
        y_radar = self.ic_reconstruction_scale * beta * H_rough * c_pilot[xp.newaxis, :]
    
        if verbose:
            radar_power = float(xp.mean(xp.abs(y_radar)**2))
            c_power = float(xp.mean(xp.abs(c_pilot)**2))
            h_power = float(xp.mean(xp.abs(H_rough)**2))
            expected_radar_power = (self.ic_reconstruction_scale * beta)**2 * h_power * c_power
        
            print(f"        Input:")
            print(f"          Ĥ_rough power: {h_power:.6e}")
            print(f"          c_pilot power: {c_power:.6e}")
            print(f"          β: {beta:.6f}")
            print(f"          Scale: {self.ic_reconstruction_scale:.2f} (safety factor)")
            print(f"        Output:")
            print(f"          Interference power: {radar_power:.6e}")
            print(f"          Expected: {expected_radar_power:.6e}")
            print(f"          (Scaled by {self.ic_reconstruction_scale:.2f} to prevent over-cancellation)")
    
        return y_radar  # (N_rx, N_pilots)
    
    def _log_cancellation_performance(
        self,
        y_original: Union[np.ndarray, 'cp.ndarray'],
        y_before: Union[np.ndarray, 'cp.ndarray'],
        y_interference: Union[np.ndarray, 'cp.ndarray'],
        y_after: Union[np.ndarray, 'cp.ndarray'],
        alpha_sq: float,
        beta_sq: float,
        iteration: int,
        total_iterations: int
    ):
        """Log detailed performance metrics for this iteration."""
        xp = self.xp
        
        power_original = float(xp.mean(xp.abs(y_original)**2))
        power_before = float(xp.mean(xp.abs(y_before)**2))
        power_interference = float(xp.mean(xp.abs(y_interference)**2))
        power_after = float(xp.mean(xp.abs(y_after)**2))
        
        reduction = power_before - power_after
        cancellation_ratio = reduction / power_interference if power_interference > 0 else 0
        gain_db = 10 * np.log10(power_before / (power_after + 1e-12))
        
        expected_clean = alpha_sq + self.sigma_n_sq
        residual = power_after - expected_clean
        
        print(f"\n    STEP 4: Performance (Iteration {iteration}/{total_iterations})")
        print(f"      Power before this iteration: {power_before:.6e}")
        print(f"      Estimated interference: {power_interference:.6e}")
        print(f"      Power after cancellation: {power_after:.6e}")
        print(f"      Reduction: {reduction:.6e}")
        print(f"      Cancellation ratio: {cancellation_ratio:.2%}")
        print(f"      Gain this iteration: {gain_db:.2f} dB")
        print(f"      Expected final power: {expected_clean:.6e}")
        print(f"      Residual interference: {residual:.6e}")
    
    # ========================================================================
    # MODIFIED: estimate_from_pilots() with Corrected IC Integration
    # ========================================================================
    
    def estimate_from_pilots(
        self,
        y_rx: Union[np.ndarray, 'cp.ndarray'],
        pilot_symbols: Union[np.ndarray, 'cp.ndarray'],
        pilot_mask: Union[np.ndarray, 'cp.ndarray'],
        n_symbols: int = 14,
        n_subcarriers: int = 1633,
        tx_antenna_idx: int = 0,
        chirp_freq: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
        **kwargs
    ) -> np.ndarray:
        
        print(f"\n{'─'*80}")
        print(f"[ESTIMATE FROM PILOTS v8.1 - CORRECTED IC]")
        print(f"{'─'*80}")
        
        xp = self.xp
        
        # Convert to GPU if needed
        if self.use_gpu:
            if isinstance(y_rx, np.ndarray):
                y_rx = cp.asarray(y_rx)
            if isinstance(pilot_symbols, np.ndarray):
                pilot_symbols = cp.asarray(pilot_symbols)
            if isinstance(pilot_mask, np.ndarray):
                pilot_mask = cp.asarray(pilot_mask)
            if chirp_freq is not None and isinstance(chirp_freq, np.ndarray):
                chirp_freq = cp.asarray(chirp_freq)
        
        print(f"  INPUT:")
        print(f"    y_rx: {y_rx.shape}, pilots: {pilot_symbols.shape}, mask: {pilot_mask.shape}")
        print(f"    chirp_freq: {'PROVIDED' if chirp_freq is not None else 'NOT PROVIDED'}")
        
        # Handle dimensions
        if y_rx.ndim == 2:
            n_rx = y_rx.shape[0]
            y_rx = y_rx[:, xp.newaxis, :]
            single_symbol = True
        elif y_rx.ndim == 3:
            n_rx, _, _ = y_rx.shape
            single_symbol = False
        else:
            raise ValueError(f"y_rx must be 2D or 3D, got {y_rx.ndim}D")
        
        # Extract pilots
        pilot_positions = xp.where(pilot_mask)
        pilot_symbol_idx = pilot_positions[0]
        pilot_freq_idx = pilot_positions[1]
        n_pilots = len(pilot_symbol_idx)
        
        print(f"  PILOTS: {n_pilots} positions extracted")
        
        y_pilots = xp.zeros((n_rx, n_pilots), dtype=xp.complex64)
        for rx_idx in range(n_rx):
            y_pilots[rx_idx, :] = y_rx[rx_idx, pilot_symbol_idx, pilot_freq_idx]
        
        y_pilots_power = float(xp.mean(xp.abs(y_pilots)**2))
        print(f"    Power before IC: {y_pilots_power:.6e}")
        
        # ====================================================================
        # APPLY CORRECTED INTERFERENCE CANCELLATION
        # ====================================================================
        
        if (self.enable_ic and 
            self.fmcw_integration_mode == 'additive' and 
            chirp_freq is not None):
            
            print(f"\n  {'='*76}")
            print(f"  APPLYING CORRECTED IC ({self.ic_iterations} iteration(s))")
            print(f"  {'='*76}")
            
            # Extract chirp at pilot positions
            c_pilots = chirp_freq[pilot_freq_idx]
            
            # Apply corrected cancellation
            y_pilots_clean, H_rough = self.cancel_radar_interference(
                y_pilots,
                pilot_symbols,
                c_pilots,
                n_iterations=self.ic_iterations,
                verbose=True
            )
            
            y_pilots = y_pilots_clean
            
            y_pilots_power_clean = float(xp.mean(xp.abs(y_pilots)**2))
            print(f"    Power after IC: {y_pilots_power_clean:.6e}")
            print(f"    Reduction: {y_pilots_power - y_pilots_power_clean:.6e}")
        else:
            if not self.enable_ic:
                print(f"\n  ⚠️  IC DISABLED")
            elif self.fmcw_integration_mode != 'additive':
                print(f"\n  ℹ️  IC not applicable (mode={self.fmcw_integration_mode})")
            elif chirp_freq is None:
                print(f"\n  ℹ️  IC not applied (no chirp provided)")
        
        # ====================================================================
        # STANDARD LS/MMSE ESTIMATION (on cleaned pilots)
        # ====================================================================
        
        print(f"\n  FINAL CHANNEL ESTIMATION:")
        
        # Use corrected regularized LS
        H_pilots = self._estimate_ls_on_pilots_corrected(
            y_pilots, pilot_symbols, kwargs.get('verbose', False)
        )
        
        # Interpolation
        n_tx = 4
        H_est = xp.zeros((n_rx, n_tx, n_subcarriers), dtype=xp.complex64)
        
        for rx_idx in range(n_rx):
            unique_freq_idx = xp.unique(pilot_freq_idx)
            H_avg_pilots = xp.zeros(len(unique_freq_idx), dtype=xp.complex64)
            
            for i, freq_idx in enumerate(unique_freq_idx):
                mask_freq = pilot_freq_idx == freq_idx
                H_avg_pilots[i] = xp.mean(H_pilots[rx_idx, mask_freq])
            
            h_interp = self._interpolate_1d(unique_freq_idx, H_avg_pilots, n_subcarriers)
            
            for tx_idx in range(n_tx):
                H_est[rx_idx, tx_idx, :] = h_interp
        
        # Post-processing
        if self.method == 'mmse':
            H_est = self._apply_wiener_filter(H_est)
        elif self.method == 'dft':
            H_est = self._apply_dft_denoising(H_est)
        
        # Convert back to CPU
        if self.use_gpu and isinstance(H_est, cp.ndarray):
            H_est = cp.asnumpy(H_est)
        
        H_final_power = np.mean(np.abs(H_est)**2)
        print(f"  OUTPUT: H_est shape={H_est.shape}, power={H_final_power:.6e}")
        print(f"{'─'*80}\n")
        
        return H_est
    
    def _estimate_ls_on_pilots_corrected(
        self,
        y_pilots: Union[np.ndarray, 'cp.ndarray'],
        pilot_symbols: Union[np.ndarray, 'cp.ndarray'],
        verbose: bool = False
    ) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Corrected LS estimation on (cleaned) pilots.
        
        Uses proper regularization: α²·|x|² + ε
        """
        xp = self.xp
        N_rx, N_pilots = y_pilots.shape
        
        pilot_power = xp.abs(pilot_symbols) ** 2
        
        if self.fmcw_integration_mode == 'additive':
            alpha_sq = self.comm_power_factor ** 2
            epsilon = self.epsilon_base
            denominator = alpha_sq * pilot_power + epsilon
        else:
            denominator = pilot_power + self.epsilon_base
        
        H_pilots = xp.zeros((N_rx, N_pilots), dtype=xp.complex64)
        
        for rx_idx in range(N_rx):
            H_pilots[rx_idx, :] = (
                y_pilots[rx_idx, :] * xp.conj(pilot_symbols) / denominator
            )
        
        if verbose:
            print(f"    LS on cleaned pilots: H_power = {float(xp.mean(xp.abs(H_pilots)**2)):.6e}")
        
        return H_pilots
    
    # ========================================================================
    # EXISTING METHODS (kept from v7.0/v8.0, no changes needed)
    # ========================================================================
    
    def _apply_wiener_filter(
        self,
        H_est: Union[np.ndarray, 'cp.ndarray']
    ) -> Union[np.ndarray, 'cp.ndarray']:
        """Wiener filter (unchanged from v8.0)."""
        xp = self.xp
        n_rx, n_tx, n_sc = H_est.shape
        
        H_power_per_sc = xp.mean(xp.abs(H_est) ** 2, axis=(0, 1))
        sigma_H_sq = xp.maximum(H_power_per_sc, 1e-12)
        
        H_mmse = xp.zeros_like(H_est)
        
        if self.fmcw_integration_mode == 'additive':
            beta_sq = self.radar_power_factor ** 2
            weights = 1.0 / (1.0 + beta_sq + self.sigma_n_sq / sigma_H_sq)
            
            for k in range(n_sc):
                H_mmse[:, :, k] = weights[k] * H_est[:, :, k]
        else:
            weights = sigma_H_sq / (sigma_H_sq + self.sigma_n_sq)
            
            for k in range(n_sc):
                H_mmse[:, :, k] = weights[k] * H_est[:, :, k]
        
        return H_mmse
    
    def estimate(
        self,
        y_rx: Union[np.ndarray, 'cp.ndarray'],
        x_tx: Union[np.ndarray, 'cp.ndarray'],
        pilot_indices: Optional[Union[np.ndarray, 'cp.ndarray']] = None
    ) -> np.ndarray:
        """Standard estimate method (unchanged)."""
        if self.use_gpu:
            if isinstance(y_rx, np.ndarray):
                y_rx = cp.asarray(y_rx)
            if isinstance(x_tx, np.ndarray):
                x_tx = cp.asarray(x_tx)
            if pilot_indices is not None and isinstance(pilot_indices, np.ndarray):
                pilot_indices = cp.asarray(pilot_indices)
        
        H_est = self._estimate_ls(y_rx, x_tx, pilot_indices)
        
        if self.method == 'mmse':
            H_est = self._apply_wiener_filter(H_est)
        elif self.method == 'dft':
            H_est = self._apply_dft_denoising(H_est)
        
        if self.use_gpu and isinstance(H_est, cp.ndarray):
            H_est = cp.asnumpy(H_est)
        
        return H_est
    
    def _estimate_ls(
        self,
        y_rx: Union[np.ndarray, 'cp.ndarray'],
        x_tx: Union[np.ndarray, 'cp.ndarray'],
        pilot_indices: Optional[Union[np.ndarray, 'cp.ndarray']] = None
    ) -> Union[np.ndarray, 'cp.ndarray']:
        """Legacy LS estimation (unchanged from v7.0)."""
        xp = self.xp
        
        if y_rx.ndim == 2:
            n_rx, n_sc = y_rx.shape
            single_symbol = True
        else:
            n_rx, n_sc, n_sym = y_rx.shape
            single_symbol = False
        
        if x_tx.ndim == 2:
            n_tx, _ = x_tx.shape
        else:
            n_tx, _, _ = x_tx.shape
        
        if pilot_indices is None:
            x_power = xp.abs(x_tx) ** 2 if x_tx.ndim == 2 else xp.abs(x_tx[:, :, 0]) ** 2
            
            if single_symbol:
                H_est = xp.zeros((n_rx, n_tx, n_sc), dtype=complex)
                
                for rx_idx in range(n_rx):
                    for tx_idx in range(n_tx):
                        if self.fmcw_integration_mode == 'additive':
                            denom = (self.comm_power_factor ** 2) * x_power[tx_idx, :] + self.epsilon_base
                        else:
                            denom = x_power[tx_idx, :] + self.epsilon_base
                        
                        x_use = x_tx[tx_idx, :]
                        y_use = y_rx[rx_idx, :]
                        H_est[rx_idx, tx_idx, :] = y_use * xp.conj(x_use) / denom
            else:
                H_est = xp.zeros((n_rx, n_tx, n_sc, n_sym), dtype=complex)
                
                for sym_idx in range(n_sym):
                    y_sym = y_rx[:, :, sym_idx]
                    x_sym = x_tx[:, :, sym_idx] if x_tx.ndim == 3 else x_tx
                    x_power_sym = xp.abs(x_sym) ** 2
                    
                    for rx_idx in range(n_rx):
                        for tx_idx in range(n_tx):
                            if self.fmcw_integration_mode == 'additive':
                                denom = (self.comm_power_factor ** 2) * x_power_sym[tx_idx, :] + self.epsilon_base
                            else:
                                denom = x_power_sym[tx_idx, :] + self.epsilon_base
                            
                            H_est[rx_idx, tx_idx, :, sym_idx] = (
                                y_sym[rx_idx, :] * xp.conj(x_sym[tx_idx, :]) / denom
                            )
        else:
            n_pilots = len(pilot_indices)
            H_pilots = xp.zeros((n_rx, n_tx, n_pilots), dtype=complex)
            x_power_pilots = xp.abs(x_tx[:, pilot_indices]) ** 2
            
            for rx_idx in range(n_rx):
                for tx_idx in range(n_tx):
                    if self.fmcw_integration_mode == 'additive':
                        denom = (self.comm_power_factor ** 2) * x_power_pilots[tx_idx, :] + self.epsilon_base
                    else:
                        denom = x_power_pilots[tx_idx, :] + self.epsilon_base
                    
                    H_pilots[rx_idx, tx_idx, :] = (
                        y_rx[rx_idx, pilot_indices] * xp.conj(x_tx[tx_idx, pilot_indices]) / denom
                    )
            
            H_est = self._interpolate_channel(H_pilots, pilot_indices, n_sc)
        
        return H_est
    
    def _interpolate_1d(
        self,
        x_known: Union[np.ndarray, 'cp.ndarray'],
        y_known: Union[np.ndarray, 'cp.ndarray'],
        n_points: int
    ) -> Union[np.ndarray, 'cp.ndarray']:
        """Linear interpolation (unchanged)."""
        if self.use_gpu:
            x_known = cp.asnumpy(x_known)
            y_known = cp.asnumpy(y_known)
        
        x_all = np.arange(n_points)
        y_real = np.interp(x_all, x_known, y_known.real)
        y_imag = np.interp(x_all, x_known, y_known.imag)
        y_interp = y_real + 1j * y_imag
        
        if self.use_gpu:
            y_interp = cp.asarray(y_interp)
        
        return y_interp
    
    def _interpolate_channel(
        self,
        H_pilots: Union[np.ndarray, 'cp.ndarray'],
        pilot_indices: Union[np.ndarray, 'cp.ndarray'],
        n_sc: int
    ) -> Union[np.ndarray, 'cp.ndarray']:
        """Channel interpolation (unchanged)."""
        xp = self.xp
        n_rx, n_tx, n_pilots = H_pilots.shape
        H_interp = xp.zeros((n_rx, n_tx, n_sc), dtype=complex)
        
        if self.use_gpu:
            pilot_indices_cpu = cp.asnumpy(pilot_indices)
            H_pilots_cpu = cp.asnumpy(H_pilots)
        else:
            pilot_indices_cpu = pilot_indices
            H_pilots_cpu = H_pilots
        
        for rx_idx in range(n_rx):
            for tx_idx in range(n_tx):
                h_interp = self._interpolate_1d(
                    pilot_indices_cpu,
                    H_pilots_cpu[rx_idx, tx_idx, :],
                    n_sc
                )
                
                if self.use_gpu and isinstance(h_interp, np.ndarray):
                    H_interp[rx_idx, tx_idx, :] = cp.asarray(h_interp)
                else:
                    H_interp[rx_idx, tx_idx, :] = h_interp
        
        return H_interp
    
    def _apply_dft_denoising(
        self,
        H_est: Union[np.ndarray, 'cp.ndarray']
    ) -> Union[np.ndarray, 'cp.ndarray']:
        """DFT denoising (unchanged)."""
        xp = self.xp
        n_rx, n_tx, n_sc = H_est.shape
        H_dft = xp.zeros_like(H_est)
        
        for rx_idx in range(n_rx):
            for tx_idx in range(n_tx):
                h = xp.fft.ifft(H_est[rx_idx, tx_idx, :], n=self.n_fft)
                
                h_clean = xp.zeros(self.n_fft, dtype=complex)
                h_clean[:self.max_delay_taps] = h[:self.max_delay_taps]
                
                H_clean = xp.fft.fft(h_clean, n=self.n_fft)
                H_dft[rx_idx, tx_idx, :] = H_clean[:n_sc]
        
        return H_dft
    
    def estimate_noise_power(
        self,
        H_est: Union[np.ndarray, 'cp.ndarray'],
        y_rx: Union[np.ndarray, 'cp.ndarray'],
        x_tx: Union[np.ndarray, 'cp.ndarray']
    ) -> float:
        """Noise power estimation (unchanged from v7.0)."""
        xp = self.xp
        
        if isinstance(H_est, np.ndarray) and self.use_gpu:
            H_est = cp.asarray(H_est)
        if isinstance(y_rx, np.ndarray) and self.use_gpu:
            y_rx = cp.asarray(y_rx)
        if isinstance(x_tx, np.ndarray) and self.use_gpu:
            x_tx = cp.asarray(x_tx)
        
        n_rx = H_est.shape[0]
        n_sc = H_est.shape[2]
        
        if self.fmcw_integration_mode == 'additive':
            x_tx_scaled = self.comm_power_factor * x_tx
        else:
            x_tx_scaled = x_tx
        
        if y_rx.ndim == 3:
            y_use = xp.mean(y_rx, axis=1)
        else:
            y_use = y_rx
        
        y_recon = xp.zeros((n_rx, n_sc), dtype=xp.complex64)
        
        for k in range(n_sc):
            if x_tx.ndim == 2:
                x_use = x_tx_scaled[:, k]
            elif x_tx.ndim == 1:
                x_use = x_tx_scaled[xp.newaxis] if k == 0 else x_tx_scaled[xp.newaxis] * 0
            else:
                x_use = x_tx_scaled[:, k] if k < x_tx.shape[1] else x_tx_scaled[:, 0]
            
            y_recon[:, k] = H_est[:, :, k] @ x_use
        
        error = y_use - y_recon
        error_power = xp.abs(error) ** 2
        
        median_error_power = float(xp.median(error_power))
        
        if self.fmcw_integration_mode == 'additive':
            beta_sq = self.radar_power_factor ** 2
            thermal_noise = median_error_power / (1.0 + beta_sq)
        else:
            thermal_noise = median_error_power
        
        thermal_noise = max(thermal_noise, 1e-12)
        
        return thermal_noise
    
    def compute_channel_quality(self, H_est: Union[np.ndarray, 'cp.ndarray']) -> dict:
        """Channel quality metrics (unchanged)."""
        xp = self.xp if isinstance(H_est, (cp.ndarray if GPU_AVAILABLE else type(None))) else np
        
        if isinstance(H_est, np.ndarray) and self.use_gpu:
            H_est = cp.asarray(H_est)
        
        if H_est.ndim == 3:
            n_rx, n_tx, n_sc = H_est.shape
        else:
            n_rx, n_tx, n_sc, _ = H_est.shape
            H_est = xp.mean(H_est, axis=3)
        
        channel_gains = xp.sum(xp.abs(H_est) ** 2, axis=(0, 1))
        avg_gain = float(xp.mean(channel_gains))
        
        return {
            'channel_gain_db': 10 * np.log10(avg_gain + 1e-12),
            'method': self.method,
            'isac_mode': self.fmcw_integration_mode,
            'comm_power_fraction': self.comm_power_fraction if self.fmcw_integration_mode == 'additive' else None,
            'interference_cancellation': self.enable_ic,
            'ic_iterations': self.ic_iterations if self.enable_ic else 0,
        }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("CHANNEL ESTIMATION v8.1 - MATHEMATICALLY CORRECTED IC")
    print("="*80)