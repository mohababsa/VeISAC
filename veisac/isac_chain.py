# isac_chain.py
"""
VeISAC — End-to-End ISAC Chain Orchestrator

Top-level pipeline coordinating the full MIMO-OFDM-FMCW ISAC signal chain: TX waveform generation, channel application, Com-RX and Sen-RX processing, and joint performance evaluation across monostatic and bistatic topologies

Paper: "VeISAC: An End-to-End MIMO-OFDM-FMCW Framework for ISAC
        in 6G Vehicular Networks"
Authors: M. Ababsa, S. Ribouh, Y. El Hillali, A. Rivenq
"""

import numpy as np
from typing import Optional, Dict, Tuple, List, Union
import warnings
import logging
import sys
from datetime import datetime
from pathlib import Path

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np

from veisac.tx.isac_transmitter import ISACTransmitter
from veisac.tx.isac_tx_config import ISACTXConfig
from veisac.rx.comm_receiver import CommunicationReceiver
from veisac.rx.sensing_receiver import SensingReceiver
from veisac.rx.isac_rx_config import ISACRXConfig

LIGHTSPEED = 299792458.0


class DualLogger:
    def __init__(self, log_file=None, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"isac_chain_{timestamp}.log"
        
        self.log_path = self.log_dir / log_file
        self.logger = logging.getLogger('ISACChain')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        
        file_handler = logging.FileHandler(self.log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"{'='*80}")
        self.logger.info(f"ISAC CHAIN v8.5 - CORRECTED RADAR + OFDM MITIGATION")
        self.logger.info(f"{'='*80}")
        self.info(f"Log: {self.log_path}")
        self.info(f"{'='*80}\n")
    
    def debug(self, msg): self.logger.debug(msg)
    def info(self, msg): self.logger.info(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)
    
    def close(self):
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers = []


class ISACChain:
    
    def __init__(self, tx_config, rx_config, use_gpu=True, verbose=False,
                 log_file=None, log_dir="logs", fmcw_integration_mode='additive',
                 comm_power_factor=np.sqrt(0.5), radar_power_factor=np.sqrt(0.5),
                 enable_interference_cancellation=True,
                 ic_iterations=2):
        
        print(f"\n{'='*80}")
        print(f"[ISAC CHAIN v8.5 - CORRECTED RADAR + ACTUAL TX WAVEFORM]")
        print(f"{'='*80}")
        
        self.tx_config = tx_config
        self.rx_config = rx_config
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.verbose = True
        self.fmcw_integration_mode = fmcw_integration_mode
        self.comm_power_factor = comm_power_factor
        self.radar_power_factor = radar_power_factor
        self.enable_ic = enable_interference_cancellation
        self.ic_iterations = ic_iterations
        
        print(f"\n  ISAC CONFIGURATION:")
        print(f"    Mode: {fmcw_integration_mode.upper()}")
        print(f"    Power factors: α={comm_power_factor:.6f}, β={radar_power_factor:.6f}")
        print(f"    Power fractions: α²={comm_power_factor**2:.6f}, β²={radar_power_factor**2:.6f}")
        
        power_sum = comm_power_factor**2 + radar_power_factor**2
        print(f"    Sum check: α²+β²={power_sum:.6f} (should be 1.0)")
        
        if abs(power_sum - 1.0) > 1e-6:
            print(f"    ⚠️  Power constraint violated!")
            warnings.warn(f"Power constraint violated: α²+β²={power_sum:.6f}")
        else:
            print(f"    ✅ Power constraint satisfied")
        
        print(f"\n  INTERFERENCE CANCELLATION:")
        print(f"    Enabled: {'YES' if enable_interference_cancellation else 'NO'}")
        if enable_interference_cancellation and fmcw_integration_mode == 'additive':
            print(f"    Pilot IC Iterations: {ic_iterations}")
            print(f"    Data-Domain IC: ENABLED (scale=1.06, threshold=40%)")
            print(f"    Expected combined benefit: Significant BER reduction")
        
        self.logger = DualLogger(log_file=log_file, log_dir=log_dir)
        
        self._validate_configs()
        self._initialize_transmitter()
        self._initialize_receivers()
        
        if self.verbose:
            self._print_initialization()
        
        print(f"\n{'='*80}")
        print(f"[INITIALIZATION COMPLETE]")
        print(f"{'='*80}\n")
    
    def _initialize_transmitter(self):
        print(f"\n{'─'*80}")
        print(f"[TRANSMITTER INITIALIZATION]")
        print(f"{'─'*80}")
        
        self.logger.info(f"\n[TRANSMITTER INIT]")
        self.logger.info(f"  Mode: {self.fmcw_integration_mode.upper()}")
        if self.fmcw_integration_mode == 'additive':
            self.logger.info(f"  α²={self.comm_power_factor**2*100:.1f}%, β²={self.radar_power_factor**2*100:.1f}%")
        
        print(f"  Creating ISACTransmitter...")
        print(f"    Integration: {self.fmcw_integration_mode}")
        print(f"    α={self.comm_power_factor:.6f}, β={self.radar_power_factor:.6f}")
        
        self.tx = ISACTransmitter(
            config=self.tx_config,
            verbose=False,
            fmcw_integration=self.fmcw_integration_mode,
            alpha=self.comm_power_factor,
            beta=self.radar_power_factor
        )
        
        print(f"  ✅ Transmitter ready")
        self.logger.info(f"  ✅ TX ready")
    
    def _initialize_receivers(self):
        print(f"\n{'─'*80}")
        print(f"[RECEIVER INITIALIZATION]")
        print(f"{'─'*80}")
        
        self.logger.info(f"\n[RECEIVER INIT]")
        
        try:
            print(f"\n  Creating CommunicationReceiver v8.3_FINAL...")
            print(f"    Channel estimation: MMSE")
            print(f"    Equalization: MMSE")
            print(f"    ISAC mode: {self.fmcw_integration_mode}")
            print(f"    Power: α={self.comm_power_factor:.6f}, β={self.radar_power_factor:.6f}")
            print(f"    Pilot IC enabled: {self.enable_ic}")
            if self.enable_ic and self.fmcw_integration_mode == 'additive':
                print(f"    Pilot IC iterations: {self.ic_iterations}")
                print(f"    Data-Domain IC: ENABLED (scale=1.06, threshold=40%)")
            
            self.comm_rx = CommunicationReceiver(
                config=self.rx_config,
                channel_est_method='mmse',
                equalization_method='mmse',
                fmcw_integration_mode=self.fmcw_integration_mode,
                comm_power_factor=self.comm_power_factor,
                radar_power_factor=self.radar_power_factor,
                enable_interference_cancellation=self.enable_ic,
                ic_iterations=self.ic_iterations,
                use_gpu=self.use_gpu,
                verbose=False
            )
            
            print(f"  ✅ COMM RX ready (v8.3_FINAL - Safe Data-Domain IC)")
            self.logger.info("  ✅ COMM RX v8.3_FINAL (Safe Data-Domain IC)")
            
        except Exception as e:
            print(f"  ⚠️  COMM RX unavailable: {e}")
            self.logger.warning(f"  ⚠️  COMM RX unavailable: {e}")
            self.comm_rx = None
        
        print(f"\n  Creating SensingReceiver v4.2...")
        print(f"    Mode:          {self.rx_config.radar_mode.upper()}")
        _n_tx_r = self.rx_config.n_radar_tx_antennas
        _n_rx_r = self.rx_config.n_radar_rx_antennas
        if _n_rx_r == 4:
            _rx_lbl = ('BS 2×2 UPA (monostatic)'
                       if self.rx_config.radar_mode == 'monostatic'
                       else 'BS 2×2 UPA (bistatic)')
        else:
            _rx_lbl = f'UE 2×{_n_rx_r // 2} ULA'
        print(f"    TX antennas:   {_n_tx_r}  (BS 2×2 UPA)")
        print(f"    RX antennas:   {_n_rx_r}  ({_rx_lbl})")
        print(f"    Virtual array: {self.rx_config.virtual_array_size} elements  "
              f"({_n_tx_r}×{_n_rx_r} Kronecker)")
        print(f"    β (radar power factor): {self.radar_power_factor:.6f}")
        
        self.sens_rx = SensingReceiver(
            config=self.rx_config,
            use_gpu=self.use_gpu,
            verbose=False
        )

        print(f"  ✅ SENS RX ready (v4.2 - Corrected)")
        print(f"    Multi-target dataset compatible")
        self.logger.info("  ✅ SENS RX v4.2 (Corrected - multi-target ready)")
    
    def _print_initialization(self):
        self.logger.info(f"\n[SYSTEM CONFIG]")
        self.logger.info(f"  TX ant (COMM): {self.tx_config.n_tx_antennas}")
        self.logger.info(f"  RX ant (COMM): {self.rx_config.n_rx_antennas}")
        self.logger.info(f"  Virtual array: {self.rx_config.virtual_array_size}")
        self.logger.info(f"  Subcarriers: {self.tx_config.n_subcarriers_actual}")
        self.logger.info(f"  Power: α²={self.comm_power_factor**2*100:.1f}%, β²={self.radar_power_factor**2*100:.1f}%")
        self.logger.info(f"  IC: {'ON' if self.enable_ic else 'OFF'}")
        if self.enable_ic:
            self.logger.info(f"  IC iterations: {self.ic_iterations}")
        self.logger.info(f"  Device: {'GPU' if self.use_gpu else 'CPU'}")
    
    def set_power_allocation(self, comm_power_factor, radar_power_factor):
        power_sum = comm_power_factor**2 + radar_power_factor**2
        if abs(power_sum - 1.0) > 1e-6:
            warnings.warn(f"Power constraint violated: α²+β²={power_sum:.6f}")
        
        self.comm_power_factor = comm_power_factor
        self.radar_power_factor = radar_power_factor
        
        self.tx.set_power_allocation(comm_power_factor, radar_power_factor)
        
        if self.comm_rx is not None:
            self.comm_rx.comm_power_factor = comm_power_factor
            self.comm_rx.radar_power_factor = radar_power_factor
            self.comm_rx.channel_estimator.comm_power_factor = comm_power_factor
            self.comm_rx.channel_estimator.radar_power_factor = radar_power_factor
            self.comm_rx.equalizer.comm_power_factor = comm_power_factor
            self.comm_rx.equalizer.radar_power_factor = radar_power_factor
        
        self.logger.info(f"\n[POWER UPDATE] α²={comm_power_factor**2*100:.1f}%, β²={radar_power_factor**2*100:.1f}%")
    
    def _validate_configs(self):
        print(f"\n{'─'*80}")
        print(f"[CONFIG VALIDATION]")
        print(f"{'─'*80}")
        
        issues = []
        
        print(f"  Checking TX/RX compatibility...")
        
        if self.tx_config.n_subcarriers_actual != self.rx_config.n_subcarriers_actual:
            issues.append(f"Subcarrier mismatch: TX={self.tx_config.n_subcarriers_actual}, RX={self.rx_config.n_subcarriers_actual}")
        else:
            print(f"    ✅ Subcarriers: {self.tx_config.n_subcarriers_actual}")
        
        if self.tx_config.n_symbols_per_slot != self.rx_config.n_symbols_per_slot:
            issues.append(f"Symbol mismatch: TX={self.tx_config.n_symbols_per_slot}, RX={self.rx_config.n_symbols_per_slot}")
        else:
            print(f"    ✅ Symbols: {self.tx_config.n_symbols_per_slot}")
        
        if self.tx_config.fmcw_n_samples_per_chirp != self.rx_config.fmcw_n_samples_per_chirp:
            issues.append(f"FMCW samples mismatch")
        else:
            print(f"    ✅ FMCW samples: {self.tx_config.fmcw_n_samples_per_chirp}")
        
        if self.tx_config.fmcw_n_chirps != self.rx_config.fmcw_n_chirps:
            issues.append(f"FMCW chirps mismatch")
        else:
            print(f"    ✅ FMCW chirps: {self.tx_config.fmcw_n_chirps}")
        
        if issues:
            error_msg = "Config mismatch: " + ", ".join(issues)
            print(f"\n  ❌ VALIDATION FAILED:")
            for issue in issues:
                print(f"     {issue}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            print(f"\n  ✅ All configs validated")
            self.logger.info(f"✅ Config validated")
    
    def process_sample_synthetic(self, comm_snr_db=20.0, radar_snr_db=25.0,
                            target_range_m=150.0, target_velocity_ms=30.0,
                            target_gts: Optional[List[Dict]] = None,
                            seed=42, use_oracle_channel=True,
                            tx_power_dbm=43.0, verbose_override=None,
                            enable_angle_estimation=False):
        
        verbose = verbose_override if verbose_override is not None else self.verbose
        np.random.seed(seed)
        
        print(f"\n{'='*80}")
        print(f"[PROCESS SAMPLE v8.4 - CORRECTED RADAR SENSING]")
        print(f"{'='*80}")
        
        if verbose:
            self.logger.info(f"\n[PROCESS SAMPLE]")
            self.logger.info(f"  COMM SNR: {comm_snr_db} dB")
            self.logger.info(f"  RADAR SNR: {radar_snr_db} dB")
            self.logger.info(f"  Target: {target_range_m}m, {target_velocity_ms:+.2f}m/s")
        
        print(f"\n  PARAMETERS:")
        print(f"    COMM SNR target: {comm_snr_db} dB")
        print(f"    RADAR SNR target: {radar_snr_db} dB")
        print(f"    Target range: {target_range_m} m")
        print(f"    Target velocity: {target_velocity_ms} m/s")
        if target_gts is not None:
            print(f"    Multi-target mode: {len(target_gts)} targets provided")
            print(f"    Closest target (for eval): {target_range_m:.2f}m")
        else:
            print(f"    Single-target synthetic mode")
        print(f"    TX power: {tx_power_dbm} dBm")
        print(f"    Oracle channel: {use_oracle_channel}")
        print(f"    IC enabled: {self.enable_ic}")
        if self.enable_ic and self.fmcw_integration_mode == 'additive':
            print(f"    Pilot IC iterations: {self.ic_iterations}")
            print(f"    Data-Domain IC: ENABLED")
            print(f"    Expected benefit: Significant BER reduction")
        print(f"    Angle estimation: {'ENABLED' if enable_angle_estimation else 'DISABLED'}")
        print(f"    Seed: {seed}")
        
        print(f"\n{'─'*80}")
        print(f"[STEP 1: GENERATE SYNTHETIC CHANNEL]")
        print(f"{'─'*80}")
        
        H_comm = self._generate_synthetic_comm_channel()
        h_power = np.mean(np.abs(H_comm)**2)
        
        print(f"  H_comm shape: {H_comm.shape}")
        print(f"  H_comm power: {h_power:.6e}")
        print(f"  Status: {'✅ Normalized' if abs(h_power - 1.0) < 0.1 else '⚠️  Not normalized'}")
        
        results = {
            'fmcw_integration_mode': self.fmcw_integration_mode,
            'comm_power_factor': self.comm_power_factor,
            'radar_power_factor': self.radar_power_factor,
            'comm_power_fraction': self.comm_power_factor ** 2,
            'radar_power_fraction': self.radar_power_factor ** 2,
            'ic_enabled': self.enable_ic,
            'ic_iterations': self.ic_iterations if self.enable_ic else 0,
        }
        
        print(f"\n{'─'*80}")
        print(f"[STEP 2: TRANSMIT SLOT]")
        print(f"{'─'*80}")
        
        print(f"  Calling transmitter...")
        tx_dict = self.tx.transmit_slot(n_ue=0, seed=seed)
        
        print(f"  TX output:")
        print(f"    freq_grid shape: {tx_dict['freq_grid'].shape}")
        print(f"    pilot_mask shape: {tx_dict['pilot_mask'].shape}")
        
        if 'pilot_symbols' not in tx_dict:
            pilot_mask = tx_dict['pilot_mask']
            tx_freq_grid = tx_dict['freq_grid']
            pilot_positions = np.where(pilot_mask)
            pilot_symbols = tx_freq_grid[pilot_positions]
            tx_dict['pilot_symbols'] = pilot_symbols
            print(f"    pilot_symbols extracted: {len(pilot_symbols)}")
        
        results['tx_power_w'] = tx_dict['metadata']['actual_power_per_antenna_w']
        results['tx_papr_db'] = tx_dict['metadata']['papr_db']
        
        print(f"    TX power: {results['tx_power_w']:.6e} W")
        print(f"    PAPR: {results['tx_papr_db']:.2f} dB")
        
        print(f"\n{'─'*80}")
        print(f"[STEP 2.5: GENERATE FMCW CHIRP FOR IC]")
        print(f"{'─'*80}")
        
        print(f"\n  Mathematical Model:")
        print(f"    Time-domain chirp: c(t) = exp(j·π·μ·t²)")
        print(f"    Frequency-domain: C[k] = FFT(c(t))")
        print(f"    Used for IC reconstruction: ŷ_radar = β·Ĥ·C[k]")
        
        chirp_freq = self._generate_fmcw_chirp_freq_domain()
        
        chirp_power = np.mean(np.abs(chirp_freq)**2)
        print(f"\n  FMCW Chirp (Frequency Domain):")
        print(f"    chirp_freq shape: {chirp_freq.shape}")
        print(f"    chirp_freq power: {chirp_power:.6e}")
        print(f"    Expected: ~1.0 (normalized)")
        print(f"    Status: {'✅ OK' if abs(chirp_power - 1.0) < 0.2 else '⚠️  CHECK'}")
        
        if self.enable_ic and self.fmcw_integration_mode == 'additive':
            print(f"\n  ✅ Chirp will be PASSED to COMM RX for IC")
        else:
            print(f"\n  ℹ️  Chirp available but IC disabled")
        
        if verbose:
            self.logger.info(f"  ✅ TX complete (with chirp)")
        
        if self.comm_rx is not None:
            print(f"\n{'─'*80}")
            print(f"[STEP 3: COMMUNICATION PROCESSING WITH IC]")
            print(f"{'─'*80}")
            
            print(f"\n  Generating OFDM waveform...")
            
            tx_freq_grid_flat = tx_dict['freq_grid']
            n_sym = self.tx_config.n_symbols_per_slot
            tx_freq_grid_single_tx = tx_freq_grid_flat[:n_sym, :]
            
            print(f"    TX freq grid (single TX): {tx_freq_grid_single_tx.shape}")
            
            print(f"\n  {'─'*76}")
            print(f"  CRITICAL: THERMAL NOISE COMPUTATION (COMM-REFERENCED)")
            print(f"  {'─'*76}")
            
            print(f"\n  Mathematical Model:")
            print(f"    y = α·H·x_comm + β·H·c + n")
            print(f"    Signal power (COMM only): P_comm = α²·|H|²·|x|²")
            print(f"    Thermal noise: σ²_n = P_comm / SNR_linear")
            print(f"    CRITICAL: Referenced to COMM signal, not total!")
            
            print(f"\n  Step 1: Measure clean COMM signal power...")
            y_temp = self._generate_ofdm_waveform_from_freq(
                tx_freq_grid_single_tx, H_comm, comm_snr_db=100.0, verbose=False
            )
            signal_power_total = np.mean(np.abs(y_temp)**2)
            
            if self.fmcw_integration_mode == 'additive':
                alpha_sq = self.comm_power_factor ** 2
                signal_power_comm = alpha_sq * signal_power_total / (alpha_sq + self.radar_power_factor**2)
                
                print(f"    Total RX power (clean): {signal_power_total:.6e}")
                print(f"    α²: {alpha_sq:.6f}")
                print(f"    β²: {self.radar_power_factor**2:.6f}")
                print(f"    COMM power (α² fraction): {signal_power_comm:.6e}")
            else:
                signal_power_comm = signal_power_total
                print(f"    Signal power: {signal_power_comm:.6e}")
            
            print(f"\n  Step 2: Compute thermal noise from COMM signal...")
            snr_linear = 10 ** (comm_snr_db / 10)
            noise_power_thermal = signal_power_comm / snr_linear
            
            print(f"    SNR target: {comm_snr_db} dB")
            print(f"    SNR linear: {snr_linear:.2f}")
            print(f"    Signal power (COMM): {signal_power_comm:.6e}")
            print(f"    Thermal noise (σ²_n): {noise_power_thermal:.6e}")
            
            if self.fmcw_integration_mode == 'additive':
                print(f"\n  Step 3: Theoretical validation...")
                expected_thermal = alpha_sq / snr_linear
                print(f"    Expected (α²/SNR): {expected_thermal:.6e}")
                print(f"    Measured: {noise_power_thermal:.6e}")
                print(f"    Match: {'✅' if abs(noise_power_thermal - expected_thermal) < 0.001 else '⚠️'}")
                
                beta_sq = self.radar_power_factor ** 2
                interference = beta_sq * np.mean(np.abs(H_comm)**2)
                expected_snr_no_ic = 1.0 / ((1/alpha_sq) * (noise_power_thermal + interference))
                expected_snr_no_ic_db = 10 * np.log10(expected_snr_no_ic)
                
                print(f"\n  Step 4: Expected performance WITHOUT IC...")
                print(f"    Full interference (β²|H|²): {interference:.6e}")
                print(f"    Total eff noise: {(1/alpha_sq) * (noise_power_thermal + interference):.6e}")
                print(f"    Expected SNR_eff: {expected_snr_no_ic_db:.2f} dB")
                
                if self.enable_ic:
                    residual_ratio = 0.1
                    residual_interference = residual_ratio * interference
                    expected_snr_with_ic = 1.0 / ((1/alpha_sq) * (noise_power_thermal + residual_interference))
                    expected_snr_with_ic_db = 10 * np.log10(expected_snr_with_ic)
                    expected_gain_db = expected_snr_with_ic_db - expected_snr_no_ic_db
                    
                    print(f"\n  Step 5: Expected performance WITH IC...")
                    print(f"    Assumed cancellation: 90%")
                    print(f"    Residual interference: {residual_interference:.6e}")
                    print(f"    Expected SNR_eff: {expected_snr_with_ic_db:.2f} dB")
                    print(f"    Expected IC gain: {expected_gain_db:.2f} dB ✅")
            
            print(f"  {'─'*76}\n")
            
            print(f"  Generating noisy waveform...")
            y_waveform = self._generate_ofdm_waveform_from_freq(
                tx_freq_grid_single_tx, H_comm, comm_snr_db,
                noise_power_override=noise_power_thermal, verbose=False
            )
            
            y_power = np.mean(np.abs(y_waveform)**2)
            print(f"    y_waveform shape: {y_waveform.shape}")
            print(f"    y_waveform power: {y_power:.6e}")
            
            print(f"\n  {'='*76}")
            print(f"  CALLING COMM RECEIVER v8.3_FINAL (Safe Data-Domain IC)")
            print(f"  {'='*76}")
            
            print(f"\n  Parameters:")
            print(f"    tx_power_dbm: {tx_power_dbm}")
            print(f"    noise_power_actual: {noise_power_thermal:.6e}")
            print(f"    target_snr_db: {comm_snr_db}")
            print(f"    use_oracle: {use_oracle_channel}")
            print(f"    chirp_freq: {'PROVIDED' if self.enable_ic else 'NOT PROVIDED'} (shape={chirp_freq.shape if self.enable_ic else 'N/A'})")
            
            if self.enable_ic and self.fmcw_integration_mode == 'additive':
                print(f"\n  ✅ TWO-STAGE IC WILL BE APPLIED:")
                print(f"    Stage 1: Pilot IC (channel estimation)")
                print(f"    Stage 2: Data-Domain IC (signal cleanup)")
                print(f"    Expected: Significant BER reduction")
            else:
                print(f"\n  ℹ️  NO IC (disabled or not applicable)")
            
            comm_result = self.comm_rx.receive_slot(
                y_waveform=y_waveform,
                tx_dict=tx_dict,
                H_true=H_comm if use_oracle_channel else None,
                use_oracle_channel=use_oracle_channel,
                tx_power_dbm=tx_power_dbm,
                noise_power_actual=noise_power_thermal,
                target_snr_db=comm_snr_db,
                chirp_freq=chirp_freq if self.enable_ic else None
            )
            
            results['comm_metrics'] = comm_result['metrics']
            results['comm_snr_db'] = comm_result['snr_est_db']
            results['comm_ber'] = comm_result.get('ber')
            results['comm_ser'] = comm_result.get('ser')
            results['comm_throughput_mbps'] = comm_result.get('throughput_mbps')
            
            if 'snr_antenna_db' in comm_result['metrics']:
                results['comm_snr_antenna_db'] = comm_result['metrics']['snr_antenna_db']
            if 'snr_antenna_linear' in comm_result['metrics']:
                results['comm_snr_antenna_linear'] = comm_result['metrics']['snr_antenna_linear']
            if 'comm_rx_power' in comm_result['metrics']:
                results['comm_rx_power'] = comm_result['metrics']['comm_rx_power']
            
            if 'ic_applied' in comm_result['metrics']:
                results['ic_applied'] = comm_result['metrics']['ic_applied']
                if comm_result['metrics']['ic_applied']:
                    results['ic_snr_gain_db'] = comm_result['metrics'].get('ic_snr_gain_db', 0)
                    results['residual_ratio'] = comm_result['metrics'].get('residual_ratio', 0)
                    results['cancellation_effectiveness'] = comm_result['metrics'].get('cancellation_effectiveness', 0)
            
            if 'data_domain_ic_applied' in comm_result['metrics']:
                results['data_domain_ic_applied'] = comm_result['metrics']['data_domain_ic_applied']
                if comm_result['metrics']['data_domain_ic_applied']:
                    results['data_ic_effectiveness'] = comm_result['metrics'].get('data_ic_effectiveness', 0)
                    results['data_ic_power_reduction_db'] = comm_result['metrics'].get('data_ic_power_reduction_db', 0)
            
            print(f"\n  {'='*76}")
            print(f"  COMM RESULTS:")
            print(f"  {'='*76}")
            
            if 'snr_antenna_db' in comm_result['metrics']:
                print(f"    SNR_antenna (pure thermal): {comm_result['metrics']['snr_antenna_db']:.2f} dB")
            
            print(f"    SINR_eff (post-EQ): {results['comm_snr_db']:.2f} dB")
            if results.get('ic_applied', False):
                print(f"    Pilot IC applied: YES ✅")
                if 'ic_snr_gain_db' in results:
                    print(f"    Pilot IC gain: {results['ic_snr_gain_db']:.2f} dB")
                if 'cancellation_effectiveness' in results:
                    print(f"    Pilot IC effectiveness: {results['cancellation_effectiveness']*100:.1f}%")
            else:
                print(f"    Pilot IC applied: NO")
            
            if results.get('data_domain_ic_applied', False):
                print(f"    Data IC applied: YES ✅")
                if 'data_ic_effectiveness' in results:
                    print(f"    Data IC effectiveness: {results['data_ic_effectiveness']*100:.1f}%")
                if 'data_ic_power_reduction_db' in results:
                    print(f"    Data IC power reduction: {results['data_ic_power_reduction_db']:.2f} dB")
            else:
                print(f"    Data IC applied: NO")
            
            if results['comm_ber']:
                print(f"    BER: {results['comm_ber']:.3e}")
            if results['comm_throughput_mbps']:
                print(f"    Throughput: {results['comm_throughput_mbps']:.2f} Mbps")
            print(f"  {'='*76}\n")
            
            if verbose:
                self.logger.info(f"  ✅ COMM complete")
                self.logger.info(f"     SNR: {results['comm_snr_db']:.2f} dB")
                if results.get('ic_applied', False):
                    self.logger.info(f"     Pilot IC gain: {results.get('ic_snr_gain_db', 0):.2f} dB")
                if results.get('data_domain_ic_applied', False):
                    self.logger.info(f"     Data IC effectiveness: {results.get('data_ic_effectiveness', 0)*100:.1f}%")
                if results['comm_ber']:
                    self.logger.info(f"     BER: {results['comm_ber']:.3e}")
        
        print(f"\n{'─'*80}")
        print(f"[STEP 4: RADAR PROCESSING - THEORETICALLY COMPLETE v8.4]")
        print(f"{'─'*80}")

        print(f"\n  THEORETICAL MODEL (Section 3.4 - Sensing Receiver):")
        print(f"  {'─'*76}")
        print(f"")
        print(f"  Received Signal:")
        print(f"    y_radar(t) = β·H_radar(t) ⊗ c(t) + α·H_radar(t) ⊗ x_ofdm(t) + n(t)")
        print(f"")
        print(f"  Where:")
        print(f"    - c(t): FMCW chirp signal (transmitted waveform)")
        print(f"    - x_ofdm(t): OFDM communication signal (interference in additive ISAC)")
        print(f"    - H_radar(t): Radar channel (target reflections + multipath)")
        print(f"    - β: Radar power allocation factor (β² = {self.radar_power_factor**2:.6f})")
        print(f"    - α: Comm power allocation factor (α² = {self.comm_power_factor**2:.6f})")
        print(f"    - n(t): Thermal noise (σ²_n = {self.rx_config.sens_noise_power_w:.6e})")
        print(f"")
        print(f"  Processing Pipeline (9 steps):")
        print(f"    1. De-chirping: Multiply by c*(t) → beat signal")
        print(f"    2. Range FFT: Compress fast-time → range dimension")
        print(f"    3. Doppler FFT: Compress slow-time → velocity dimension")
        print(f"    4. Static clutter removal: Zero DC bin (if enabled)")
        print(f"    5. Amplitude scaling: Normalize per-channel power")
        print(f"    6. Hybrid combining: Coherent TX + Non-coherent RX")
        print(f"    7. CFAR detection: Adaptive thresholding")
        print(f"    8. Peak extraction: Top N detections")
        print(f"    9. Parameter estimation: Range, velocity, angle (MUSIC)")
        print(f"  {'─'*76}")

        print(f"\n  IMPLEMENTATION STATUS:")
        print(f"    ✅ Raw received signal with OFDM interference (NEW!)")
        print(f"    ✅ β (radar power factor) scaling applied")
        print(f"    ✅ Virtual array extraction for MUSIC")
        print(f"    ✅ Vectorized de-chirping in SensingReceiver")
        print(f"    ✅ Adaptive SNR computation for dense scenes")
        print(f"    ✅ Multi-target ground truth support")
        
        print(f"\n  Generating synthetic raw received signal...")
        print(f"    β (radar power factor): {self.radar_power_factor:.6f}")
        print(f"    β² (power fraction): {self.radar_power_factor**2:.6f}")
        print(f"    Target SNR (after β scaling): {radar_snr_db} dB")

        # CRITICAL THEORETICAL ADDITION: Include OFDM interference in additive ISAC mode
        add_ofdm_interference = (self.fmcw_integration_mode == 'additive')

        if add_ofdm_interference:
            print(f"\n  THEORETICAL SIGNAL MODEL (Section 3.4):")
            print(f"    y_radar = β·H⊗c(t) + α·H⊗x_ofdm(t) + n(t)")
            print(f"              ↑ Radar echo  ↑ OFDM interference  ↑ Noise")
            print(f"    α (comm power): {self.comm_power_factor:.6f}")
            print(f"    Expected ISR (α²/β²): {(self.comm_power_factor**2 / self.radar_power_factor**2):.2f}")
            print(f"    ✅ OFDM interference WILL BE ADDED")
        else:
            print(f"    Mode: {self.fmcw_integration_mode}")
            print(f"    ℹ️  OFDM interference NOT applicable (non-additive mode)")

        raw_received_signal = self._generate_synthetic_raw_radar_signal(
            target_range_m=target_range_m,
            target_velocity_ms=target_velocity_ms,
            target_snr_db=radar_snr_db,
            beta=self.radar_power_factor,
            seed=seed,
            add_ofdm_interference=add_ofdm_interference  # ← NEW PARAMETER
        )
        
        print(f"\n  Raw received signal generated:")
        print(f"    Shape: {raw_received_signal.shape}")

        # Compute and validate signal power components
        total_power = np.mean(np.abs(raw_received_signal)**2)
        print(f"    Total power: {total_power:.6e}")

        if self.fmcw_integration_mode == 'additive':
            # Theoretical power breakdown
            beta_sq = self.radar_power_factor ** 2
            alpha_sq = self.comm_power_factor ** 2
            noise_power = self.rx_config.sens_noise_power_w
    
            # Expected power contributions
            snr_linear = 10 ** (radar_snr_db / 10)
            expected_radar_power = beta_sq * snr_linear * noise_power
            expected_ofdm_power = alpha_sq / beta_sq * expected_radar_power  # ISR
            expected_total = expected_radar_power + expected_ofdm_power + noise_power
    
            print(f"\n    THEORETICAL POWER VALIDATION:")
            print(f"      Expected radar echo: {expected_radar_power:.6e} (β²·SNR·σ²_n)")
            print(f"      Expected OFDM interference: {expected_ofdm_power:.6e} (α²/β²·P_radar)")
            print(f"      Expected noise: {noise_power:.6e}")
            print(f"      Expected total: {expected_total:.6e}")
            print(f"      Measured total: {total_power:.6e}")
    
            power_match = abs(total_power - expected_total) / expected_total < 0.3  # 30% tolerance
            print(f"      Match: {'✅' if power_match else '⚠️'} (tolerance: 30%)")

        print(f"\n    Status: ✅ PRE-DECHIRP (will be dechirped in SensingReceiver)")

        print(f"\n  Processing radar frame...")
        print(f"    Extract virtual array: {enable_angle_estimation}")
        if target_gts is not None:
            print(f"    Multi-target environment: {len(target_gts)} targets")
            print(f"    Evaluating against closest: {target_range_m:.2f}m")

        gt_for_eval = {'range_m': target_range_m, 'velocity_ms': target_velocity_ms}

        # ────────────────────────────────────────────────────────────────────
        # EXTRACT OFDM REFERENCE FROM ACTUAL TRANSMITTED ISAC WAVEFORM (v8.5)
        # ────────────────────────────────────────────────────────────────────
        
        tx_ofdm_td = None
        
        # Check if OFDM mitigation is enabled in sensing receiver
        if hasattr(self.sens_rx, 'enable_ofdm_mitigation') and \
           self.sens_rx.enable_ofdm_mitigation and \
           self.fmcw_integration_mode == 'additive':
            
            print(f"\n  OFDM MITIGATION ENABLED - Extracting reference...")
            
            # Generate synthetic H_radar for this example
            # In real data processing, H_radar would come from channel estimation
            # or from the dataset
            N_rx = self.rx_config.n_radar_rx_antennas
            N_tx = self.rx_config.n_radar_tx_antennas
            N_samples = self.rx_config.fmcw_n_samples_per_chirp
            N_chirps = self.rx_config.fmcw_n_chirps
            
            # Synthetic H_radar (simplified - unit channel)
            # In real processing, use actual channel estimate
            H_radar_synthetic = np.ones((N_rx, N_tx, N_samples, N_chirps), dtype=complex) / np.sqrt(N_rx * N_tx)
            
            # Extract OFDM reference from actual TX waveform
            tx_ofdm_td = self._extract_ofdm_from_transmitted_isac(H_radar_synthetic)
            
            if tx_ofdm_td is not None:
                print(f"  ✅ OFDM reference extracted successfully")
                print(f"     Will be passed to SensingReceiver for NLMS")
            else:
                print(f"  ⚠️  Could not extract OFDM reference")
                print(f"     NLMS mitigation will be disabled")
        else:
            print(f"\n  ℹ️  OFDM mitigation disabled or not applicable")
        
        # ────────────────────────────────────────────────────────────────────
        # CALL SENSING RECEIVER WITH OFDM REFERENCE
        # ────────────────────────────────────────────────────────────────────

        # Compute thermal noise floor for the radar path
        # This is the per-channel noise power added in _generate_synthetic_raw_radar_signal
        # Formula: noise_power = sens_noise_power_w (already computed there)
        # The thermal model in estimate_noise_power() uses: noise_combined = thermal × N_TX × N_RX
        _radar_thermal_noise = self.tx_config.thermal_noise_power_w

        print(f"\n  [RADAR FRAME CALL - SNR/CRB INPUTS]")
        print(f"    thermal_noise_power: {_radar_thermal_noise:.6e}")
        print(f"    input_snr_db:        {radar_snr_db:.2f} dB  ← exact design SNR for CRB")

        radar_result = self.sens_rx.process_radar_frame(
            received_signal=raw_received_signal,
            ground_truth=gt_for_eval,
            extract_virtual_array=enable_angle_estimation,
            tx_comm_signal=tx_ofdm_td,
            enable_ofdm_mitigation=True if tx_ofdm_td is not None else False,
            thermal_noise_power=_radar_thermal_noise,
            input_snr_db=float(radar_snr_db)   # ← exact design SNR passed for CRB
        )
        
        results['radar_snr_db'] = radar_result['snr_db']
        results['radar_n_detections'] = radar_result['n_detections']
        results['radar_n_targets_synthetic'] = len(target_gts) if target_gts is not None else 1

        print(f"  {'='*76}")
        print(f"  RADAR RESULTS:")
        print(f"  {'='*76}")

        if target_gts is not None:
            print(f"    Targets in scene: {len(target_gts)}")
            print(f"    Evaluation target: Closest ({target_range_m:.2f}m)")

        # THEORETICAL CONTEXT: Report OFDM interference impact
        if self.fmcw_integration_mode == 'additive':
            alpha_sq = self.comm_power_factor ** 2
            beta_sq = self.radar_power_factor ** 2
            isr_db = 10 * np.log10(alpha_sq / beta_sq)
    
            print(f"\n    ADDITIVE ISAC MODE:")
            print(f"      β² (radar fraction): {beta_sq*100:.1f}%")
            print(f"      α² (comm fraction): {alpha_sq*100:.1f}%")
            print(f"      ISR (α²/β²): {isr_db:+.2f} dB")
            print(f"      OFDM interference: {'INCLUDED' if add_ofdm_interference else 'EXCLUDED'}")
    
            if add_ofdm_interference:
                # Expected SNR degradation from OFDM interference
                # SINR = SNR / (1 + ISR)
                isr_linear = alpha_sq / beta_sq
                expected_sinr_db = radar_snr_db - 10*np.log10(1 + isr_linear)
                degradation_db = radar_snr_db - expected_sinr_db
        
                print(f"      Expected SNR degradation: {degradation_db:.2f} dB")
                print(f"      Expected SINR (w/ interference): {expected_sinr_db:.2f} dB")

        print(f"\n    MEASURED PERFORMANCE:")
        print(f"      SNR (RDM, post-processing): {results['radar_snr_db']:.2f} dB")
        print(f"      N detections: {results['radar_n_detections']}")

        if len(radar_result['targets']) > 0:
            t0 = radar_result['targets'][0]
            results['radar_top_range_m']     = t0['range_m']
            results['radar_top_velocity_ms'] = t0['velocity_ms']

            # ── SNR: read new names first, fall back to legacy keys ───────
            rdm_db   = t0.get('radar_snr_rdm_db',   t0.get('snr_db',       float('nan')))
            input_db = t0.get('radar_snr_input_db', t0.get('snr_input_db', float('nan')))
            src      = t0.get('radar_snr_input_source',
                               t0.get('snr_input_source', 'unknown'))

            # ── Store in results dict (CSV-ready names) ───────────────────
            results['radar_snr_rdm_db']      = rdm_db
            results['radar_snr_input_db']    = input_db
            results['radar_snr_input_source'] = src
            results['radar_crb_range_m']     = t0.get('crb_range_m',    float('nan'))
            results['radar_crb_velocity_ms'] = t0.get('crb_velocity_ms', float('nan'))

            print(f"    Top target range:    {results['radar_top_range_m']:.2f} m")
            print(f"    Top target velocity: {results['radar_top_velocity_ms']:+.2f} m/s")

            # ── SNR breakdown ─────────────────────────────────────────────
            print(f"\n    [SNR BREAKDOWN]")
            print(f"      radar_snr_rdm   (post-processing): {rdm_db:.2f} dB  "
                  f"← CFAR/ranking")
            print(f"      radar_snr_input (pre-processing):  {input_db:.2f} dB  "
                  f"← CRB reference (design target: {radar_snr_db:.1f} dB)")
            print(f"      SNR source: {src}")
            _n_tx_p  = self.rx_config.n_radar_tx_antennas
            _n_rx_p  = self.rx_config.n_radar_rx_antennas
            _mode_p  = self.rx_config.radar_mode
            _exp_g   = 44.0 if _mode_p == 'monostatic' else 41.0
            print(f"      Processing gain (rdm - input): {rdm_db - input_db:.1f} dB  "
                  f"(expected ≈ {_exp_g:.0f} dB for {_n_tx_p}×{_n_rx_p}, "
                  f"{_mode_p}, 200 MHz, β²=0.5)")
            if abs(input_db - radar_snr_db) < 3.0:
                print(f"      ✅ radar_snr_input matches design target")
            else:
                print(f"      ❌ radar_snr_input = {input_db:.1f} dB — mismatch!  "
                      f"Check input_snr_db is passed correctly")

            # ── CRB printout ──────────────────────────────────────────────
            crb_r = results['radar_crb_range_m']
            crb_v = results['radar_crb_velocity_ms']
            print(f"\n    [CRB LOWER BOUNDS — {t0.get('radar_mode','monostatic').upper()}, "
                  f"radar_snr_input referenced]")
            print(f"      CRB_range:    {crb_r:.6f} m")
            print(f"      CRB_velocity: {crb_v:.8f} m/s")
            if crb_r > 0.005:
                print(f"      ✅ CRB_range = {crb_r:.4f} m — physically meaningful")
            else:
                print(f"      ❌ CRB_range nearly zero — check input_snr_db flow")

            if 'angle_deg' in t0 and t0['angle_deg'] is not None:
                results['radar_top_angle_deg'] = t0['angle_deg']
                crb_angle = t0.get('crb_angle_deg', float('nan'))
                results['radar_crb_angle_deg'] = crb_angle
                print(f"    Top target angle: {results['radar_top_angle_deg']:.2f} deg  "
                      f"(CRB_angle: {crb_angle:.4f} deg) ✅ (MUSIC)")

        if 'range_error_m' in radar_result:
            results['radar_range_error_m']    = radar_result['range_error_m']
            results['radar_velocity_error_ms'] = radar_result['velocity_error_ms']

            print(f"\n    [ESTIMATION ACCURACY vs CRB]")
            print(f"      Range error (mean):    {results['radar_range_error_m']:.4f} m")
            print(f"      Velocity error (mean): {results['radar_velocity_error_ms']:.4f} m/s")

            crb_r = results.get('radar_crb_range_m', float('nan'))
            crb_v = results.get('radar_crb_velocity_ms', float('nan'))
            if not (crb_r != crb_r) and crb_r > 1e-9:  # not NaN and > floor
                eff_r = abs(results['radar_range_error_m'])    / (crb_r + 1e-12)
                eff_v = abs(results['radar_velocity_error_ms']) / (crb_v + 1e-12)
                results['radar_range_efficiency']    = float(eff_r)
                results['radar_velocity_efficiency'] = float(eff_v)
                status_r = ("✅ near-optimal" if eff_r < 5
                            else "⚠️  above CRB" if eff_r < 20
                            else "❌ far from CRB")
                print(f"      CRB_range:                       {crb_r:.6f} m  "
                      f"← radar_snr_input referenced")
                print(f"      CRB_velocity:                    {crb_v:.8f} m/s")
                print(f"      Range efficiency (|err|/CRB):    {eff_r:.1f}×  {status_r}")
                print(f"      Velocity efficiency (|err|/CRB): {eff_v:.1f}×")

        # ── SCNR ─────────────────────────────────────────────────────────
        if radar_result.get('scnr_db') is not None:
            results['radar_scnr_db'] = radar_result['scnr_db']
            print(f"\n    [SCNR — Signal vs Clutter+Noise]")
            print(f"      SCNR: {results['radar_scnr_db']:.2f} dB")

        results['radar_p_fa']             = radar_result.get('false_alarm_probability')
        results['radar_detection_matched'] = radar_result.get('detection_matched')

        print(f"\n    Detection matched: "
              f"{'✅ YES' if results['radar_detection_matched'] else '❌ NO'}")
        if results['radar_p_fa'] is not None:
            print(f"    P_fa: {results['radar_p_fa']:.6f}")

        # ── Final logger summary ──────────────────────────────────────────
        self.logger.info(f"  RADAR radar_snr_rdm:   {results.get('radar_snr_rdm_db', results.get('radar_snr_db', float('nan'))):.2f} dB")
        self.logger.info(f"  RADAR radar_snr_input: {results.get('radar_snr_input_db', float('nan')):.2f} dB")
        self.logger.info(f"  RADAR CRB_range:       {results.get('radar_crb_range_m', float('nan')):.6f} m")
        self.logger.info(f"  RADAR CRB_velocity:    {results.get('radar_crb_velocity_ms', float('nan')):.8f} m/s")
        self.logger.info(f"  RADAR SNR source:      {results.get('radar_snr_input_source', 'unknown')}")
        if results.get('radar_scnr_db') is not None:
            self.logger.info(f"  RADAR SCNR:            {results['radar_scnr_db']:.2f} dB")
        if results.get('radar_range_efficiency') is not None:
            self.logger.info(f"  Range efficiency:      {results['radar_range_efficiency']:.1f}× CRB")

        print(f"  {'='*76}\n")
        
        if verbose:
            self.logger.info(f"  ✅ RADAR complete (v8.4 - Corrected)")
            self.logger.info(f"     Detected: {'YES' if results['radar_detection_matched'] else 'NO'}")
            if results.get('radar_range_error_m') is not None:
                self.logger.info(f"     Range error: {results['radar_range_error_m']:.3f} m")
        
        results['success'] = True
        
        print(f"\n{'='*80}")
        print(f"[SAMPLE PROCESSING COMPLETE v8.4]")
        print(f"{'='*80}")
        
        if results.get('ic_applied', False) or results.get('data_domain_ic_applied', False):
            print(f"\n  ✅ INTERFERENCE CANCELLATION WAS APPLIED")
            if results.get('ic_applied', False) and 'ic_snr_gain_db' in results:
                print(f"    Pilot IC gain: {results['ic_snr_gain_db']:.2f} dB")
            if results.get('data_domain_ic_applied', False) and 'data_ic_effectiveness' in results:
                print(f"    Data IC effectiveness: {results['data_ic_effectiveness']*100:.1f}%")
        
        print(f"\n  FINAL METRICS:")
        if self.comm_rx:
            print(f"    COMM SNR_antenna: {results.get('comm_snr_antenna_db', float('nan')):.2f} dB")
            print(f"    COMM SINR_eff:    {results['comm_snr_db']:.2f} dB")
            if results['comm_ber']:
                print(f"    COMM BER:         {results['comm_ber']:.3e}")

        # ── radar_snr_rdm: read new key, fall back to legacy ──────────────
        _rdm_db   = results.get('radar_snr_rdm_db',   results.get('radar_snr_db',    float('nan')))
        _input_db = results.get('radar_snr_input_db', float('nan'))
        _crb_r    = results.get('radar_crb_range_m',  float('nan'))
        _crb_v    = results.get('radar_crb_velocity_ms', float('nan'))
        print(f"    RADAR radar_snr_rdm:   {_rdm_db:.2f} dB  ← post-processing (CFAR/ranking)")
        print(f"    RADAR radar_snr_input: {_input_db:.2f} dB  ← pre-processing (CRB ref, target={radar_snr_db:.0f} dB)")
        print(f"    RADAR CRB_range:       {_crb_r:.6f} m")
        print(f"    RADAR CRB_velocity:    {_crb_v:.8f} m/s")
        if results.get('radar_scnr_db') is not None:
            print(f"    RADAR SCNR:            {results['radar_scnr_db']:.2f} dB")
        print(f"    RADAR detected:        {'YES ✅' if results['radar_detection_matched'] else 'NO ❌'}")
        if results.get('radar_range_error_m') is not None:
            print(f"    RADAR range err:       {abs(results['radar_range_error_m']):.4f} m")
        if results.get('radar_range_efficiency') is not None:
            eff_r = results['radar_range_efficiency']
            status = ("✅" if eff_r < 5 else "⚠️ " if eff_r < 20 else "❌")
            print(f"    Range efficiency:      {eff_r:.1f}× CRB  {status}")

        print(f"{'='*80}\n")
        
        return results
    
    def _generate_synthetic_raw_radar_signal(
        self,
        target_range_m: float,
        target_velocity_ms: float,
        target_snr_db: float,
        beta: float,
        seed: Optional[int] = None,
        add_ofdm_interference: bool = True
    ) -> np.ndarray:
    
        if seed is not None:
            np.random.seed(seed)
    
        n_rx = self.rx_config.n_radar_rx_antennas
        n_tx = self.rx_config.n_radar_tx_antennas
        n_samples = self.rx_config.fmcw_n_samples_per_chirp
        n_chirps = self.rx_config.fmcw_n_chirps
    
        # Time grids
        t_fast = np.arange(n_samples) / self.rx_config.fmcw_sampling_rate_hz
        t_slow = np.arange(n_chirps) * self.rx_config.fmcw_t_chirp_s
    
        # ========== TERM 1: β·H_radar ⊗ c(t) - RADAR ECHO ==========
    
        # Compute delay and Doppler — mode-aware factors
        # monostatic: delay_factor=2 (round-trip), doppler_factor=2 (two-way)
        # bistatic:   delay_factor=1 (one-way),    doppler_factor=1 (one-way)
        _delay_f   = float(getattr(self.rx_config, 'delay_factor',
                                   2.0 if self.rx_config.radar_mode == 'monostatic' else 1.0))
        _doppler_f = float(getattr(self.rx_config, 'doppler_factor',
                                   2.0 if self.rx_config.radar_mode == 'monostatic' else 1.0))
        tau = (_delay_f * target_range_m) / LIGHTSPEED
        f_d = (_doppler_f * target_velocity_ms * self.rx_config.carrier_freq_hz) / LIGHTSPEED
    
        chirp_slope = self.rx_config.fmcw_bandwidth_hz / self.rx_config.fmcw_t_chirp_s
    
        # Power scaling for radar echo
        # Theory: SNR = (β²·|echo|²) / σ²_n
        # Therefore: |echo|² = (SNR × σ²_n) / β²
        snr_linear = 10 ** (target_snr_db / 10)
        noise_power = self.rx_config.sens_noise_power_w
        n_channels = n_rx * n_tx
    
        beta_sq = beta ** 2
        amplitude_radar = np.sqrt(beta_sq * snr_linear * noise_power / n_channels)
    
        # Generate radar echo signal
        received_signal = np.zeros((n_rx, n_tx, n_samples, n_chirps), dtype=complex)
    
        for m in range(n_chirps):
            # Doppler phase shift (varies with chirp index / slow time)
            phase_doppler = 2 * np.pi * f_d * t_slow[m]
        
            # Delayed chirp (target range causes time delay)
            t_delayed = t_fast - tau
            t_delayed = np.maximum(t_delayed, 0)  # Causality
        
            # Chirp phase (FMCW quadratic phase)
            phase_chirp = np.pi * chirp_slope * t_delayed**2
        
            # Complete radar echo for this chirp
            target_echo = amplitude_radar * np.exp(1j * (phase_chirp + phase_doppler))
        
            # Replicate across all TX/RX pairs (simplified channel model)
            for rx_idx in range(n_rx):
                for tx_idx in range(n_tx):
                    received_signal[rx_idx, tx_idx, :, m] = target_echo
    
            # ========== TERM 2: α·H_radar ⊗ x_ofdm(t) - OFDM INTERFERENCE ==========
    
            if add_ofdm_interference and self.fmcw_integration_mode == 'additive':
                # OFDM interference is the communication waveform convolved with radar channel
                # Simplified model: Random OFDM-like symbols modulated onto subcarriers
        
                alpha_sq = self.comm_power_factor ** 2
        
                # Power of OFDM interference relative to radar echo
                # Theory: OFDM interference power ≈ α²·|H|²·|x_ofdm|²
                # We set |x_ofdm|² ≈ 1 (normalized), so interference power ∝ α²
        
            # Scale OFDM interference to have appropriate power relative to radar echo
            # Interference-to-Signal Ratio (ISR) = α² / β²
            isr = alpha_sq / beta_sq
            amplitude_ofdm = amplitude_radar * np.sqrt(isr)
        
            # Generate pseudo-OFDM interference
            # Model as wideband structured noise (approximates OFDM after channel)
            # In real system, this would be actual OFDM symbols convolved with H_radar
        
            # Use different random seed for interference
            if seed is not None:
                np.random.seed(seed + 1000)
        
            # Generate OFDM-like interference with time-frequency structure
            # Simplification: Use filtered complex Gaussian noise to mimic OFDM spectrum
            ofdm_interference = np.zeros_like(received_signal)
        
            for rx_idx in range(n_rx):
                for tx_idx in range(n_tx):
                    # Generate wideband interference for each antenna pair
                    interference_raw = (np.random.randn(n_samples, n_chirps) + 
                                        1j * np.random.randn(n_samples, n_chirps)) / np.sqrt(2)
                
                    # Apply spectral shaping to mimic OFDM (bandlimited to FMCW bandwidth)
                    # Simple approach: FFT → mask → IFFT
                    interference_fft = np.fft.fft(interference_raw, axis=0)
                
                    # Keep only components within FMCW bandwidth
                    # (This mimics OFDM subcarrier structure)
                    bandwidth_fraction = 0.8  # OFDM uses ~80% of available bandwidth
                    n_keep = int(n_samples * bandwidth_fraction / 2)
                    mask = np.zeros(n_samples, dtype=bool)
                    mask[:n_keep] = True
                    mask[-n_keep:] = True
                
                    interference_fft[~mask, :] = 0
                    interference_shaped = np.fft.ifft(interference_fft, axis=0)
                
                    # Scale to correct power level
                    current_power = np.mean(np.abs(interference_shaped)**2)
                    interference_scaled = interference_shaped * (amplitude_ofdm / np.sqrt(current_power))
                
                    ofdm_interference[rx_idx, tx_idx, :, :] = interference_scaled
        
            # Add OFDM interference to received signal
            received_signal += ofdm_interference
        
            if seed is not None:
                np.random.seed(seed)  # Reset seed for noise generation
    
        # ========== TERM 3: n(t) - THERMAL NOISE ==========
    
        noise_std = np.sqrt(noise_power / 2)
        noise = noise_std * (np.random.randn(*received_signal.shape) + 
                            1j * np.random.randn(*received_signal.shape))
    
        received_signal += noise
    
        return received_signal
    
    def _extract_ofdm_from_transmitted_isac(self, H_radar: np.ndarray) -> Optional[np.ndarray]:
        
        # ═══════════════════════════════════════════════════════════════════════
        # VALIDATION: Check if TX saved the pre-normalization ISAC waveform
        # ═══════════════════════════════════════════════════════════════════════
        
        if not hasattr(self.tx, 'last_transmitted_isac') or self.tx.last_transmitted_isac is None:
            print(f"  ⚠️  No pre-normalization ISAC waveform saved in transmitter")
            print(f"      Required: ISACTransmitter v5.1+ with last_transmitted_isac")
            return None
        
        print(f"\n{'='*80}")
        print(f"[EXTRACT OFDM - v8.8 EXACT CHIRP SUBTRACTION]")
        print(f"{'='*80}")
        print(f"  Method: Direct subtraction from ISAC waveform")
        print(f"  Formula: x_ofdm = (x_isac - β·chirp) / α")
        print(f"  Advantage: No tiling, no normalization mismatch!")
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: Get transmitted ISAC waveform (BEFORE final normalization)
        # ═══════════════════════════════════════════════════════════════════════
        
        print(f"\n[STEP 1: Get ISAC waveform and power factors]")
        
        # Get pre-normalization ISAC waveform
        # Shape: (N_tx, 43008) - this is α·x_ofdm + β·chirp BEFORE normalization
        x_isac = self.tx.last_transmitted_isac
        
        # Get power allocation factors from TX
        alpha = getattr(self.tx, 'comm_power_factor', np.sqrt(0.5))
        beta = getattr(self.tx, 'radar_power_factor', np.sqrt(0.5))
        
        N_tx = x_isac.shape[0]
        N_isac_samples = x_isac.shape[1]
        
        p_isac = np.mean(np.abs(x_isac)**2)
        
        print(f"  ISAC waveform (pre-normalization):")
        print(f"    Shape: {x_isac.shape}")
        print(f"    Power: {p_isac:.6e}")
        print(f"    α (comm factor): {alpha:.6f}")
        print(f"    β (radar factor): {beta:.6f}")
        print(f"    α² + β² = {alpha**2 + beta**2:.6f} (should be 1.0)")
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: Generate EXACT chirp (same as transmitter used)
        # ═══════════════════════════════════════════════════════════════════════
        
        print(f"\n[STEP 2: Generate exact chirp for subtraction]")
        
        # CRITICAL: Generate chirp at SAME length as ISAC waveform (43008)
        chirp_1d = self._generate_fmcw_chirp_for_extraction(n_samples=N_isac_samples)
        
        # Verify chirp properties
        chirp_power = np.mean(np.abs(chirp_1d)**2)
        print(f"  Generated chirp:")
        print(f"    Length: {len(chirp_1d)} samples")
        print(f"    Power: {chirp_power:.6f} (should be 1.0)")
        print(f"    Status: {'✅ OK' if abs(chirp_power - 1.0) < 1e-5 else '⚠️  CHECK'}")
        
        # Replicate chirp to all TX antennas (same chirp on all antennas)
        chirp_multi = np.tile(chirp_1d, (N_tx, 1))  # (N_tx, 43008)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: Extract pure OFDM component by subtraction
        # ═══════════════════════════════════════════════════════════════════════
        
        print(f"\n[STEP 3: Extract OFDM via direct subtraction]")
        print(f"  Formula: x_ofdm = (x_isac - β·chirp) / α")
        
        # Direct subtraction to extract OFDM component
        # x_isac = α·x_ofdm + β·chirp
        # x_ofdm = (x_isac - β·chirp) / α
        x_ofdm_extracted = (x_isac - beta * chirp_multi) / alpha
        
        # Verify extraction
        p_ofdm_extracted = np.mean(np.abs(x_ofdm_extracted)**2)
        
        print(f"  Extracted OFDM:")
        print(f"    Shape: {x_ofdm_extracted.shape}")
        print(f"    Power: {p_ofdm_extracted:.6e}")
        
        # Theoretical validation
        # Expected OFDM power ≈ P_isac × (α²/(α²+β²)) since x_ofdm and chirp are uncorrelated
        expected_ofdm_power = p_isac * (alpha**2 / (alpha**2 + beta**2))
        power_match_pct = abs(p_ofdm_extracted - expected_ofdm_power) / expected_ofdm_power * 100
        
        print(f"  Theoretical validation:")
        print(f"    Expected OFDM power: {expected_ofdm_power:.6e}")
        print(f"    Measured OFDM power: {p_ofdm_extracted:.6e}")
        print(f"    Deviation: {power_match_pct:.2f}%")
        print(f"    Status: {'✅ GOOD' if power_match_pct < 10 else '⚠️  CHECK'}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: Reshape to radar frame dimensions
        # ═══════════════════════════════════════════════════════════════════════
        
        print(f"\n[STEP 4: Reshape to radar frame]")
        
        N_radar_samples = self.tx_config.fmcw_n_samples_per_chirp
        N_radar_chirps = self.tx_config.fmcw_n_chirps
        radar_total = N_radar_samples * N_radar_chirps
        
        print(f"  OFDM samples: {N_isac_samples}")
        print(f"  Radar frame: {N_radar_samples} × {N_radar_chirps} = {radar_total}")
        
        # Tile or truncate to match radar frame length
        x_ofdm_matched = np.zeros((N_tx, radar_total), dtype=complex)
        
        for tx_idx in range(N_tx):
            if N_isac_samples < radar_total:
                # Tile and truncate
                n_repeats = int(np.ceil(radar_total / N_isac_samples))
                ofdm_tiled = np.tile(x_ofdm_extracted[tx_idx], n_repeats)
                x_ofdm_matched[tx_idx] = ofdm_tiled[:radar_total]
                if tx_idx == 0:
                    print(f"  Action: Tiled {n_repeats}× and truncated")
                    print(f"  ⚠️  Note: Tiling still creates discontinuities, but power is correct!")
            else:
                # Truncate only
                x_ofdm_matched[tx_idx] = x_ofdm_extracted[tx_idx, :radar_total]
                if tx_idx == 0:
                    print(f"  Action: Truncated to radar frame")
        
        # Reshape to 3D: (N_tx, N_samples, N_chirps)
        x_ofdm_3d = x_ofdm_matched.reshape(N_tx, N_radar_samples, N_radar_chirps)
        
        p_ofdm_reshaped = np.mean(np.abs(x_ofdm_3d)**2)
        print(f"  Reshaped OFDM:")
        print(f"    Shape: {x_ofdm_3d.shape}")
        print(f"    Power: {p_ofdm_reshaped:.6e}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 5: Apply H_radar channel
        # ═══════════════════════════════════════════════════════════════════════
        
        print(f"\n[STEP 5: Apply H_radar channel]")
        
        N_rx, N_tx_check, _, _ = H_radar.shape
        
        if N_tx_check != N_tx:
            print(f"  ⚠️  H_radar TX dimension mismatch: {N_tx_check} vs {N_tx}")
        
        rx_ofdm_interference = np.zeros((N_rx, N_tx, N_radar_samples, N_radar_chirps), dtype=complex)
        
        # Apply channel: y = H * x
        for rx_idx in range(N_rx):
            for tx_idx in range(min(N_tx, N_tx_check)):
                rx_ofdm_interference[rx_idx, tx_idx] = H_radar[rx_idx, tx_idx] * x_ofdm_3d[tx_idx]
        
        p_rx_ofdm = np.mean(np.abs(rx_ofdm_interference)**2)
        
        print(f"  RX OFDM interference:")
        print(f"    Shape: {rx_ofdm_interference.shape}")
        print(f"    Power: {p_rx_ofdm:.6e}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 6: Average over RX-TX for 2D NLMS reference
        # ═══════════════════════════════════════════════════════════════════════
        
        print(f"\n[STEP 6: Create 2D reference for NLMS]")
        
        # Average over spatial dimensions (RX × TX)
        tx_ofdm_td = np.mean(rx_ofdm_interference, axis=(0, 1))
        
        p_final = np.mean(np.abs(tx_ofdm_td)**2)
        
        print(f"  Final NLMS reference:")
        print(f"    Shape: {tx_ofdm_td.shape}")
        print(f"    Power: {p_final:.6e}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # FINAL STATUS
        # ═══════════════════════════════════════════════════════════════════════
        
        print(f"\n{'='*80}")
        print(f"[OFDM REFERENCE READY - v8.8 EXACT SUBTRACTION]")
        print(f"{'='*80}")
        print(f"  ✅ Method: Direct chirp subtraction")
        print(f"  ✅ Source: Pre-normalization ISAC waveform")
        print(f"  ✅ No power approximation (exact formula)")
        print(f"  ⚠️  Tiling discontinuities still present (but power is correct)")
        print(f"")
        print(f"  EXPECTED IMPROVEMENT:")
        print(f"    Correlation: 2-4% → 60-90% (MAJOR BOOST!)")
        print(f"    NLMS gain: 0 dB → 10-15 dB")
        print(f"    Detection rate: +10-20 percentage points")
        print(f"{'='*80}\n")
        
        return tx_ofdm_td
        
    def _generate_fmcw_chirp_for_extraction(self, n_samples: Optional[int] = None) -> np.ndarray:

        config = self.tx_config
        
        # Use OFDM slot length if not specified
        # CRITICAL: Chirp is generated at OFDM length (43008), NOT radar frame length!
        if n_samples is None:
            # Compute OFDM slot length
            n_symbols = config.n_symbols_per_slot
            n_fft = config.n_fft
            n_cp = config.cp_length_samples
            n_samples = n_symbols * (n_fft + n_cp)  # 14 * (2048 + 1024) = 43008
        
        # Get chirp parameters from config
        fs = config.fmcw_sampling_rate_hz
        slope = config.fmcw_chirp_slope
        
        print(f"\n[CHIRP EXTRACTION - v8.8]")
        print(f"  Generating exact chirp matching TX...")
        print(f"  N_samples: {n_samples}")
        print(f"  Sampling rate: {fs/1e6:.2f} MHz")
        print(f"  Chirp slope: {slope/1e12:.1f} THz/s")
        
        # EXACT REPLICA of FMCWChirpGenerator.generate_chirp()
        # Time vector
        dt = 1.0 / fs
        t = np.arange(n_samples, dtype=np.float32) * dt
        
        # Instantaneous phase: φ(t) = π * slope * t²
        phi = np.pi * slope * t**2
        
        # Complex chirp (unit amplitude)
        chirp = np.exp(1j * phi).astype(np.complex64)
        
        # Verify unit power (should be exactly 1.0)
        chirp_power = np.mean(np.abs(chirp)**2)
        
        print(f"  Chirp power (before normalization): {chirp_power:.6f}")
        
        # Ensure exact unit power (match transmitter)
        if abs(chirp_power - 1.0) > 1e-6:
            print(f"  ⚠️  Normalizing chirp to unit power...")
            chirp = chirp / np.sqrt(chirp_power)
            chirp_power = np.mean(np.abs(chirp)**2)
            print(f"  Chirp power (after normalization): {chirp_power:.6f}")
        else:
            print(f"  ✅ Chirp already has unit power")
        
        return chirp

    def _generate_fmcw_chirp_freq_domain(self) -> np.ndarray:
        n_samples = self.tx_config.fmcw_n_samples_per_chirp
        bandwidth_hz = self.tx_config.fmcw_bandwidth_hz
        t_chirp = self.tx_config.fmcw_t_chirp_s
        
        t = np.linspace(0, t_chirp, n_samples, endpoint=False)
        mu = bandwidth_hz / t_chirp
        
        chirp_time = np.exp(1j * np.pi * mu * t**2)
        
        chirp_fft = np.fft.fft(chirp_time) / np.sqrt(n_samples)
        
        n_sc = self.tx_config.n_subcarriers_actual
        n_left = n_sc // 2
        n_right = n_sc - n_left
        sc_indices = np.concatenate([np.arange(1, n_right + 1), np.arange(-n_left, 0)])
        
        chirp_freq = chirp_fft[sc_indices]
        
        chirp_power = np.mean(np.abs(chirp_freq)**2)
        chirp_freq = chirp_freq / np.sqrt(chirp_power)
        
        return chirp_freq
    
    def test_multi_snr(self, snr_values=None, n_trials=5, target_range_m=150.0,
                      target_velocity_ms=30.0, use_oracle=True, tx_power_dbm=43.0,
                      enable_angle_estimation=False):
        if snr_values is None:
            snr_values = [0, 5, 10, 15, 20, 25, 30]
        
        self.logger.info(f"\n[MULTI-SNR TEST v8.4]")
        self.logger.info(f"  SNR: {snr_values} dB")
        self.logger.info(f"  Trials: {n_trials}")
        self.logger.info(f"  IC: {'ON' if self.enable_ic else 'OFF'}")
        self.logger.info(f"  Angle estimation: {'ON' if enable_angle_estimation else 'OFF'}")
        self.logger.info(f"  Mode: Single-target synthetic (use real data for multi-target)")
        
        results = {
            'snr_values': snr_values,
            'n_trials': n_trials,
            'ic_enabled': self.enable_ic,
            'ic_iterations': self.ic_iterations if self.enable_ic else 0,
            'ber_mean': [], 'ber_std': [],
            'snr_eff_mean': [], 'snr_eff_std': [],
            'sinr_eff_mean': [], 'sinr_eff_std': [],
            'detection_rate': [],
            'ic_gain_mean': [], 'ic_gain_std': [],
            'data_ic_eff_mean': [], 'data_ic_eff_std': [],
            'radar_range_error_mean': [], 'radar_range_error_std': [],
        }
        
        for snr_db in snr_values:
            self.logger.info(f"\n  [SNR={snr_db}dB]")
            
            ber_vals, snr_eff_vals, ic_gain_vals, data_ic_eff_vals = [], [], [], []
            range_errors = []
            detection_count = 0
            
            for trial in range(n_trials):
                res = self.process_sample_synthetic(
                    comm_snr_db=snr_db, radar_snr_db=snr_db,
                    target_range_m=target_range_m,
                    target_velocity_ms=target_velocity_ms,
                    seed=42 + trial, use_oracle_channel=use_oracle,
                    tx_power_dbm=tx_power_dbm, verbose_override=False,
                    enable_angle_estimation=enable_angle_estimation
                )
                
                if res.get('comm_ber'):
                    ber_vals.append(res['comm_ber'])
                
                if 'comm_snr_db' in res:
                    sinr_eff = res['comm_snr_db']
                    if sinr_eff and sinr_eff != -np.inf:
                        snr_eff_vals.append(sinr_eff)
                
                if res.get('ic_applied', False) and 'ic_snr_gain_db' in res:
                    ic_gain_vals.append(res['ic_snr_gain_db'])
                
                if res.get('data_domain_ic_applied', False) and 'data_ic_effectiveness' in res:
                    data_ic_eff_vals.append(res['data_ic_effectiveness'])
                
                if res.get('radar_detection_matched'):
                    detection_count += 1
                
                if res.get('radar_range_error_m') is not None:
                    range_errors.append(abs(res['radar_range_error_m']))
            
            def safe_stats(vals):
                return (np.mean(vals), np.std(vals)) if vals else (np.nan, np.nan)
            
            results['ber_mean'].append(safe_stats(ber_vals)[0])
            results['ber_std'].append(safe_stats(ber_vals)[1])
            sinr_mean, sinr_std = safe_stats(snr_eff_vals)
            results['snr_eff_mean'].append(sinr_mean)
            results['snr_eff_std'].append(sinr_std)
            results['sinr_eff_mean'].append(sinr_mean)
            results['sinr_eff_std'].append(sinr_std)
            results['detection_rate'].append(detection_count / n_trials)
            results['ic_gain_mean'].append(safe_stats(ic_gain_vals)[0])
            results['ic_gain_std'].append(safe_stats(ic_gain_vals)[1])
            results['data_ic_eff_mean'].append(safe_stats(data_ic_eff_vals)[0])
            results['data_ic_eff_std'].append(safe_stats(data_ic_eff_vals)[1])
            results['radar_range_error_mean'].append(safe_stats(range_errors)[0])
            results['radar_range_error_std'].append(safe_stats(range_errors)[1])
            
            if ber_vals:
                self.logger.info(f"    BER: {np.mean(ber_vals):.3e}")
            if snr_eff_vals:
                self.logger.info(f"    SINR_eff: {np.mean(snr_eff_vals):.2f} dB")
            if ic_gain_vals:
                self.logger.info(f"    Pilot IC gain: {np.mean(ic_gain_vals):.2f} dB")
            if data_ic_eff_vals:
                self.logger.info(f"    Data IC eff: {np.mean(data_ic_eff_vals)*100:.1f}%")
            self.logger.info(f"    P_d: {detection_count/n_trials*100:.0f}%")
            if range_errors:
                self.logger.info(f"    Range error: {np.mean(range_errors):.3f} m")
        
        self.logger.info(f"\n[SUMMARY v8.4]")
        if self.enable_ic:
            self.logger.info(f"{'SNR':<6} {'BER':<12} {'SINR_eff':<11} {'Pilot IC':<10} {'Data IC':<10} {'P_d':<8} {'R_err':<8}")
            self.logger.info(f"{'-'*69}")
        else:
            self.logger.info(f"{'SNR':<6} {'BER':<12} {'SINR_eff':<11} {'P_d':<8} {'R_err':<8}")
            self.logger.info(f"{'-'*49}")
        
        for i, snr in enumerate(snr_values):
            ber_str = f"{results['ber_mean'][i]:.2e}" if not np.isnan(results['ber_mean'][i]) else "N/A"
            sinr_eff_str = f"{results['sinr_eff_mean'][i]:.1f}dB" if not np.isnan(results['sinr_eff_mean'][i]) else "N/A"
            pd_str = f"{results['detection_rate'][i]*100:.0f}%"
            r_err_str = f"{results['radar_range_error_mean'][i]:.2f}m" if not np.isnan(results['radar_range_error_mean'][i]) else "N/A"
            
            if self.enable_ic:
                ic_str = f"{results['ic_gain_mean'][i]:.1f}dB" if not np.isnan(results['ic_gain_mean'][i]) else "N/A"
                data_ic_str = f"{results['data_ic_eff_mean'][i]*100:.0f}%" if not np.isnan(results['data_ic_eff_mean'][i]) else "N/A"
                self.logger.info(f"{snr:<6} {ber_str:<12} {sinr_eff_str:<11} {ic_str:<10} {data_ic_str:<10} {pd_str:<8} {r_err_str:<8}")
            else:
                self.logger.info(f"{snr:<6} {ber_str:<12} {sinr_eff_str:<11} {pd_str:<8} {r_err_str:<8}")
        
        return results
    
    def run_comprehensive_validation(self, enable_angle_estimation=False):
        self.logger.info(f"\n[VALIDATION SUITE v8.4]")
        self.logger.info(f"  IC: {'ON' if self.enable_ic else 'OFF'}")
        self.logger.info(f"  Angle estimation: {'ON' if enable_angle_estimation else 'OFF'}")
        self.logger.info(f"  Note: Synthetic tests use single target; real HDF5 has multi-target")
        
        report = {'tests': [], 'summary': {}}
        
        scenarios = [
            {'name': 'Equal Power (50/50)', 'alpha': np.sqrt(0.5), 'beta': np.sqrt(0.5)},
            {'name': 'COMM-Dominant (90/10)', 'alpha': 0.9487, 'beta': 0.3162},
            {'name': 'RADAR-Dominant (10/90)', 'alpha': 0.3162, 'beta': 0.9487}
        ]
        
        for scenario in scenarios:
            self.logger.info(f"\n  [{scenario['name']}]")
            self.set_power_allocation(scenario['alpha'], scenario['beta'])
            
            result = self.process_sample_synthetic(
                comm_snr_db=20.0, radar_snr_db=25.0,
                target_range_m=150.0, target_velocity_ms=30.0,
                verbose_override=True,
                enable_angle_estimation=enable_angle_estimation
            )
            
            report['tests'].append({
                'name': scenario['name'],
                'alpha': scenario['alpha'],
                'beta': scenario['beta'],
                'result': result
            })
        
        n_passed = 0
        for test in report['tests']:
            detected = test['result'].get('radar_detection_matched', False)
            range_err = abs(test['result'].get('radar_range_error_m', 999))
            vel_err = abs(test['result'].get('radar_velocity_error_ms', 999))
            
            passed = detected and range_err < 2.0 and vel_err < 2.0
            if passed:
                n_passed += 1
            
            status = "✅ PASS" if passed else "❌ FAIL"
            msg = f"  {test['name']}: {status}"
            if test['result'].get('ic_applied', False):
                ic_gain = test['result'].get('ic_snr_gain_db', 0)
                msg += f" (Pilot IC: {ic_gain:.1f}dB"
                if test['result'].get('data_domain_ic_applied', False):
                    data_eff = test['result'].get('data_ic_effectiveness', 0)
                    msg += f", Data IC: {data_eff*100:.0f}%"
                msg += ")"
            self.logger.info(msg)
        
        report['summary'] = {
            'all_tests_passed': n_passed == len(report['tests']),
            'n_tests': len(report['tests']),
            'n_passed': n_passed,
        }
        
        self.logger.info(f"\n[VALIDATION SUMMARY v8.4]")
        self.logger.info(f"  Passed: {n_passed}/{len(report['tests'])}")
        
        if report['summary']['all_tests_passed']:
            self.logger.info(f"  ✅ ALL TESTS PASSED")
        else:
            self.logger.info(f"  ⚠️  SOME TESTS FAILED")
        
        return report
    
    def _generate_synthetic_comm_channel(self):
        n_rx = self.rx_config.n_rx_antennas
        n_tx = self.tx_config.n_tx_antennas
        n_sc = self.tx_config.n_subcarriers_actual
        
        H_comm = (np.random.randn(n_rx, n_tx, n_sc) + 
                  1j * np.random.randn(n_rx, n_tx, n_sc)) / np.sqrt(2)
        
        return H_comm
    
    def _generate_ofdm_waveform_from_freq(self, freq_grid, H_comm, comm_snr_db,
                                         noise_power_override=None, verbose=False):
        n_rx = H_comm.shape[0]
        n_symbols = freq_grid.shape[0]
        n_fft = self.tx_config.n_fft
        n_cp = self.tx_config.cp_length_samples
        n_sc = self.tx_config.n_subcarriers_actual
        
        samples_per_symbol = n_fft + n_cp
        total_samples = samples_per_symbol * n_symbols
        
        y_waveform = np.zeros((n_rx, total_samples), dtype=complex)
        
        n_left = n_sc // 2
        n_right = n_sc - n_left
        sc_indices = np.concatenate([np.arange(1, n_right + 1), np.arange(-n_left, 0)])
        
        for sym_idx in range(n_symbols):
            tx_freq_sym = freq_grid[sym_idx, :]
            
            rx_freq_sym = np.zeros((n_rx, n_sc), dtype=complex)
            for sc_idx in range(n_sc):
                H_sc = H_comm[:, 0, sc_idx]
                X_sc = tx_freq_sym[sc_idx]
                rx_freq_sym[:, sc_idx] = H_sc * X_sc
            
            rx_freq_full = np.zeros((n_rx, n_fft), dtype=complex)
            rx_freq_full[:, sc_indices] = rx_freq_sym
            
            rx_time_sym = np.fft.ifft(rx_freq_full, axis=1) * np.sqrt(n_fft)
            
            cp = rx_time_sym[:, -n_cp:]
            rx_time_sym_with_cp = np.concatenate([cp, rx_time_sym], axis=1)
            
            start_idx = sym_idx * samples_per_symbol
            y_waveform[:, start_idx:start_idx + samples_per_symbol] = rx_time_sym_with_cp
        
        signal_power = np.mean(np.abs(y_waveform) ** 2)
        
        if noise_power_override is not None:
            noise_power = noise_power_override
        else:
            snr_linear = 10 ** (comm_snr_db / 10)
            noise_power = signal_power / snr_linear
        
        noise_std = np.sqrt(noise_power / 2)
        noise = noise_std * (np.random.randn(*y_waveform.shape) + 
                            1j * np.random.randn(*y_waveform.shape))
        
        y_waveform += noise
        
        return y_waveform
    
    def close(self):
        self.logger.info(f"\n[SESSION COMPLETE v8.4]")
        self.logger.info(f"  Log: {self.logger.log_path}")
        self.logger.close()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ISAC CHAIN v8.4 - CORRECTED RADAR SENSING")
    print("="*80)