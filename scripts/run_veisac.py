# run_veisac.py
"""
VeISAC — End-to-End Evaluation Script

Runs the full MIMO-OFDM-FMCW ISAC chain across one or all baseline topologies (B1: Mono. BS, B2: Bist. BS, B3: Bist. UE), sweeping SNR operating points and reporting communication, sensing, and unified metrics.

Paper: "VeISAC: An End-to-End MIMO-OFDM-FMCW Framework for ISAC
        in 6G Vehicular Networks"
Authors: M. Ababsa, S. Ribouh, Y. El Hillali, A. Rivenq
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from tqdm import tqdm
import warnings
import logging
import sys
from datetime import datetime
from typing import Dict, Any, List
import traceback

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np

warnings.filterwarnings('ignore')

# ── TOPOLOGY SWITCH ─────────────────────────────────────────────────────────
# Set RADAR_MODE to 'monostatic' or 'bistatic' to switch between datasets.
# Everything below auto-configures from this single flag.
RADAR_MODE = 'bistatic_ue'   # ← 'monostatic' | 'bistatic_ue' | 'bistatic_bs'

if RADAR_MODE == 'monostatic':
    HDF5_FILE   = Path('/media/mababsa/Expansion/Exported_Datasets/'
                       'exported_dataset_O1_28GHz_200MHz/hdf5_datasets/'
                       'BC_mono_BS/isac_BC_Monostatic_BS_0-499.h5')
    OUTPUT_BASE = Path('/home/mababsa/DeepVerse-6G/logs_28GHz_200MHz/BC_mono_BS')
    N_RADAR_RX_ANTENNAS = 4          # BS 2×2 UPA
    DELAY_FACTOR        = 2.0        # round-trip
    DOPPLER_FACTOR      = 2.0        # two-way
    _BISTATIC_MODE      = 'monostatic'
elif RADAR_MODE == 'bistatic_bs':
    HDF5_FILE   = Path('/media/mababsa/Expansion/Exported_Datasets/'
                       'exported_dataset_O1_28GHz_200MHz/hdf5_datasets/'
                       'BC_bist_BS/isac_BC_Bistatic_BS_0-499.h5')
    OUTPUT_BASE = Path('/home/mababsa/DeepVerse-6G/logs_28GHz_200MHz/BC_bist_BS')
    N_RADAR_RX_ANTENNAS = 4          # BS 2×2 UPA
    DELAY_FACTOR        = 1.0        # one-way TX→target→RX
    DOPPLER_FACTOR      = 1.0        # one-way
    _BISTATIC_MODE      = 'bistatic'
else:   # bistatic_ue
    HDF5_FILE   = Path('/media/mababsa/Expansion/Exported_Datasets/'
                       'exported_dataset_O1_28GHz_200MHz/hdf5_datasets/'
                       'BC_bist_UE/isac_BC_Bistatic_UE_0-499.h5')
    OUTPUT_BASE = Path('/home/mababsa/DeepVerse-6G/logs_28GHz_200MHz/BC_bist_UE')
    N_RADAR_RX_ANTENNAS = 2          # UE 2×1 ULA
    DELAY_FACTOR        = 1.0        # one-way TX→target→RX
    DOPPLER_FACTOR      = 1.0        # one-way
    _BISTATIC_MODE      = 'bistatic'

COMM_OUTPUT_DIR   = OUTPUT_BASE / 'communication'
SENSING_OUTPUT_DIR = OUTPUT_BASE / 'sensing'

MAX_SAMPLES = None
USE_GPU = False
VERBOSE = True

# ========== BATCH PROCESSING CONFIGURATION ==========
# Process dataset in batches to prevent RAM exhaustion.
# Each batch is saved to CSV independently then cleared from memory.
# Tune BATCH_SIZE based on available RAM:
#   - 16 GB RAM → BATCH_SIZE = 50
#   - 32 GB RAM → BATCH_SIZE = 100
#   - 64 GB RAM → BATCH_SIZE = 200
BATCH_SIZE = 25
# =====================================================

COMM_SNR_DB = 5.0
RADAR_SNR_DB = 10.0

FMCW_INTEGRATION_MODE = 'additive'
COMM_POWER_FACTOR = np.sqrt(0.5)
RADAR_POWER_FACTOR = np.sqrt(0.5)

ENABLE_INTERFERENCE_CANCELLATION = False
IC_ITERATIONS = 2

ENABLE_ANGLE_ESTIMATION = True

# ========== DYNAMIC TARGET FILTERING (MTI MODE) ==========
# Filter radar ground truth to evaluate only moving targets
# Simulates Moving Target Indication (MTI) for automotive scenarios
ENABLE_DYNAMIC_ONLY_EVALUATION = False
DYNAMIC_VELOCITY_THRESHOLD = 2.0   # m/s (anything slower = static clutter)
# ==========================================================

# ========== OFDM INTERFERENCE MITIGATION IN SENSING (NEW) ==========
# NLMS adaptive filtering to remove OFDM interference from radar path
# Applied in time domain BEFORE de-chirping in SensingReceiver
# Theory: Removes α·H_radar ⊗ x_ofdm(t) while preserving β·H_radar ⊗ c(t)

ENABLE_OFDM_MITIGATION = True   # Enable NLMS in sensing receiver
NLMS_FILTER_LENGTH = 32        # Filter taps (32-128 typical)
NLMS_STEP_SIZE = 0.01          # Learning rate (0.001-0.05 typical)

# Expected impact when enabled (α²=β²=0.5):
#   - SINR improvement: ~3-5 dB+

#   - P_fa reduction: ~50-70%
#   - Detection probability increase: ~5-10%
# ====================================================================

LIGHTSPEED = 299792458.0
CARRIER_FREQ_HZ = 28e9

try:
    from veisac.isac_chain import ISACChain
    from veisac.tx.isac_tx_config import ISACTXConfig, get_default_config as get_default_tx_config
    from veisac.rx.isac_rx_config import ISACRXConfig, get_default_config as get_default_rx_config
    print("✓ ISAC modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import ISAC modules: {e}")
    sys.exit(1)

def setup_logging(output_dir: Path, log_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f'{log_name}_{timestamp}.txt'
    
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def load_radar_ground_truths(sample_group, use_closest=True):
    radar_gt_grp = sample_group['radar/gt']
    
    gt_closest = {}
    closest_grp = radar_gt_grp['gt_closest']
    for key in closest_grp.attrs.keys():
        gt_closest[key] = closest_grp.attrs[key]
    
    all_gts = []
    for key in sorted(radar_gt_grp.keys()):
        if key.startswith('gt_') and key != 'gt_closest':
            tgt_grp = radar_gt_grp[key]
            tgt = {}
            for k in tgt_grp.attrs.keys():
                tgt[k] = tgt_grp.attrs[k]
            all_gts.append(tgt)
    
    if use_closest:
        return gt_closest, all_gts
    else:
        return all_gts

class DeepVerse6GLoader:
    def __init__(self, h5_file: Path, logger=None):
        self.h5_file = h5_file
        self.logger = logger or logging.getLogger(__name__)
        self.f = None
        self.n_samples = 0
        
        if not self.h5_file.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_file}")
    
    def __enter__(self):
        self.f = h5py.File(self.h5_file, 'r')
        self.n_samples = len([k for k in self.f.keys() if k.startswith('sample_')])
        self.logger.info(f"Opened HDF5: {self.h5_file}")
        self.logger.info(f"Total samples: {self.n_samples}")
        return self
    
    def __exit__(self, *args):
        if self.f:
            self.f.close()
    
    def load_sample(self, sample_idx: int) -> Dict[str, Any]:
        sample_key = f'sample_{sample_idx:06d}'
        if sample_key not in self.f:
            raise KeyError(f"Sample {sample_key} not found")
    
        sample = self.f[sample_key]
        metadata = dict(sample['metadata'].attrs)
    
        H_radar = sample['radar/waveform'][:]
    
        gt_closest, all_gts = load_radar_ground_truths(sample, use_closest=True)
    
        H_comm = sample['comm/channel'][:]
        comm_gt_closest = dict(sample['comm/gt_closest'].attrs)
        comm_gt_strongest = dict(sample['comm/gt_strongest'].attrs)
        comm_metrics = dict(sample['comm/channel_metrics'].attrs)
    
        return {
            'sample_idx': sample_idx,
            'metadata': metadata,
            'radar': {
                'H_channel': H_radar,
                'gt': gt_closest,
                'all_gts': all_gts,
                'n_targets': len(all_gts)
            },
            'comm': {
                'H_channel': H_comm,
                'gt_closest': comm_gt_closest,
                'gt_strongest': comm_gt_strongest,
                'channel_metrics': comm_metrics,
            }
        }

def apply_radar_channel_vectorized(H_radar, tx_signal, beta, use_gpu=False):
    print(f"\n  {'─'*76}")
    print(f"  [apply_radar_channel_vectorized - v8.4 CORRECTED]")
    print(f"  {'─'*76}")
    print(f"    H_radar shape: {H_radar.shape}")
    print(f"    tx_signal shape: {tx_signal.shape}")
    print(f"    β (radar power factor): {beta:.6f}")
    print(f"    β² (power fraction): {beta**2:.6f}")
    
    xp = cp if use_gpu and GPU_AVAILABLE else np
    
    if use_gpu and GPU_AVAILABLE:
        H_radar = cp.asarray(H_radar)
        tx_signal = cp.asarray(tx_signal)
    
    N_rx, N_tx, N_samples, N_chirps = H_radar.shape
    
    tx_signal_scaled = beta * tx_signal
    
    tx_power_before = np.mean(np.abs(tx_signal)**2)
    tx_power_after = np.mean(np.abs(tx_signal_scaled)**2)
    
    print(f"    TX power (before β): {tx_power_before:.6e}")
    print(f"    TX power (after β): {tx_power_after:.6e}")
    print(f"    Scaling ratio: {tx_power_after/tx_power_before:.6f} (should be β²={beta**2:.6f})")
    
    rx_signal = xp.zeros((N_rx, N_tx, N_samples, N_chirps), dtype=complex)
    
    for rx_idx in range(N_rx):
        for tx_idx in range(N_tx):
            rx_signal[rx_idx, tx_idx] = H_radar[rx_idx, tx_idx] * tx_signal_scaled[tx_idx]
    
    if use_gpu and GPU_AVAILABLE:
        rx_signal = cp.asnumpy(rx_signal)
    
    rx_power = np.mean(np.abs(rx_signal)**2)
    print(f"    RX signal power (after channel): {rx_power:.6e}")
    print(f"    Status: ✅ β-scaled correctly")
    print(f"  {'─'*76}\n")
    
    return rx_signal

def add_awgn_radar_aware(signal, snr_db, beta, use_gpu=False, logger=None):
    print(f"\n  {'─'*76}")
    print(f"  [add_awgn_radar_aware - v8.4 CORRECTED]")
    print(f"  {'─'*76}")
    print(f"    SNR target: {snr_db} dB")
    print(f"    β (radar power factor): {beta:.6f}")
    print(f"    β²: {beta**2:.6f}")
    
    xp = cp if use_gpu and GPU_AVAILABLE else np
    
    if use_gpu and GPU_AVAILABLE:
        signal = cp.asarray(signal)
    
    signal_power_measured = xp.mean(xp.abs(signal)**2)
    print(f"    Signal power (measured): {float(signal_power_measured):.6e}")
    
    snr_linear = 10**(snr_db/10)
    noise_power = signal_power_measured / snr_linear
    
    print(f"    SNR linear: {snr_linear:.2f}")
    print(f"    Noise power: {float(noise_power):.6e}")
    print(f"    Noise referenced to β-scaled signal power")
    
    if logger:
        logger.info(f"  RADAR noise (β-aware): {float(noise_power):.6e}")
    
    noise_std = xp.sqrt(noise_power / 2)
    noise = noise_std * (xp.random.randn(*signal.shape) + 1j * xp.random.randn(*signal.shape))
    
    noisy_signal = signal + noise
    
    actual_signal_power = xp.mean(xp.abs(signal)**2)
    actual_noise_power = xp.mean(xp.abs(noise)**2)
    actual_snr_linear = actual_signal_power / actual_noise_power
    actual_snr_db = 10 * xp.log10(actual_snr_linear)
    
    print(f"    Actual SNR achieved: {float(actual_snr_db):.2f} dB")
    print(f"    Actual noise power: {float(actual_noise_power):.6e}")
    print(f"    Status: ✅ RADAR noise correct")
    print(f"  {'─'*76}\n")
    
    if logger:
        logger.info(f"  Actual RADAR SNR: {float(actual_snr_db):.2f} dB")
    
    if use_gpu and GPU_AVAILABLE:
        noisy_signal = cp.asnumpy(noisy_signal)
    
    return noisy_signal, float(noise_power)

def apply_comm_channel_freq_domain(H_comm, tx_freq_grid, use_gpu=False):
    print(f"\n  [apply_comm_channel_freq_domain]")
    print(f"    H_comm shape: {H_comm.shape}")
    print(f"    tx_freq_grid shape: {tx_freq_grid.shape}")
    
    xp = cp if use_gpu and GPU_AVAILABLE else np
    
    if use_gpu and GPU_AVAILABLE:
        H_comm = cp.asarray(H_comm)
        tx_freq_grid = cp.asarray(tx_freq_grid)
    
    if H_comm.shape[0] > H_comm.shape[1]:
        H_comm = xp.transpose(H_comm, (1, 0, 2))
        print(f"    Transposed H_comm to: {H_comm.shape}")
    
    N_rx, N_tx, N_subcarriers = H_comm.shape
    
    if tx_freq_grid.ndim == 2:
        N_symbols = tx_freq_grid.shape[0]
        tx_freq_grid_expanded = xp.zeros((N_tx, N_symbols, N_subcarriers), dtype=complex)
        tx_freq_grid_expanded[0, :, :] = tx_freq_grid
        tx_freq_grid = tx_freq_grid_expanded
        print(f"    Expanded tx_freq_grid to: {tx_freq_grid.shape}")
    
    N_tx, N_symbols, N_subcarriers = tx_freq_grid.shape
    rx_freq_grid = xp.zeros((N_rx, N_symbols, N_subcarriers), dtype=complex)
    
    for sym_idx in range(N_symbols):
        for sc_idx in range(N_subcarriers):
            tx_vec = tx_freq_grid[:, sym_idx, sc_idx]
            rx_freq_grid[:, sym_idx, sc_idx] = H_comm[:, :, sc_idx] @ tx_vec
    
    if use_gpu and GPU_AVAILABLE:
        rx_freq_grid = cp.asnumpy(rx_freq_grid)
    
    rx_power = np.mean(np.abs(rx_freq_grid)**2)
    print(f"    rx_freq_grid power: {rx_power:.6e}")
    
    return rx_freq_grid

def add_awgn(signal, snr_db, use_gpu=False, logger=None, noise_power_override=None):
    print(f"\n  [add_awgn v8.3_FINAL]")
    print(f"    SNR target: {snr_db} dB")
    print(f"    Override provided: {noise_power_override is not None}")
    
    xp = cp if use_gpu and GPU_AVAILABLE else np
    
    if use_gpu and GPU_AVAILABLE:
        signal = cp.asarray(signal)
    
    signal_power_measured = xp.mean(xp.abs(signal)**2)
    print(f"    Signal power (measured): {float(signal_power_measured):.6e}")
    
    if noise_power_override is not None:
        noise_power = noise_power_override
        print(f"    ✅ Using OVERRIDE noise power: {noise_power:.6e}")
        if logger:
            logger.info(f"  Using OVERRIDE noise power: {noise_power:.6e}")
    else:
        signal_power = xp.mean(xp.abs(signal)**2)
        snr_linear = 10**(snr_db/10)
        noise_power = signal_power / snr_linear
        print(f"    Computing noise from signal power:")
        print(f"      Signal power: {float(signal_power):.6e}")
        print(f"      SNR linear: {snr_linear:.2f}")
        print(f"      Noise power: {float(noise_power):.6e}")
        if logger:
            logger.info(f"  Signal power: {float(signal_power):.6e}")
            logger.info(f"  Noise power: {float(noise_power):.6e}")
    
    noise_std = xp.sqrt(noise_power / 2)
    noise = noise_std * (xp.random.randn(*signal.shape) + 1j * xp.random.randn(*signal.shape))
    
    noisy_signal = signal + noise
    
    actual_signal_power = xp.mean(xp.abs(signal)**2)
    actual_noise_power = xp.mean(xp.abs(noise)**2)
    actual_snr_linear = actual_signal_power / actual_noise_power
    actual_snr_db = 10 * xp.log10(actual_snr_linear)
    
    print(f"    Actual SNR achieved: {float(actual_snr_db):.2f} dB")
    print(f"    Actual noise power: {float(actual_noise_power):.6e}")
    
    if logger:
        logger.info(f"  Actual SNR: {float(actual_snr_db):.2f} dB")
    
    if use_gpu and GPU_AVAILABLE:
        noisy_signal = cp.asnumpy(noisy_signal)
    
    return noisy_signal


def generate_fmcw_chirp_freq_domain(tx_config) -> np.ndarray:
    print(f"\n{'─'*80}")
    print(f"[GENERATE FMCW CHIRP FOR IC - v8.3_FINAL]")
    print(f"{'─'*80}")
    
    n_samples = tx_config.fmcw_n_samples_per_chirp
    bandwidth_hz = tx_config.fmcw_bandwidth_hz
    t_chirp = tx_config.fmcw_chirp_duration_s
    
    print(f"  FMCW Parameters:")
    print(f"    N_samples: {n_samples}")
    print(f"    Bandwidth: {bandwidth_hz/1e6:.1f} MHz")
    print(f"    T_chirp: {t_chirp*1e6:.2f} μs")
    
    t = np.linspace(0, t_chirp, n_samples, endpoint=False)
    mu = bandwidth_hz / t_chirp
    
    print(f"    Chirp rate μ: {mu:.3e} Hz/s")
    
    chirp_time = np.exp(1j * np.pi * mu * t**2)
    
    chirp_fft = np.fft.fft(chirp_time) / np.sqrt(n_samples)
    
    n_sc = tx_config.n_subcarriers_actual
    n_left = n_sc // 2
    n_right = n_sc - n_left
    sc_indices = np.concatenate([np.arange(1, n_right + 1), np.arange(-n_left, 0)])
    
    chirp_freq = chirp_fft[sc_indices]
    
    chirp_power = np.mean(np.abs(chirp_freq)**2)
    chirp_freq = chirp_freq / np.sqrt(chirp_power)
    
    chirp_power_normalized = np.mean(np.abs(chirp_freq)**2)
    
    print(f"\n  Output:")
    print(f"    chirp_freq shape: {chirp_freq.shape}")
    print(f"    chirp_freq power: {chirp_power_normalized:.6e}")
    print(f"    Status: {'✅ OK' if abs(chirp_power_normalized - 1.0) < 0.2 else '⚠️  CHECK'}")
    
    print(f"\n  Mathematical Model:")
    print(f"    Time-domain: c(t) = exp(j·π·μ·t²)")
    print(f"    Frequency-domain: C[k] = FFT(c(t))")
    print(f"    Stage 1 (Pilot IC): Clean pilots → better H_est")
    print(f"    Stage 2 (Data IC): Reconstruct ŷ_radar = 1.06 × β × H × C[k]")
    print(f"{'─'*80}\n")
    
    return chirp_freq


class ISACChainRealChannelProcessor:
    def __init__(self, isac_chain: ISACChain, comm_snr_db: float, radar_snr_db: float, 
                 enable_ic: bool = True, enable_angle_estimation: bool = False, logger=None):
        self.chain = isac_chain
        self.comm_snr_db = comm_snr_db
        self.radar_snr_db = radar_snr_db
        self.enable_ic = enable_ic
        self.enable_angle_estimation = enable_angle_estimation
        self.logger = logger or logging.getLogger(__name__)
        
        self.tx_config = isac_chain.tx_config
        self.rx_config = isac_chain.rx_config
        self.use_gpu = isac_chain.use_gpu
        
        self.fmcw_integration_mode = isac_chain.fmcw_integration_mode
        self.comm_power_factor = isac_chain.comm_power_factor
        self.radar_power_factor = isac_chain.radar_power_factor
        
        print(f"\n{'='*80}")
        print(f"[PROCESSOR INITIALIZED v8.5 - ACTUAL TX WAVEFORM]")
        print(f"{'='*80}")
        print(f"  ISAC mode: {self.fmcw_integration_mode.upper()}")
        print(f"  Power: α={self.comm_power_factor:.6f}, β={self.radar_power_factor:.6f}")
        print(f"  Power fractions: α²={self.comm_power_factor**2:.6f}, β²={self.radar_power_factor**2:.6f}")
        print(f"  COMM SNR target: {self.comm_snr_db} dB")
        print(f"  RADAR SNR target: {self.radar_snr_db} dB")
        print(f"  IC enabled: {self.enable_ic}")
        print(f"  Angle estimation: {self.enable_angle_estimation}")
        if self.enable_ic and self.fmcw_integration_mode == 'additive':
            print(f"  Pilot IC iterations: {isac_chain.ic_iterations}")
            print(f"  Data-Domain IC: ENABLED (scale=1.06, threshold=40%)")
            print(f"  Expected: Significant BER reduction")
        print(f"\n  RADAR CRITICAL FIXES:")
        print(f"    ✅ β power scaling applied to TX signal")
        print(f"    ✅ Correct parameter name: received_signal (not if_signal)")
        print(f"    ✅ Virtual array extraction for MUSIC: {self.enable_angle_estimation}")
        print(f"    ✅ β-aware noise computation")

        # NEW: OFDM mitigation status
        if hasattr(self.chain.sens_rx, 'enable_ofdm_mitigation'):
            ofdm_mit_status = self.chain.sens_rx.enable_ofdm_mitigation
        else:
            ofdm_mit_status = False

        print(f"\n  OFDM INTERFERENCE MITIGATION:")
        print(f"    Enabled in sensing: {ofdm_mit_status}")
        if ofdm_mit_status:
            print(f"    Algorithm: NLMS adaptive filter")
            print(f"    Filter length: {getattr(self.chain.sens_rx, 'nlms_filter_length', 64)} taps")
            print(f"    Step size: {getattr(self.chain.sens_rx, 'nlms_step_size', 0.01)}")
            print(f"    Expected: SINR improvement, lower P_fa")

        print(f"{'='*80}\n")
    
    def process_real_sample(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        sample_idx = sample_data['sample_idx']
        metadata = sample_data['metadata']
    
        H_radar_raw = sample_data['radar']['H_channel']
        radar_gt = sample_data['radar']['gt']
        radar_all_gts = sample_data['radar']['all_gts']
        radar_n_targets = sample_data['radar']['n_targets']
    
        H_comm_raw = sample_data['comm']['H_channel']
        comm_gt = sample_data['comm']['gt_closest']
        
        print(f"\n{'='*80}")
        print(f"[PROCESSING SAMPLE {sample_idx} - v8.5 ACTUAL TX WAVEFORM]")
        print(f"{'='*80}")
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Processing Sample {sample_idx} (v8.5 ACTUAL TX WAVEFORM)")
        self.logger.info(f"{'='*80}")
        
        try:
            print(f"\n{'─'*80}")
            print(f"[STEP 1: CHANNEL NORMALIZATION]")
            print(f"{'─'*80}")
            
            H_comm_power_original = np.mean(np.abs(H_comm_raw)**2)
            print(f"  H_comm (original):")
            print(f"    Shape: {H_comm_raw.shape}")
            print(f"    Power: {H_comm_power_original:.6e}")
            
            H_comm = H_comm_raw / np.sqrt(H_comm_power_original)
            H_comm_power_normalized = np.mean(np.abs(H_comm)**2)
            print(f"  H_comm (normalized):")
            print(f"    Power: {H_comm_power_normalized:.6f}")
            print(f"    Status: {'✅ OK' if abs(H_comm_power_normalized - 1.0) < 0.01 else '⚠️  ISSUE'}")
            
            H_radar_power_original = np.mean(np.abs(H_radar_raw)**2)
            print(f"\n  H_radar (original):")
            print(f"    Shape: {H_radar_raw.shape}")
            print(f"    Power: {H_radar_power_original:.6e}")
            
            H_radar = H_radar_raw / np.sqrt(H_radar_power_original)
            H_radar_power_normalized = np.mean(np.abs(H_radar)**2)
            print(f"  H_radar (normalized):")
            print(f"    Power: {H_radar_power_normalized:.6f}")
            print(f"    Status: {'✅ OK' if abs(H_radar_power_normalized - 1.0) < 0.01 else '⚠️  ISSUE'}")
            
            self.logger.info(f"✅ Channels normalized: H_comm {H_comm_power_normalized:.6f}, H_radar {H_radar_power_normalized:.6f}")

            print(f"\n{'─'*80}")
            print(f"[RADAR GROUND TRUTH - MULTI-TARGET ENVIRONMENT]")
            print(f"{'─'*80}")
            print(f"  Total targets in scene: {radar_n_targets}")
            
            # ═══════════════════════════════════════════════════════════════
            # DYNAMIC-ONLY FILTERING (MTI MODE)
            # ═══════════════════════════════════════════════════════════════
            gt_for_evaluation = radar_gt  # default = closest target
            
            if ENABLE_DYNAMIC_ONLY_EVALUATION:
                # Filter for dynamic targets only
                dynamic_gts = [
                    tgt for tgt in radar_all_gts 
                    if abs(tgt.get('velocity_ms', 0.0)) > DYNAMIC_VELOCITY_THRESHOLD
                ]
                
                print(f"\n  DYNAMIC FILTERING ENABLED:")
                print(f"    Threshold: {DYNAMIC_VELOCITY_THRESHOLD} m/s")
                print(f"    Total dynamic targets: {len(dynamic_gts)}")
                print(f"    Static targets removed: {len(radar_all_gts) - len(dynamic_gts)}")
                
                if len(dynamic_gts) > 0:
                    # Use closest dynamic target for evaluation
                    closest_dynamic = min(dynamic_gts, key=lambda t: t.get('range_m', float('inf')))
                    gt_for_evaluation = closest_dynamic
                    
                    print(f"\n  CLOSEST DYNAMIC TARGET (used for evaluation):")
                    print(f"    Range: {closest_dynamic.get('range_m', 'N/A'):.3f} m")
                    print(f"    Velocity: {closest_dynamic.get('velocity_ms', 0.0):+.2f} m/s")
                    print(f"    Power: {closest_dynamic.get('power_db', 'N/A'):.1f} dB")
                    
                    # Show some other dynamic targets
                    if len(dynamic_gts) > 1:
                        print(f"\n  Additional dynamic targets ({len(dynamic_gts)-1}):")
                        for i, tgt in enumerate(dynamic_gts[1:6], start=1):
                            print(f"    Target {i+1}: range={tgt['range_m']:.2f}m, vel={tgt['velocity_ms']:+.2f}m/s, pwr={tgt['power_db']:.1f}dB")
                        if len(dynamic_gts) > 6:
                            print(f"    ... and {len(dynamic_gts) - 6} more dynamic targets")
                else:
                    print(f"\n  ⚠️  WARNING: No dynamic targets found!")
                    print(f"    Falling back to closest target (may be static)")
                    print(f"    Range: {radar_gt.get('range_m', 'N/A')} m")
                    print(f"    Velocity: {radar_gt.get('velocity_ms', 'N/A')} m/s")
                    gt_for_evaluation = radar_gt
            else:
                # Standard mode: pass ALL GT targets for full multi-target evaluation
                print(f"\n  DYNAMIC FILTERING: DISABLED")
                print(f"  Closest target (used for evaluation):")
                print(f"    Range: {radar_gt.get('range_m', 'N/A')} m")
                print(f"    Velocity: {radar_gt.get('velocity_ms', 'N/A')} m/s")
                print(f"    Power: {radar_gt.get('power_db', 'N/A')} dB")

                # ← KEY FIX: use full list not just closest target
                gt_for_evaluation = radar_all_gts

                if radar_n_targets > 1:
                    print(f"\n  Additional targets ({radar_n_targets - 1}):")
                    for i, tgt in enumerate(radar_all_gts[1:6], start=1):
                        print(f"    Target {i+1}: range={tgt['range_m']:.2f}m, vel={tgt['velocity_ms']:+.2f}m/s, pwr={tgt['power_db']:.1f}dB")
                    if radar_n_targets > 6:
                        print(f"    ... and {radar_n_targets - 6} more targets")

            self.logger.info(f"  Radar targets: {radar_n_targets} (closest: {radar_gt.get('range_m', 'N/A')}m)")
            if ENABLE_DYNAMIC_ONLY_EVALUATION and 'dynamic_gts' in locals():
                self.logger.info(f"  Dynamic-only evaluation: {len(dynamic_gts)} targets")

            print(f"\n{'─'*80}")
            print(f"[STEP 2: SENSING PROCESSING - v8.4 CORRECTED]")
            print(f"{'─'*80}")
            
            self.logger.info("\n--- SENSING (v8.4 CORRECTED) ---")
            
            N_tx = self.tx_config.n_tx_antennas
            N_samples = self.tx_config.fmcw_n_samples_per_chirp
            N_chirps = self.tx_config.fmcw_n_chirps
            
            print(f"\n  CRITICAL FIXES APPLIED:")
            print(f"    ✅ β={self.radar_power_factor:.6f} power scaling")
            print(f"    ✅ Correct parameter: received_signal (not if_signal)")
            print(f"    ✅ Virtual array extraction: {self.enable_angle_estimation}")
            print(f"    ✅ β-aware noise computation")
            
            print(f"\n  Generating TX FMCW chirps...")
            print(f"    N_tx: {N_tx}, N_samples: {N_samples}, N_chirps: {N_chirps}")
            
            tx_radar_signal = np.zeros((N_tx, N_samples, N_chirps), dtype=complex)
            for tx_idx in range(N_tx):
                for chirp_idx in range(N_chirps):
                    t = np.arange(N_samples) / self.tx_config.fmcw_sampling_rate_hz
                    phase = 2 * np.pi * (
                        self.tx_config.carrier_freq_hz * t +
                        0.5 * self.tx_config.fmcw_chirp_slope * t**2
                    )
                    tx_radar_signal[tx_idx, :, chirp_idx] = np.exp(1j * phase)
            
            tx_power = np.mean(np.abs(tx_radar_signal)**2)
            print(f"    TX signal power (before β): {tx_power:.6e}")
            
            self.logger.info(f"  β (radar power factor): {self.radar_power_factor:.6f}")
            self.logger.info(f"  TX power (before β): {tx_power:.6e}")
            
            rx_radar_signal = apply_radar_channel_vectorized(
                H_radar, tx_radar_signal, beta=self.radar_power_factor, use_gpu=self.use_gpu
            )
            
            self.logger.info(f"  RX power (after β-scaled channel): {np.mean(np.abs(rx_radar_signal)**2):.6e}")
            
            rx_radar_signal_noisy, radar_thermal_noise_power = add_awgn_radar_aware(
                rx_radar_signal, self.radar_snr_db, beta=self.radar_power_factor,
                use_gpu=self.use_gpu, logger=self.logger
            )
            
            tx_ofdm_td = None
            
            # ════════════════════════════════════════════════════════════════
            # STEP 2.5: GENERATE TX OFDM SLOT
            # Must run BEFORE Step 2B so last_transmitted_ofdm is populated.
            # tx_result is reused in Step 3 (COMM) — same seed ensures
            # the OFDM interference added to rx_radar and the comm data bits
            # correspond to the exact same transmitted slot.
            # ════════════════════════════════════════════════════════════════
            print(f"\n{'─'*80}")
            print(f"[STEP 2.5: GENERATE TX OFDM SLOT]")
            print(f"{'─'*80}")
            seed = sample_idx + 1000
            tx_result = self.chain.tx.transmit_slot(n_ue=0, seed=seed)

            freq_grid_tx  = tx_result['freq_grid']
            pilot_mask    = tx_result['pilot_mask']
            data_bits     = tx_result['data_bits']
            data_symbols  = tx_result['data_symbols']

            pilot_positions = np.where(pilot_mask)
            pilot_symbols   = freq_grid_tx[pilot_positions]
            tx_freq_power   = np.mean(np.abs(freq_grid_tx)**2)

            print(f"  seed:             {seed}")
            print(f"  freq_grid shape:  {freq_grid_tx.shape}")
            print(f"  pilot_mask shape: {pilot_mask.shape}")
            print(f"  N pilots:         {len(pilot_symbols)}")
            print(f"  TX freq power:    {tx_freq_power:.6e}")
            print(f"  last_transmitted_ofdm available: "
                  f"{'✅ YES' if self.chain.tx.last_transmitted_ofdm is not None else '❌ NO'}")
            if self.chain.tx.last_transmitted_ofdm is not None:
                print(f"  last_transmitted_ofdm shape: "
                      f"{self.chain.tx.last_transmitted_ofdm.shape}")
                print(f"  last_transmitted_ofdm power: "
                      f"{np.mean(np.abs(self.chain.tx.last_transmitted_ofdm)**2):.4e}")

            self.logger.info(f"  TX freq power: {tx_freq_power:.6e}")

            # ════════════════════════════════════════════════════════════════
            # STEP 2B: PROPAGATE OFDM THROUGH H_RADAR — ADD INTERFERENCE
            # ════════════════════════════════════════════════════════════════
            # The correct additive ISAC received radar signal is:
            #   r(t) = β·H·c(t) + α·H·x_ofdm(t) + n(t)
            # Currently rx_radar_signal_noisy = β·H·c(t) + n(t) only.
            # last_transmitted_ofdm = α·x_ofdm (shape N_tx × 43008)
            # already stored by ISACTransmitter.transmit_slot() above.
            # We propagate it through H_radar and add — NO new signals generated.
            # ════════════════════════════════════════════════════════════════
            print(f"\n{'─'*80}")
            print(f"[STEP 2B: OFDM INTERFERENCE PROPAGATION THROUGH H_RADAR]")
            print(f"{'─'*80}")

            if (self.fmcw_integration_mode == 'additive' and
                    hasattr(self.chain.tx, 'last_transmitted_ofdm') and
                    self.chain.tx.last_transmitted_ofdm is not None and
                    hasattr(self.chain.sens_rx, 'enable_ofdm_mitigation') and
                    self.chain.sens_rx.enable_ofdm_mitigation):

                x_ofdm_tx = self.chain.tx.last_transmitted_ofdm   # (N_tx, 43008)
                print(f"  Source: last_transmitted_ofdm (α·x_ofdm from ISACTransmitter)")
                print(f"  Shape:  {x_ofdm_tx.shape}   (N_tx × N_ofdm_samples)")
                print(f"  Power:  {np.mean(np.abs(x_ofdm_tx)**2):.4e}")
                print(f"  α:      {self.comm_power_factor:.6f}  "
                      f"(α² = {self.comm_power_factor**2:.4f})")

                N_tx_r = self.tx_config.n_tx_antennas
                N_rx_r = H_radar.shape[0]   # 4 mono/bistatic-BS, 2 bistatic-UE
                N_s    = self.tx_config.fmcw_n_samples_per_chirp   # 1664
                N_c    = self.tx_config.fmcw_n_chirps               # 128
                N_tot  = N_s * N_c                                   # 212992
                N_ofdm = x_ofdm_tx.shape[1]                         # 43008
                # bistatic: N_rx_r=2 → rx_ofdm shape (2,4,1664,128) — correct

                print(f"\n  Radar frame: N_rx={N_rx_r}, N_tx={N_tx_r}, "
                      f"N_s={N_s}, N_c={N_c}, total={N_tot}")
                print(f"  OFDM slot:   {N_ofdm} samples")
                print(f"  Tiling:      ceil({N_tot}/{N_ofdm}) = "
                      f"{int(np.ceil(N_tot/N_ofdm))} tiles")

                # ── Step A: Tile OFDM to full radar frame length, reshape ──
                print(f"\n  [Step A] Tile → reshape to (N_tx, N_s={N_s}, N_c={N_c})...")
                x_ofdm_frame = np.zeros((N_tx_r, N_s, N_c), dtype=complex)
                for tx_idx in range(N_tx_r):
                    n_tiles = int(np.ceil(N_tot / N_ofdm))
                    tiled   = np.tile(x_ofdm_tx[tx_idx], n_tiles)[:N_tot]
                    x_ofdm_frame[tx_idx] = tiled.reshape(N_s, N_c)

                print(f"    x_ofdm_frame shape: {x_ofdm_frame.shape}")
                print(f"    x_ofdm_frame power: {np.mean(np.abs(x_ofdm_frame)**2):.4e}")
                print(f"    ✅ Matches rx_radar_signal layout (N_tx, N_s, N_c)")

                # ── Step B: Propagate through H_radar per (rx, tx) pair ──
                # H_radar[rx, tx] shape: (N_s, N_c)
                # element-wise mult = flat-fading channel per delay-Doppler cell
                print(f"\n  [Step B] Propagate through H_radar "
                      f"(element-wise per (rx,tx) pair)...")
                rx_ofdm = np.zeros((N_rx_r, N_tx_r, N_s, N_c), dtype=complex)
                for rx_idx in range(N_rx_r):
                    for tx_idx in range(N_tx_r):
                        rx_ofdm[rx_idx, tx_idx] = (H_radar[rx_idx, tx_idx]
                                                    * x_ofdm_frame[tx_idx])

                p_rx_ofdm  = np.mean(np.abs(rx_ofdm)**2)
                p_rx_radar = np.mean(np.abs(rx_radar_signal)**2)
                isr_db     = 10 * np.log10(p_rx_ofdm / (p_rx_radar + 1e-12))
                exp_isr_db = 10 * np.log10(
                    self.comm_power_factor**2 / (self.radar_power_factor**2 + 1e-12))

                print(f"    rx_ofdm shape:  {rx_ofdm.shape}")
                print(f"    rx_ofdm power:  {p_rx_ofdm:.4e}")
                print(f"    rx_radar power: {p_rx_radar:.4e}  (β·H·c — reference)")
                print(f"    ISR measured:   {isr_db:.2f} dB")
                print(f"    ISR expected:   {exp_isr_db:.2f} dB  "
                      f"(α²/β² = {self.comm_power_factor**2:.3f}/"
                      f"{self.radar_power_factor**2:.3f})")
                if abs(isr_db - exp_isr_db) < 3.0:
                    print(f"    ✅ ISR consistent with power allocation")
                else:
                    print(f"    ⚠️  ISR deviation {isr_db-exp_isr_db:+.1f} dB "
                          f"(H_radar normalization effect)")

                # ── Step C: Add OFDM interference to received radar signal ──
                # Before: rx_radar_signal_noisy = β·H·c(t) + n(t)
                # After:  rx_radar_signal_noisy = β·H·c(t) + α·H·x_ofdm(t) + n(t)
                print(f"\n  [Step C] Add OFDM interference to rx_radar_signal_noisy...")
                p_before = np.mean(np.abs(rx_radar_signal_noisy)**2)
                rx_radar_signal_noisy = rx_radar_signal_noisy + rx_ofdm
                p_after  = np.mean(np.abs(rx_radar_signal_noisy)**2)
                isr_lin  = p_rx_ofdm / (p_rx_radar + 1e-12)
                print(f"    Power before: {p_before:.4e}")
                print(f"    Power after:  {p_after:.4e}")
                print(f"    Power increase: {10*np.log10(p_after/p_before):.2f} dB  "
                      f"(expected ≈ {10*np.log10(1+isr_lin):.2f} dB)")
                print(f"    ✅ rx_radar_signal_noisy = β·H·c(t) + α·H·x_ofdm(t) + n(t)")

                # ── Step D: NLMS reference = full per-channel rx_ofdm ───────
                # Pass rx_ofdm (4×4×1664×128) directly — NOT the mean.
                # The NLMS 4D branch in sensing_receiver.py uses each
                # (rx,tx) channel's own reference, giving maximum correlation.
                # Mean would lose per-channel amplitude/phase → weaker NLMS.
                tx_ofdm_td = rx_ofdm.copy()   # (N_rx, N_tx, N_s, N_c)
                print(f"\n  [Step D] Build NLMS reference (full per-channel)...")
                print(f"    tx_ofdm_td shape: {tx_ofdm_td.shape}  "
                      f"(N_rx, N_tx, N_s, N_c) — 4D per-channel reference")
                print(f"    tx_ofdm_td power: {np.mean(np.abs(tx_ofdm_td)**2):.4e}")
                print(f"    ✅ Each (rx,tx) pair has its own matched interference pattern")
                print(f"    ✅ NLMS 4D branch will be used → maximum correlation")

                # ── Step E: Validate correlation ──
                # Theory: corr ≈ α/√(α²+β²) = 0.707 at equal split
                rx_flat  = rx_radar_signal_noisy[0, 0].flatten()[:3000]
                ref_flat = tx_ofdm_td[0, 0].flatten()[:3000]   # (rx=0,tx=0) channel
                corr = float(
                    np.abs(np.dot(ref_flat.conj(), rx_flat)) /
                    (np.linalg.norm(ref_flat) * np.linalg.norm(rx_flat) + 1e-12))
                exp_corr = float(self.comm_power_factor /
                                 np.sqrt(self.comm_power_factor**2
                                         + self.radar_power_factor**2 + 1e-12))

                print(f"\n  [Step E] Correlation validation...")
                print(f"    Correlation (ref vs RX[0,0]): {corr:.4f} ({corr*100:.1f}%)")
                print(f"    Expected (theory):            {exp_corr:.4f} "
                      f"({exp_corr*100:.1f}%)  [α/√(α²+β²)]")
                if corr > 0.3:
                    print(f"    ✅ Good — NLMS will converge, expect 10-15 dB gain")
                elif corr > 0.1:
                    print(f"    ⚠️  Moderate — partial NLMS benefit, expect 3-8 dB")
                else:
                    print(f"    ❌ Low — tiling discontinuities dominate")

                self.logger.info(f"  OFDM added to RX: ISR={isr_db:.2f} dB, "
                                 f"corr={corr:.4f} ({corr*100:.1f}%)")

            else:
                # Report why Step 2B was skipped
                reasons = []
                if self.fmcw_integration_mode != 'additive':
                    reasons.append(f"mode={self.fmcw_integration_mode} (not additive)")
                if not hasattr(self.chain.tx, 'last_transmitted_ofdm') or \
                   self.chain.tx.last_transmitted_ofdm is None:
                    reasons.append("last_transmitted_ofdm not available")
                if not hasattr(self.chain.sens_rx, 'enable_ofdm_mitigation') or \
                   not self.chain.sens_rx.enable_ofdm_mitigation:
                    reasons.append("OFDM mitigation disabled in sensing RX")
                print(f"  ℹ️  Step 2B skipped: {'; '.join(reasons) or 'unknown'}")
                print(f"  tx_ofdm_td = None → NLMS will not run")
                self.logger.info(f"  Step 2B skipped: {'; '.join(reasons)}")

            print(f"{'─'*80}\n")
            
            print(f"\n{'─'*80}")
            print(f"[STEP 3: COMMUNICATION PROCESSING WITH IC (v8.3_FINAL)]")
            print(f"{'─'*80}")
            # tx_result, freq_grid_tx, pilot_mask, data_bits, data_symbols,
            # pilot_symbols already set in Step 2.5 — same seed, same slot.
            print(f"  Reusing tx_result from Step 2.5 (seed={seed}) ✅")
            self.logger.info("\n--- COMMUNICATION (v8.3_FINAL WITH DATA IC) ---")
            self.logger.info(f"  TX freq power: {tx_freq_power:.6e}")

            if self.enable_ic and self.fmcw_integration_mode == 'additive':
                chirp_freq = generate_fmcw_chirp_freq_domain(self.tx_config)
                self.logger.info(
                    f"  ✅ Chirp generated for IC "
                    f"(power: {np.mean(np.abs(chirp_freq)**2):.6e})")
            else:
                chirp_freq = None
                print(f"  ℹ️  IC disabled or not applicable")
                self.logger.info(f"  IC disabled")
            
            # ========== NOW CALL SENSING RECEIVER WITH OFDM WAVEFORM ==========
            print(f"\n  {'─'*76}")
            print(f"  CALLING SENSING RECEIVER v4.2 (WITH OFDM MITIGATION)")
            print(f"  {'─'*76}")
            print(f"    Parameter: received_signal (✅ CORRECT)")
            print(f"    Extract virtual array: {self.enable_angle_estimation}")
            print(f"    OFDM mitigation: {'ENABLED' if tx_ofdm_td is not None else 'DISABLED'}")
            if tx_ofdm_td is not None:
                print(f"    tx_comm_signal shape: {tx_ofdm_td.shape}")
                print(f"    NLMS will cancel: α·H_radar ⊗ x_ofdm(t)")
                print(f"    Expected: SINR improvement ~3-5 dB")
            print(f"    Total targets in scene: {radar_n_targets}")
            if ENABLE_DYNAMIC_ONLY_EVALUATION:
                print(f"    EVALUATION MODE: Dynamic-only (threshold = {DYNAMIC_VELOCITY_THRESHOLD} m/s)")
            print(f"    GT (for evaluation):")
            if isinstance(gt_for_evaluation, list):
                print(f"      Total GT targets passed: {len(gt_for_evaluation)}")
                for i, gt in enumerate(gt_for_evaluation[:3], 1):
                    print(f"      GT{i}: Range={gt.get('range_m','N/A'):.2f}m, "
                          f"Vel={gt.get('velocity_ms',0.0):+.2f}m/s")
                if len(gt_for_evaluation) > 3:
                    print(f"      ... and {len(gt_for_evaluation)-3} more")
            else:
                print(f"      Range: {gt_for_evaluation.get('range_m', 'N/A')} m")
                print(f"      Velocity: {gt_for_evaluation.get('velocity_ms', 'N/A')} m/s")

            self.logger.info(f"  Calling SensingReceiver:")
            self.logger.info(f"    extract_virtual_array: {self.enable_angle_estimation}")
            self.logger.info(f"    OFDM mitigation: {'YES' if tx_ofdm_td is not None else 'NO'}")

            print(f"\n  [RADAR FRAME CALL - SNR/CRB INPUTS DIAGNOSTIC]")
            print(f"    thermal_noise_power: {radar_thermal_noise_power:.6e}  ← Tier 1 noise floor")
            print(f"    input_snr_db:        {self.radar_snr_db:.2f} dB  ← exact design SNR for CRB")
            _mode = getattr(self.chain.sens_rx.config, 'radar_mode', 'monostatic')
            if _mode == 'monostatic':
                _crb_r_exp = 0.750 / (2.0 * (10 ** (self.radar_snr_db / 20)))
                _crb_v_exp = 5.027 / (10 ** (self.radar_snr_db / 20))
                print(f"    Expected CRB_range:  ~{_crb_r_exp:.4f} m  "
                      f"({self.radar_snr_db:.0f} dB, monostatic, 200 MHz BW)")
                print(f"    Expected CRB_vel:    ~{_crb_v_exp:.6f} m/s")
            else:
                _crb_r_exp = 1.501 / (2.0 * (10 ** (self.radar_snr_db / 20)))
                _crb_v_exp = 10.054 / (10 ** (self.radar_snr_db / 20))
                print(f"    Expected CRB_range:  ~{_crb_r_exp:.4f} m  "
                      f"({self.radar_snr_db:.0f} dB, bistatic, 200 MHz BW)")
                print(f"    Expected CRB_vel:    ~{_crb_v_exp:.6f} m/s")
                print(f"    Note: bistatic factors ×2 larger (delay/doppler = 1×)")

            # Velocity matching threshold — must be ≥ 1 Doppler resolution cell.
            # Bistatic:   Δv = λ/(N×T) ≈ 10.05 m/s  → use 6.0 m/s (< 1 bin)
            # Monostatic: Δv = λ/(2×N×T) ≈ 5.03 m/s → use 0.15 m/s (sub-bin)
            _radar_mode_p = getattr(self.chain.sens_rx.config, 'radar_mode', 'monostatic')
            _vel_match_thresh = 0.70 if _radar_mode_p == 'bistatic' else 0.15

            import time
            radar_start = time.time()
            try:
                sensing_result = self.chain.sens_rx.process_radar_frame(
                    received_signal=rx_radar_signal_noisy,
                    ground_truth=gt_for_evaluation,
                    detection_match_threshold_range_m=6.3,
                    detection_match_threshold_velocity_ms=_vel_match_thresh,
                    extract_virtual_array=self.enable_angle_estimation,
                    tx_comm_signal=tx_ofdm_td,
                    enable_ofdm_mitigation=True if tx_ofdm_td is not None else False,
                    thermal_noise_power=radar_thermal_noise_power,
                    input_snr_db=float(self.radar_snr_db)
                )
                radar_time = time.time() - radar_start
                
                print(f"\n  {'─'*76}")
                print(f"  RADAR RESULTS:")
                print(f"  {'─'*76}")
                print(f"    Processing time: {radar_time:.2f}s")
                print(f"    SNR (RDM, post-processing): {sensing_result.get('snr_db', np.nan):.2f} dB")
                print(f"    N detections: {sensing_result.get('n_detections', 0)}")

                if len(sensing_result.get('targets', [])) > 0:
                    top_target = sensing_result['targets'][0]
                    print(f"    Top target range:    {top_target['range_m']:.4f} m")
                    print(f"    Top target velocity: {top_target['velocity_ms']:+.4f} m/s")

                    # ── SNR: new names first, fall back to legacy ─────────────
                    rdm_db   = top_target.get('radar_snr_rdm_db',
                                              top_target.get('snr_db', float('nan')))
                    input_db = top_target.get('radar_snr_input_db',
                                              top_target.get('snr_input_db', float('nan')))
                    src      = top_target.get('radar_snr_input_source',
                                              top_target.get('snr_input_source', 'unknown'))
                    crb_r    = top_target.get('crb_range_m',    float('nan'))
                    crb_v    = top_target.get('crb_velocity_ms', float('nan'))

                    print(f"\n    [SNR BREAKDOWN]")
                    print(f"      radar_snr_rdm   (post-processing): {rdm_db:.2f} dB  "
                          f"← CFAR/ranking")
                    print(f"      radar_snr_input (pre-processing):  {input_db:.2f} dB  "
                          f"← CRB reference (design target: {self.radar_snr_db:.1f} dB)")
                    print(f"      SNR source: {src}")
                    _n_tx_p = self.chain.sens_rx.config.n_radar_tx_antennas
                    _n_rx_p = self.chain.sens_rx.config.n_radar_rx_antennas
                    _mode_p = getattr(self.chain.sens_rx.config, 'radar_mode', 'monostatic')
                    _n_rx_gain = self.chain.sens_rx.config.n_radar_rx_antennas
                    if _mode_p == 'monostatic':
                        _exp_gain = 44.0   # 4×4, delay=2, doppler=2
                    elif _n_rx_gain == 4:
                        _exp_gain = 44.0   # bistatic BS: 4×4, delay=1, doppler=1
                    else:
                        _exp_gain = 41.0   # bistatic UE: 4×2, delay=1, doppler=1
                    print(f"      Processing gain (rdm - input): {rdm_db - input_db:.1f} dB  "
                          f"(expected ≈ {_exp_gain:.0f} dB for {_n_tx_p}×{_n_rx_p}, "
                          f"{_mode_p}, 200 MHz, β²=0.5)")
                    if abs(input_db - self.radar_snr_db) < 3.0:
                        print(f"      ✅ radar_snr_input matches design target")
                    else:
                        print(f"      ❌ radar_snr_input = {input_db:.1f} dB — mismatch! "
                              f"Check input_snr_db flow")

                    print(f"\n    [CRB LOWER BOUNDS — {top_target.get('radar_mode','monostatic').upper()}, "
                          f"radar_snr_input referenced]")
                    print(f"      CRB_range:    {crb_r:.6f} m")
                    print(f"      CRB_velocity: {crb_v:.8f} m/s")
                    if crb_r > 0.005:
                        print(f"      ✅ CRB_range = {crb_r:.4f} m — physically meaningful")
                    else:
                        print(f"      ❌ CRB_range nearly zero — check input_snr_db")

                    if 'angle_deg' in top_target and top_target['angle_deg'] is not None:
                        crb_angle = top_target.get('crb_angle_deg', float('nan'))
                        print(f"    Top target angle: {top_target['angle_deg']:.2f} deg  "
                              f"(CRB_angle: {crb_angle:.4f} deg) ✅ (MUSIC)")
                        self.logger.info(f"  Angle estimated: {top_target['angle_deg']:.2f} deg (MUSIC)")

                if 'range_error_m' in sensing_result:
                    r_err = sensing_result['range_error_m']
                    v_err = sensing_result['velocity_error_ms']
                    print(f"\n    [ESTIMATION ACCURACY vs CRB]")
                    print(f"      Range error (mean):    {r_err:.4f} m")
                    print(f"      Velocity error (mean): {v_err:.4f} m/s")
                    if len(sensing_result.get('targets', [])) > 0:
                        crb_r = sensing_result['targets'][0].get('crb_range_m', float('nan'))
                        crb_v = sensing_result['targets'][0].get('crb_velocity_ms', float('nan'))
                        if crb_r > 1e-9:
                            eff_r = abs(r_err) / (crb_r + 1e-12)
                            eff_v = abs(v_err) / (crb_v + 1e-12)
                            status_r = ("✅ near-optimal" if eff_r < 5
                                        else "⚠️  above CRB" if eff_r < 20
                                        else "❌ far from CRB")
                            print(f"      CRB_range:             {crb_r:.4f} m")
                            print(f"      Range efficiency (err/CRB): {eff_r:.1f}×  {status_r}")
                            print(f"      Velocity efficiency:        {eff_v:.1f}×")

                # ── SCNR ─────────────────────────────────────────────────────
                if sensing_result.get('scnr_db') is not None:
                    print(f"\n    [SCNR — Signal vs Clutter+Noise]")
                    print(f"      SCNR: {sensing_result['scnr_db']:.2f} dB")

                print(f"\n    Detection matched: "
                      f"{'✅ YES' if sensing_result.get('detection_matched') else '❌ NO'}")
                if ENABLE_DYNAMIC_ONLY_EVALUATION:
                    print(f"    → Evaluation mode: DYNAMIC TARGETS ONLY")
                print(f"    ✅ RADAR processing complete")
                print(f"  {'─'*76}\n")

                self.logger.info(f"  ✅ RADAR time: {radar_time:.2f}s")
                self.logger.info(f"  RADAR radar_snr_rdm:   {sensing_result.get('snr_db', np.nan):.2f} dB")
                if len(sensing_result.get('targets', [])) > 0:
                    t0 = sensing_result['targets'][0]
                    _inp = t0.get('radar_snr_input_db', t0.get('snr_input_db', float('nan')))
                    _src = t0.get('radar_snr_input_source', t0.get('snr_input_source', 'unknown'))
                    _crb_r = t0.get('crb_range_m', float('nan'))
                    _crb_v = t0.get('crb_velocity_ms', float('nan'))
                    self.logger.info(f"  RADAR radar_snr_input: {_inp:.2f} dB  (source: {_src})")
                    self.logger.info(f"  RADAR CRB_range:       {_crb_r:.6f} m")
                    self.logger.info(f"  RADAR CRB_velocity:    {_crb_v:.8f} m/s")
                if sensing_result.get('scnr_db') is not None:
                    self.logger.info(f"  RADAR SCNR:            {sensing_result['scnr_db']:.2f} dB")
                self.logger.info(f"  Detection: {'YES' if sensing_result.get('detection_matched') else 'NO'}")
                
            except Exception as e:
                print(f"\n    ❌ RADAR failed: {e}")
                print(f"    {traceback.format_exc()}")
                self.logger.error(f"  RADAR failed: {e}")
                sensing_result = {
                    'range_est_m': np.nan, 'velocity_est_ms': np.nan,
                    'detection_matched': False, 'snr_db': np.nan,
                    'n_detections': 0, 'targets': []
                }
            # ========== END SENSING RECEIVER CALL ==========

            # ── CRITICAL: Strip large arrays from sensing_result ──────────
            # range_doppler_map, detections, threshold_map, virtual_array_data
            # can each be 50–200 MB. We only need scalar metrics for the CSV.
            for _heavy_key in ('range_doppler_map', 'detections',
                               'threshold_map', 'detection_list'):
                sensing_result.pop(_heavy_key, None)
            # Also strip large arrays from each target dict
            for _tgt in sensing_result.get('targets', []):
                for _k in ('virtual_snapshot', 'range_profile', 'doppler_profile'):
                    _tgt.pop(_k, None)
            # ─────────────────────────────────────────────────────────────
            
            print(f"\n  Applying channel...")
            rx_freq_grid = apply_comm_channel_freq_domain(H_comm, freq_grid_tx, self.use_gpu)
            rx_power_before_noise = np.mean(np.abs(rx_freq_grid)**2)
            
            print(f"    RX power (BEFORE noise): {rx_power_before_noise:.6e}")
            self.logger.info(f"  RX power (BEFORE noise): {rx_power_before_noise:.6e}")
            
            print(f"\n{'─'*80}")
            print(f"[COMM-REFERENCED THERMAL NOISE (v7.2)]")
            print(f"{'─'*80}")
            
            self.logger.info(f"\nCOMM-referenced thermal noise (v7.2)")
            
            print(f"  COMPUTATION:")
            print(f"    Total RX power (α²+β²): {rx_power_before_noise:.6e}")
            
            comm_only_power = (self.comm_power_factor ** 2) * rx_power_before_noise
            print(f"    COMM-only power (α² × RX): {comm_only_power:.6e}")
            
            snr_linear = 10 ** (self.comm_snr_db / 10)
            noise_power_thermal = comm_only_power / snr_linear
            
            print(f"    Thermal noise (COMM-referenced): {noise_power_thermal:.6e}")
            print(f"    Formula: (α² × RX) / SNR_linear")
            print(f"           = ({self.comm_power_factor**2:.6f} × {rx_power_before_noise:.6e}) / {snr_linear:.2f}")
            print(f"           = {noise_power_thermal:.6e}")
            
            self.logger.info(f"  COMM-only power: {comm_only_power:.6e}")
            self.logger.info(f"  Thermal noise: {noise_power_thermal:.6e}")
            
            if self.fmcw_integration_mode == 'additive':
                print(f"\n{'─'*80}")
                print(f"[EXPECTED PERFORMANCE - THEORETICAL]")
                print(f"{'─'*80}")
                
                alpha_sq = self.comm_power_factor ** 2
                beta_sq = self.radar_power_factor ** 2
                h_power = np.mean(np.abs(H_comm)**2)
                
                interference_full = beta_sq * h_power
                noise_eff_no_ic = (1/alpha_sq) * (noise_power_thermal + interference_full)
                snr_no_ic_linear = 1.0 / noise_eff_no_ic
                snr_no_ic_db = 10 * np.log10(snr_no_ic_linear)
                
                print(f"\n  WITHOUT IC:")
                print(f"    Full interference (β²|H|²): {interference_full:.6e}")
                print(f"    Effective noise: {noise_eff_no_ic:.6e}")
                print(f"    Expected SNR_eff: {snr_no_ic_db:.2f} dB")
                
                if self.enable_ic and chirp_freq is not None:
                    residual_ratio = 0.1
                    residual_interference = residual_ratio * interference_full
                    noise_eff_with_ic = (1/alpha_sq) * (noise_power_thermal + residual_interference)
                    snr_with_ic_linear = 1.0 / noise_eff_with_ic
                    snr_with_ic_db = 10 * np.log10(snr_with_ic_linear)
                    expected_gain_db = snr_with_ic_db - snr_no_ic_db
                    
                    print(f"\n  WITH IC (expected 90% cancellation):")
                    print(f"    Residual interference: {residual_interference:.6e}")
                    print(f"    Effective noise: {noise_eff_with_ic:.6e}")
                    print(f"    Expected SNR_eff: {snr_with_ic_db:.2f} dB")
                    print(f"    Expected IC gain: {expected_gain_db:.2f} dB ✅")
                    
                    self.logger.info(f"  Expected WITHOUT IC: {snr_no_ic_db:.2f} dB")
                    self.logger.info(f"  Expected WITH IC: {snr_with_ic_db:.2f} dB")
                    self.logger.info(f"  Expected IC gain: {expected_gain_db:.2f} dB")
                
                print(f"{'─'*80}\n")
            
            print(f"  Adding AWGN with COMM-referenced noise...")
            rx_freq_grid_noisy = add_awgn(
                rx_freq_grid, 
                self.comm_snr_db, 
                self.use_gpu, 
                self.logger,
                noise_power_override=noise_power_thermal
            )
            
            rx_power_after_noise = np.mean(np.abs(rx_freq_grid_noisy)**2)
            print(f"    RX power (AFTER noise): {rx_power_after_noise:.6e}")
            self.logger.info(f"  RX power (AFTER noise): {rx_power_after_noise:.6e}")
            
            print(f"\n  Converting to time-domain waveform...")
            N_rx = rx_freq_grid_noisy.shape[0]
            N_symbols = rx_freq_grid_noisy.shape[1]
            N_fft = self.tx_config.n_fft
            N_cp = self.tx_config.cp_length_samples
            N_sc = self.tx_config.n_subcarriers_actual
            
            samples_per_symbol = N_fft + N_cp
            total_samples = samples_per_symbol * N_symbols
            
            print(f"    N_rx: {N_rx}, N_symbols: {N_symbols}")
            print(f"    Total samples: {total_samples}")
            
            rx_waveform = np.zeros((N_rx, total_samples), dtype=complex)
            
            n_left = N_sc // 2
            n_right = N_sc - n_left
            sc_indices = np.concatenate([np.arange(1, n_right + 1), np.arange(-n_left, 0)])
            
            for rx_idx in range(N_rx):
                for sym_idx in range(N_symbols):
                    rx_freq_sym = rx_freq_grid_noisy[rx_idx, sym_idx, :]
                    
                    rx_freq_full = np.zeros(N_fft, dtype=complex)
                    rx_freq_full[sc_indices] = rx_freq_sym
                    
                    rx_time_sym = np.fft.ifft(rx_freq_full) * np.sqrt(N_fft)
                    
                    cp = rx_time_sym[-N_cp:]
                    rx_time_sym_with_cp = np.concatenate([cp, rx_time_sym])
                    
                    start_idx = sym_idx * samples_per_symbol
                    rx_waveform[rx_idx, start_idx:start_idx + samples_per_symbol] = rx_time_sym_with_cp
            
            rx_waveform_power = np.mean(np.abs(rx_waveform)**2)
            print(f"    rx_waveform power: {rx_waveform_power:.6e}")
            
            tx_dict = {
                'data_bits': data_bits,
                'data_symbols': data_symbols,
                'freq_grid': freq_grid_tx,
                'pilot_mask': pilot_mask,
                'pilot_symbols': pilot_symbols,
            }
            
            tx_power_dbm = 43.0
            if 'metadata' in tx_result:
                tx_power_w = tx_result['metadata'].get('actual_power_per_antenna_w', None)
                if tx_power_w is not None:
                    tx_power_dbm = 10 * np.log10(tx_power_w * 1000)
            
            print(f"    TX power: {tx_power_dbm:.1f} dBm")
            
            print(f"\n{'─'*80}")
            print(f"[CALLING COMM RECEIVER v8.3_FINAL WITH DATA IC]")
            print(f"{'─'*80}")
            
            if self.chain.comm_rx is None:
                error_msg = (
                    "❌ COMM RX is None! This means CommunicationReceiver initialization failed.\n"
                    "   SOLUTION: Replace comm_receiver.py with v8.3_FINAL version and restart Python"
                )
                print(error_msg)
                self.logger.error("COMM RX is None - initialization failed")
                raise RuntimeError(error_msg)
            
            print(f"  Parameters:")
            print(f"    tx_power_dbm: {tx_power_dbm:.1f}")
            print(f"    noise_power_actual: {noise_power_thermal:.6e}")
            print(f"    target_snr_db: {self.comm_snr_db}")
            print(f"    chirp_freq: {'PROVIDED' if chirp_freq is not None else 'NOT PROVIDED'}")
            
            if chirp_freq is not None:
                print(f"    chirp_freq shape: {chirp_freq.shape}")
                print(f"    ✅ TWO-STAGE IC WILL BE APPLIED:")
                print(f"       Stage 1: Pilot IC (channel estimation)")
                print(f"       Stage 2: Data-Domain IC (signal cleanup)")
            else:
                print(f"    ℹ️  NO IC (disabled or not applicable)")
            
            self.logger.info(f"  Calling COMM RX with chirp_freq: {chirp_freq is not None}")
            
            comm_result = self.chain.comm_rx.receive_slot(
                y_waveform=rx_waveform,
                tx_dict=tx_dict,
                H_true=H_comm,
                use_oracle_channel=not (self.enable_ic and self.fmcw_integration_mode == 'additive'),
                tx_power_dbm=tx_power_dbm,
                noise_power_actual=noise_power_thermal,
                target_snr_db=self.comm_snr_db,
                chirp_freq=chirp_freq
            )
            
            comm_metrics_all = comm_result['metrics']
            
            print(f"\n{'─'*80}")
            print(f"[COMM RESULTS v8.3_FINAL]")
            print(f"{'─'*80}")
            
            self.logger.info(f"\n✅ COMM Results (v8.3_FINAL):")
            
            ic_applied = comm_metrics_all.get('ic_applied', False)
            ic_gain_db = comm_metrics_all.get('ic_snr_gain_db', 0) if ic_applied else 0
            residual_ratio = comm_metrics_all.get('residual_ratio', 0) if ic_applied else 0
            cancellation_effectiveness = comm_metrics_all.get('cancellation_effectiveness', 0) if ic_applied else 0
            
            data_ic_applied = comm_metrics_all.get('data_domain_ic_applied', False)
            data_ic_effectiveness = comm_metrics_all.get('data_ic_effectiveness', 0) if data_ic_applied else 0
            data_ic_power_reduction_db = comm_metrics_all.get('data_ic_power_reduction_db', 0) if data_ic_applied else 0
            
            if ic_applied:
                print(f"  PILOT IC STATUS: ✅ APPLIED")
                print(f"    Pilot IC gain: {ic_gain_db:.2f} dB")
                print(f"    Residual ratio: {residual_ratio:.2%}")
                print(f"    Cancellation: {cancellation_effectiveness*100:.1f}%")
                self.logger.info(f"  Pilot IC applied: YES, gain: {ic_gain_db:.2f} dB")
            else:
                print(f"  PILOT IC STATUS: NOT APPLIED")
                self.logger.info(f"  Pilot IC applied: NO")
            
            if data_ic_applied:
                print(f"\n  DATA IC STATUS: ✅ APPLIED")
                print(f"    Data IC effectiveness: {data_ic_effectiveness*100:.1f}%")
                print(f"    Power reduction: {data_ic_power_reduction_db:.2f} dB")
                self.logger.info(f"  Data IC applied: YES, effectiveness: {data_ic_effectiveness*100:.1f}%")
            else:
                print(f"\n  DATA IC STATUS: NOT APPLIED")
                self.logger.info(f"  Data IC applied: NO")
            
            if 'snr_antenna_db' in comm_metrics_all:
                snr_antenna = comm_metrics_all['snr_antenna_db']
                print(f"\n  SNR_antenna (pure thermal): {snr_antenna:.2f} dB")
                self.logger.info(f"  SNR_antenna: {snr_antenna:.2f} dB")
            
            if 'sinr_eff_db' in comm_metrics_all:
                sinr_eff = comm_metrics_all['sinr_eff_db']
                print(f"  SINR_eff (post-EQ): {sinr_eff:.2f} dB")
                self.logger.info(f"  SINR_eff: {sinr_eff:.2f} dB")
            elif 'snr_effective_db' in comm_metrics_all:
                snr_eff = comm_metrics_all['snr_effective_db']
                print(f"  SINR_eff (post-EQ): {snr_eff:.2f} dB (legacy key)")
                self.logger.info(f"  SINR_eff: {snr_eff:.2f} dB")
            
            if 'inr_db' in comm_metrics_all and comm_metrics_all['inr_db'] is not None:
                inr = comm_metrics_all['inr_db']
                print(f"  INR: {inr:.2f} dB")
                self.logger.info(f"  INR: {inr:.2f} dB")
            
            ber = comm_result.get('ber', None)
            if ber is not None:
                print(f"  BER: {ber:.6e}")
                self.logger.info(f"  BER: {ber:.6e}")
            
            throughput = comm_metrics_all.get('throughput_mbps', None)
            if throughput is not None:
                print(f"  Throughput: {throughput:.2f} Mbps")
            
            results = {
                'sample_idx': sample_idx,
                'time_index': metadata['time_index'],
                'radar_n_targets': radar_n_targets,
                'radar_gt_closest_range_m': radar_gt.get('range_m', np.nan),
                'radar_gt_closest_velocity_ms': radar_gt.get('velocity_ms', np.nan),
                'radar_gt_closest_power_db': radar_gt.get('power_db', np.nan),
                'fmcw_integration_mode': self.fmcw_integration_mode,
                'comm_power_factor': self.comm_power_factor,
                'radar_power_factor': self.radar_power_factor,
                'comm_power_fraction': self.comm_power_factor ** 2,
                'radar_power_fraction': self.radar_power_factor ** 2,
                'comm_snr_target_db': self.comm_snr_db,
                'radar_snr_target_db': self.radar_snr_db,
                'h_comm_power_original': H_comm_power_original,
                'h_comm_power_normalized': H_comm_power_normalized,
                'h_radar_power_original': H_radar_power_original,
                'h_radar_power_normalized': H_radar_power_normalized,
                'comm_snr_effective_db': comm_metrics_all.get('snr_effective_db', np.nan),
                'comm_sinr_eff_db': comm_metrics_all.get('sinr_eff_db', comm_metrics_all.get('snr_effective_db', np.nan)),
                'comm_inr_db': comm_metrics_all.get('inr_db', np.nan),
                'comm_snr_antenna_db': comm_metrics_all.get('snr_antenna_db', np.nan),
                'comm_snr_antenna_linear': comm_metrics_all.get('snr_antenna_linear', np.nan),
                'comm_rx_power': comm_metrics_all.get('comm_rx_power', np.nan),
                'comm_ber': comm_metrics_all.get('ber', np.nan),
                'comm_throughput_mbps': comm_metrics_all.get('throughput_mbps', np.nan),
                'comm_noise_power_thermal': noise_power_thermal,
                'ic_enabled': self.enable_ic,
                'ic_applied': ic_applied,
                'ic_snr_gain_db': ic_gain_db,
                'residual_ratio': residual_ratio,
                'cancellation_effectiveness': cancellation_effectiveness,
                'data_domain_ic_applied': data_ic_applied,
                'data_ic_effectiveness': data_ic_effectiveness,
                'data_ic_power_reduction_db': data_ic_power_reduction_db,
                
                # ── Advanced communication metrics (from _compute_advanced_comm_metrics) ──
                # 1. Goodput — throughput × (1 - BER)  [Proakis & Salehi §8.2]
                #    Model: goodput = throughput × (1 - BER)  (bit-level, not symbol-level)
                'comm_goodput_mbps':         comm_metrics_all.get('goodput_mbps',         np.nan),
                'comm_goodput_bps':          comm_metrics_all.get('goodput_bps',           np.nan),
                'comm_ber_used_for_goodput': comm_metrics_all.get('ber_used_for_goodput',  np.nan),

                # 2. Data Rate — Shannon bound and practical rate
                #    Shannon: C = B·log2(1+SINR)  [Shannon 1948]
                #    Practical: n_data_res × bits_per_symbol / slot_duration
                'comm_shannon_capacity_mbps':        comm_metrics_all.get('shannon_capacity_mbps',        np.nan),
                'comm_shannon_capacity_bps':         comm_metrics_all.get('shannon_capacity_bps',         np.nan),
                'comm_practical_data_rate_mbps':     comm_metrics_all.get('practical_data_rate_mbps',     np.nan),
                'comm_practical_data_rate_bps':      comm_metrics_all.get('practical_data_rate_bps',      np.nan),

                # 3. Spectral Efficiency — bits/s/Hz
                #    Shannon bound: η = log2(1+SINR)
                #    Practical: practical_data_rate / bandwidth
                'comm_spectral_efficiency_shannon_bps_hz':  comm_metrics_all.get('spectral_efficiency_shannon_bps_hz',  np.nan),
                'comm_spectral_efficiency_practical_bps_hz': comm_metrics_all.get('spectral_efficiency_practical_bps_hz', np.nan),
                'comm_bandwidth_hz': comm_metrics_all.get('bandwidth_hz', np.nan),

                # 4. Ergodic Capacity — single-realization lower bound
                #    Full ergodic: E_H[log2(1+SINR(H))]; here approximated at one point
                #    Reference: Tse & Viswanath "Fundamentals of Wireless Communication" §4.4
                'comm_ergodic_capacity_bps_hz':        comm_metrics_all.get('ergodic_capacity_bps_hz', np.nan),
                'comm_ergodic_capacity_is_lower_bound': True,   # single-realization ≠ full ergodic average
                'comm_energy_efficiency_model':         'normalised',  # P_tx = α², not physical watts

                # 5. Outage Probability — P(SINR < SINR_threshold)
                #    Deterministic at single operating point → 0 or 1
                #    Three thresholds:
                #      (a) QPSK minimum viability floor: 6.8 dB (BER=10⁻³)
                #      (b) Practical QoS floor:          9.6 dB (BER=10⁻⁴)
                #      (c) Design target SNR:            20.0 dB (ISAC tradeoff indicator)
                #    Reference: Goldsmith "Wireless Communications" §4.2
                'comm_outage_prob_qpsk':          comm_metrics_all.get('outage_probability_qpsk',          np.nan),
                'comm_outage_prob_qos':           comm_metrics_all.get('outage_probability_qos',           np.nan),
                'comm_outage_prob_design_target': comm_metrics_all.get('outage_probability_design_target', np.nan),
                'comm_sinr_threshold_qpsk_db':    comm_metrics_all.get('sinr_threshold_qpsk_db',          np.nan),
                'comm_sinr_threshold_qos_db':     comm_metrics_all.get('sinr_threshold_qos_db',           np.nan),
                'comm_sinr_threshold_design_db':  comm_metrics_all.get('sinr_threshold_design_db',        np.nan),

                # 6. Outage Capacity — ε-capacity at ε=10% outage threshold
                #    C_ε = C if not in outage, 0 if in outage
                #    Reference: Biglieri et al. IEEE Trans. IT 1998
                'comm_outage_capacity_bps_hz': comm_metrics_all.get('outage_capacity_bps_hz', np.nan),
                'comm_outage_capacity_mbps':   comm_metrics_all.get('outage_capacity_mbps',   np.nan),
                'comm_epsilon_outage':         comm_metrics_all.get('epsilon_outage',          np.nan),

                # 7. Energy Efficiency — bits/Joule
                #    η_EE = Goodput / P_tx  where P_tx = α² (normalised, not physical watts)
                #    Reference: Bjornson et al. "Optimal Resource Allocation" §1.4
                'comm_energy_efficiency_bits_per_joule':   comm_metrics_all.get('energy_efficiency_bits_per_joule',   np.nan),
                'comm_energy_efficiency_mbits_per_joule':  comm_metrics_all.get('energy_efficiency_mbits_per_joule',  np.nan),
                'comm_p_tx_normalised':                    comm_metrics_all.get('p_tx_normalised',                    np.nan),
                'comm_energy_efficiency_model':            comm_metrics_all.get('energy_efficiency_model',            'normalised'),
                'comm_ergodic_capacity_is_lower_bound':    comm_metrics_all.get('ergodic_capacity_is_lower_bound',    True),
                
                'radar_snr_db': sensing_result.get('snr_db', np.nan),
                'radar_n_detections': sensing_result.get('n_detections', 0),
                'radar_detection_matched': sensing_result.get('detection_matched', False),
                'radar_range_error_m':    sensing_result.get('range_error_m',    np.nan),
                'radar_range_error_median_m':    sensing_result.get('range_error_median_m',    np.nan),
                'radar_velocity_error_ms': sensing_result.get('velocity_error_ms', np.nan),
                'radar_velocity_error_median_ms':    sensing_result.get('velocity_error_median_ms',    np.nan),
                'radar_angle_error_deg':  sensing_result.get('angle_error_deg',  np.nan),
                'radar_angle_rmse_deg':   sensing_result.get('angle_rmse_deg',   np.nan),
                'n_angle_invisible':    sensing_result.get('n_angle_invisible',    np.nan),
                'n_gt_fov_visible':     sensing_result.get('n_gt_fov_visible',     np.nan),
                'n_gt_fov_invisible':   sensing_result.get('n_gt_fov_invisible',   np.nan),
                'n_gt_total_scene':     sensing_result.get('n_total_gt',           np.nan),
                # ── CRB / SNR / SCNR — new names with legacy fallbacks ────────
                'radar_snr_rdm_db': sensing_result.get('snr_db', np.nan),
                'radar_snr_input_db': (
                    sensing_result['targets'][0].get(
                        'radar_snr_input_db',
                        sensing_result['targets'][0].get('snr_input_db', np.nan))
                    if len(sensing_result.get('targets', [])) > 0 else np.nan),
                'radar_snr_input_source': (
                    sensing_result['targets'][0].get(
                        'radar_snr_input_source',
                        sensing_result['targets'][0].get('snr_input_source', 'unknown'))
                    if len(sensing_result.get('targets', [])) > 0 else 'no_targets'),
                'radar_crb_range_m': (
                    sensing_result['targets'][0].get('crb_range_m', np.nan)
                    if len(sensing_result.get('targets', [])) > 0 else np.nan),
                'radar_crb_velocity_ms': (
                    sensing_result['targets'][0].get('crb_velocity_ms', np.nan)
                    if len(sensing_result.get('targets', [])) > 0 else np.nan),
                'radar_crb_angle_deg': (
                    sensing_result['targets'][0].get('crb_angle_deg', np.nan)
                    if len(sensing_result.get('targets', [])) > 0 else np.nan),
                'radar_range_efficiency': (
                    abs(sensing_result.get('range_error_m', np.nan)) /
                    (sensing_result['targets'][0].get('crb_range_m', np.nan) + 1e-12)
                    if len(sensing_result.get('targets', [])) > 0
                    and sensing_result.get('range_error_m') is not None
                    and not (sensing_result.get('range_error_m') != sensing_result.get('range_error_m'))
                    and sensing_result['targets'][0].get('crb_range_m', 0) > 1e-9
                    else np.nan),
                'radar_velocity_efficiency': (
                    abs(sensing_result.get('velocity_error_ms', np.nan)) /
                    (sensing_result['targets'][0].get('crb_velocity_ms', np.nan) + 1e-12)
                    if len(sensing_result.get('targets', [])) > 0
                    and sensing_result.get('velocity_error_ms') is not None
                    and not (sensing_result.get('velocity_error_ms') != sensing_result.get('velocity_error_ms'))
                    and sensing_result['targets'][0].get('crb_velocity_ms', 0) > 1e-12
                    else np.nan),
                'radar_scnr_db': sensing_result.get('scnr_db', np.nan),
                # ── thermal / NLMS ──────────────────────────────────────────
                'ofdm_mitigation_enabled': tx_ofdm_td is not None,
                'radar_thermal_noise_power': radar_thermal_noise_power,
                'radar_thermal_noise_db':  float(10 * np.log10(radar_thermal_noise_power + 1e-12)),
                'radar_mode':              getattr(self.chain.sens_rx.config,
                                                   'radar_mode', 'monostatic'),
                'nlms_filter_length': self.chain.sens_rx.nlms_filter_length if hasattr(self.chain.sens_rx, 'nlms_filter_length') else np.nan,
                'nlms_step_size': self.chain.sens_rx.nlms_step_size if hasattr(self.chain.sens_rx, 'nlms_step_size') else np.nan,
                'processing_success': True,
            }
            
            if len(sensing_result.get('targets', [])) > 0:
                top_target = sensing_result['targets'][0]
                results['radar_top_range_m'] = top_target.get('range_m', np.nan)
                results['radar_top_velocity_ms'] = top_target.get('velocity_ms', np.nan)
                if 'angle_deg' in top_target and top_target['angle_deg'] is not None:
                    results['radar_top_angle_deg'] = top_target['angle_deg']
            
            print(f"\n{'='*80}")
            print(f"[SAMPLE {sample_idx} COMPLETE - v8.5]")
            if ic_applied or data_ic_applied:
                print(f"  ✅ IC APPLIED")
                if ic_applied:
                    print(f"    Pilot IC gain: {ic_gain_db:.2f} dB")
                if data_ic_applied:
                    print(f"    Data IC effectiveness: {data_ic_effectiveness*100:.1f}%")
            if sensing_result.get('detection_matched'):
                print(f"  ✅ RADAR DETECTION: MATCHED")
                if 'radar_range_error_m' in results and not np.isnan(results['radar_range_error_m']):
                    print(f"    Range error: {abs(results['radar_range_error_m']):.3f} m")
            print(f"\n  RADAR SNR/CRB/SCNR SUMMARY:")
            _rdm   = results.get('radar_snr_rdm_db',   results.get('radar_snr_db',    float('nan')))
            _input = results.get('radar_snr_input_db', float('nan'))
            _crb_r = results.get('radar_crb_range_m',  float('nan'))
            _crb_v = results.get('radar_crb_velocity_ms', float('nan'))
            print(f"    radar_snr_rdm   (post-processing): {_rdm:.2f} dB  ← CFAR/ranking")
            print(f"    radar_snr_input (pre-processing):  {_input:.2f} dB  ← CRB reference")
            print(f"    CRB_range:    {_crb_r:.6f} m")
            print(f"    CRB_velocity: {_crb_v:.8f} m/s")
            _scnr = results.get('radar_scnr_db')
            if _scnr is not None and _scnr == _scnr:   # not None, not NaN
                print(f"    SCNR:         {_scnr:.2f} dB")
            _eff_r = results.get('radar_range_efficiency')
            if _eff_r is not None and _eff_r == _eff_r:
                status = ("✅" if _eff_r < 5 else "⚠️ " if _eff_r < 20 else "❌")
                print(f"    Range eff:    {_eff_r:.1f}× CRB  {status}")
            _eff_v = results.get('radar_velocity_efficiency')
            if _eff_v is not None and _eff_v == _eff_v:
                print(f"    Velocity eff: {_eff_v:.1f}× CRB")
                
            print(f"{'='*80}\n")
            
            # ── Explicit memory release before returning ───────────────────
            # Release large intermediate arrays so Python GC can reclaim RAM
            # before the next sample is loaded.
            del rx_radar_signal, rx_radar_signal_noisy
            if tx_ofdm_td is not None:
                del tx_ofdm_td
            if 'rx_ofdm' in dir():
                del rx_ofdm
            del rx_freq_grid, rx_freq_grid_noisy, rx_waveform
            del H_radar, H_comm
            del sensing_result
            import gc
            gc.collect()
            # ─────────────────────────────────────────────────────────────
            
            return results
            
        except Exception as e:
            print(f"\n❌ ERROR processing sample {sample_idx}: {e}")
            print(traceback.format_exc())
            self.logger.error(f"Error: {e}")
            return {'sample_idx': sample_idx, 'processing_success': False, 'error_message': str(e)}

def save_results_to_csv(results_list: List[Dict], output_dir: Path, prefix: str, logger=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results_list)
    
    per_sample_file = output_dir / f'{prefix}_metrics.csv'
    df.to_csv(per_sample_file, index=False)
    
    if logger:
        logger.info(f"✓ Saved: {per_sample_file}")
    
    print(f"\n✓ Saved CSV: {per_sample_file}")
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    
    return df

def main():
    import gc

    print("\n" + "="*80)
    print("ISAC v8.5 - CORRECTED RADAR + ACTUAL TX WAVEFORM")
    print("="*80)
    print(f"  Radar mode:       {RADAR_MODE.upper()}")
    print(f"  HDF5:             {HDF5_FILE.name}")
    print(f"  Output:           {OUTPUT_BASE}")
    _rx_label_main = ('BS 2×2 UPA (monostatic)' if RADAR_MODE == 'monostatic'
                      else 'BS 2×2 UPA (bistatic)' if RADAR_MODE == 'bistatic_bs'
                      else 'UE 2×1 ULA (bistatic)')
    print(f"  RX antennas:      {N_RADAR_RX_ANTENNAS}  ({_rx_label_main})")
    print(f"  Delay factor:     {DELAY_FACTOR:.0f}×  "
          f"({'round-trip' if RADAR_MODE == 'monostatic' else 'one-way'})")
    print(f"  Power allocation: α²={COMM_POWER_FACTOR**2:.4f}, β²={RADAR_POWER_FACTOR**2:.4f}")
    print(f"  COMM SNR target:  {COMM_SNR_DB} dB")
    print(f"  RADAR SNR target: {RADAR_SNR_DB} dB")
    print(f"  Max samples:      {MAX_SAMPLES}")
    print(f"  Batch size:       {BATCH_SIZE}")
    print(f"  Angle estimation: {ENABLE_ANGLE_ESTIMATION}")
    print(f"  IC enabled:       {ENABLE_INTERFERENCE_CANCELLATION}")
    print(f"  OFDM mitigation:  {ENABLE_OFDM_MITIGATION}")
    print("="*80 + "\n")

    main_logger = setup_logging(OUTPUT_BASE, 'isac_v8_5_actual_tx')

    try:
        print("Initializing ISAC chain...")
        tx_config = get_default_tx_config()
        rx_config = get_default_rx_config(use_gpu=USE_GPU)

        rx_config.enable_ofdm_mitigation   = ENABLE_OFDM_MITIGATION
        rx_config.nlms_filter_length       = NLMS_FILTER_LENGTH
        rx_config.nlms_step_size           = NLMS_STEP_SIZE
        rx_config.enable_mti              = False
        rx_config.mti_velocity_threshold  = DYNAMIC_VELOCITY_THRESHOLD
        # ── Bistatic / monostatic mode ────────────────────────────────────
        rx_config.radar_mode              = _BISTATIC_MODE
        rx_config.n_radar_rx_antennas     = N_RADAR_RX_ANTENNAS
        rx_config.radar_rx_antenna_shape  = ((2, 2) if N_RADAR_RX_ANTENNAS == 4
                                             else (2, 1))

        isac_chain = ISACChain(
            tx_config=tx_config,
            rx_config=rx_config,
            use_gpu=USE_GPU,
            verbose=True,
            log_dir=str(OUTPUT_BASE),
            fmcw_integration_mode=FMCW_INTEGRATION_MODE,
            comm_power_factor=COMM_POWER_FACTOR,
            radar_power_factor=RADAR_POWER_FACTOR,
            enable_interference_cancellation=ENABLE_INTERFERENCE_CANCELLATION,
            ic_iterations=IC_ITERATIONS
        )
        print("✓ ISACChain initialized\n")
        main_logger.info("✓ ISACChain initialized")

    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        main_logger.error(f"Failed: {e}")
        return 1

    processor = ISACChainRealChannelProcessor(
        isac_chain,
        comm_snr_db=COMM_SNR_DB,
        radar_snr_db=RADAR_SNR_DB,
        enable_ic=ENABLE_INTERFERENCE_CANCELLATION,
        enable_angle_estimation=ENABLE_ANGLE_ESTIMATION,
        logger=main_logger
    )

    batch_files     = []
    total_processed = 0
    total_failed    = 0

    try:
        with DeepVerse6GLoader(HDF5_FILE, logger=main_logger) as loader:
            n_to_process = (
                min(MAX_SAMPLES, loader.n_samples) if MAX_SAMPLES
                else loader.n_samples
            )
            n_batches = (n_to_process + BATCH_SIZE - 1) // BATCH_SIZE

            print(f"Processing {n_to_process} samples in "
                  f"{n_batches} batches of ≤{BATCH_SIZE}...\n")
            main_logger.info(
                f"Batch processing: {n_to_process} samples, "
                f"batch_size={BATCH_SIZE}, n_batches={n_batches}"
            )

            overall_bar = tqdm(
                total=n_to_process,
                desc="Overall",
                unit="sample",
                position=0,
                leave=True,
                colour="green"
            )

            for batch_idx in range(n_batches):
                batch_start = batch_idx * BATCH_SIZE
                batch_end   = min(batch_start + BATCH_SIZE, n_to_process)

                print(f"\n{'='*80}")
                print(f"BATCH {batch_idx+1}/{n_batches}  "
                      f"[samples {batch_start}–{batch_end-1}]")
                print(f"{'='*80}\n")
                main_logger.info(
                    f"Batch {batch_idx+1}/{n_batches} "
                    f"(samples {batch_start}–{batch_end-1})"
                )

                batch_results = []

                batch_bar = tqdm(
                    range(batch_start, batch_end),
                    desc=f"  Batch {batch_idx+1:03d}",
                    unit="sample",
                    position=1,
                    leave=False,
                    colour="cyan"
                )

                for sample_idx in batch_bar:
                    batch_bar.set_postfix({
                        "idx": sample_idx,
                        "ok":  total_processed,
                        "err": total_failed
                    })
                    try:
                        sample_data = loader.load_sample(sample_idx)
                        result      = processor.process_real_sample(sample_data)
                        batch_results.append(result)
                        total_processed += 1
                        del sample_data
                        gc.collect()

                    except Exception as e:
                        total_failed += 1
                        print(f"\n❌ Sample {sample_idx} failed: {e}")
                        main_logger.error(f"Sample {sample_idx} failed: {e}")
                        batch_results.append({
                            'sample_idx':         sample_idx,
                            'processing_success': False,
                            'error_message':      str(e)
                        })

                    overall_bar.update(1)

                batch_bar.close()

                if batch_results:
                    batch_prefix = f'isac_v8_5_actual_tx_batch{batch_idx:04d}'
                    save_results_to_csv(
                        batch_results, OUTPUT_BASE, batch_prefix, main_logger
                    )
                    batch_file = OUTPUT_BASE / f'{batch_prefix}_metrics.csv'
                    batch_files.append(batch_file)
                    print(f"\n  ✅ Batch {batch_idx+1} saved → {batch_file.name} "
                          f"({len(batch_results)} rows)")
                    main_logger.info(
                        f"Batch {batch_idx+1} saved: "
                        f"{len(batch_results)} rows → {batch_file}"
                    )

                del batch_results
                gc.collect()
                print(f"  [Memory released after batch {batch_idx+1}]")

            overall_bar.close()

    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        main_logger.error(f"Critical error: {e}")
        if not batch_files:
            return 1

    # ── Consolidate all batch CSVs ────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"CONSOLIDATING {len(batch_files)} BATCH FILE(S)")
    print(f"{'='*80}")
    main_logger.info(f"Consolidating {len(batch_files)} batch files")

    df = pd.DataFrame()

    if batch_files:
        valid_dfs = []
        for bf in tqdm(batch_files, desc="Merging batches", unit="file"):
            if bf.exists():
                try:
                    valid_dfs.append(pd.read_csv(bf))
                except Exception as e:
                    print(f"  ⚠️  Could not read {bf.name}: {e}")
                    main_logger.warning(f"Could not read {bf}: {e}")

        if valid_dfs:
            df          = pd.concat(valid_dfs, ignore_index=True)
            final_csv   = OUTPUT_BASE / 'isac_v8_5_actual_tx_metrics.csv'
            df.to_csv(final_csv, index=False)
            print(f"\n✓ Final CSV: {final_csv}")
            print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
            main_logger.info(
                f"Final CSV: {final_csv} "
                f"({len(df)} rows, {len(df.columns)} columns)"
            )
            for bf in batch_files:
                try:
                    bf.unlink()
                except Exception:
                    pass
            print("  Batch files cleaned up.")
        else:
            print("  ❌ No valid batch files to consolidate.")
    else:
        print("  ❌ No batch files produced.")

    print(f"\n  Total attempted:  {total_processed + total_failed}")
    print(f"  Processed OK:     {total_processed}")
    print(f"  Failed:           {total_failed}")

    # ── Final summary statistics ──────────────────────────────────────────
    if len(df) == 0:
        print("\n  No data to summarise.")
        return 0

    df_success = df[df['processing_success'] == True]
    if len(df_success) == 0:
        print("\n  No successful samples to summarise.")
        return 0

    sinr_col = ('comm_sinr_eff_db'
                if 'comm_sinr_eff_db' in df_success.columns
                else 'comm_snr_effective_db')

    print(f"\n{'='*80}")
    print("[FINAL RESULTS SUMMARY]")
    print(f"{'='*80}")
    print(f"  Successful samples: {len(df_success)}")

    if 'comm_snr_antenna_db' in df_success.columns:
        m = df_success['comm_snr_antenna_db'].mean()
        s = df_success['comm_snr_antenna_db'].std()
        print(f"  SNR_antenna (thermal): {m:.2f} ± {s:.2f} dB")

    if sinr_col in df_success.columns:
        m = df_success[sinr_col].mean()
        s = df_success[sinr_col].std()
        print(f"  SINR_eff (post-EQ):    {m:.2f} ± {s:.2f} dB")

    if 'comm_ber' in df_success.columns:
        m = df_success['comm_ber'].mean()
        s = df_success['comm_ber'].std()
        print(f"  BER:                   {m:.3e} ± {s:.3e}")

    df_ic = df_success[df_success.get('ic_applied', pd.Series(False)) == True] \
            if 'ic_applied' in df_success.columns else pd.DataFrame()
    if len(df_ic) > 0:
        print(f"\n  PILOT IC:")
        print(f"    Samples:     {len(df_ic)}")
        print(f"    Mean gain:   {df_ic['ic_snr_gain_db'].mean():.2f} ± "
              f"{df_ic['ic_snr_gain_db'].std():.2f} dB")
        print(f"    Cancellation:{df_ic['cancellation_effectiveness'].mean()*100:.1f}%")

    df_data_ic = df_success[df_success.get('data_domain_ic_applied', pd.Series(False)) == True] \
                 if 'data_domain_ic_applied' in df_success.columns else pd.DataFrame()
    if len(df_data_ic) > 0:
        print(f"\n  DATA-DOMAIN IC:")
        print(f"    Samples:        {len(df_data_ic)}")
        print(f"    Mean eff.:      "
              f"{df_data_ic['data_ic_effectiveness'].mean()*100:.1f}% ± "
              f"{df_data_ic['data_ic_effectiveness'].std()*100:.1f}%")

    if 'radar_detection_matched' in df_success.columns:
        det_rate = df_success['radar_detection_matched'].mean()
        print(f"\n  RADAR:")
        print(f"    Detection rate: {det_rate*100:.1f}%")

        df_det = df_success[df_success['radar_detection_matched'] == True]
        if len(df_det) > 0:
            for col, label in [
                ('radar_range_error_m',    'Range error'),
                ('radar_velocity_error_ms','Velocity error'),
            ]:
                if col in df_det.columns:
                    m = df_det[col].abs().mean()
                    s = df_det[col].abs().std()
                    unit = 'm' if 'range' in col else 'm/s'
                    print(f"    {label}:   {m:.4f} ± {s:.4f} {unit}")

            if 'radar_angle_error_deg' in df_det.columns:
                valid = df_det['radar_angle_error_deg'].dropna()
                if len(valid) > 0:
                    print(f"    Angle error:    {valid.mean():.3f} ± {valid.std():.3f} deg")

    if 'radar_snr_input_db' in df_success.columns:
        rdm_col = ('radar_snr_rdm_db'
                   if 'radar_snr_rdm_db' in df_success.columns
                   else 'radar_snr_db')
        print(f"\n  SNR / CRB / SCNR:")
        print(f"    radar_snr_rdm:    "
              f"{df_success[rdm_col].mean():.2f} ± "
              f"{df_success[rdm_col].std():.2f} dB")
        print(f"    radar_snr_input:  "
              f"{df_success['radar_snr_input_db'].mean():.2f} ± "
              f"{df_success['radar_snr_input_db'].std():.2f} dB  "
              f"(target {RADAR_SNR_DB:.0f} dB)")

        for col, label in [
            ('radar_crb_range_m',    'CRB_range'),
            ('radar_crb_velocity_ms','CRB_velocity'),
        ]:
            if col in df_success.columns:
                v = df_success[col].dropna()
                if len(v) > 0:
                    unit = 'm' if 'range' in col else 'm/s'
                    print(f"    {label}:     {v.mean():.6f} ± {v.std():.6f} {unit}")

        for col, label in [
            ('radar_range_efficiency',   'Range eff.'),
            ('radar_velocity_efficiency','Velocity eff.'),
        ]:
            if col in df_success.columns:
                v = df_success[col].dropna()
                if len(v) > 0:
                    status = ("✅" if v.mean() < 5 else "⚠️ " if v.mean() < 20 else "❌")
                    print(f"    {label}:   {v.mean():.1f}× CRB  {status}")

        if 'radar_scnr_db' in df_success.columns:
            v = df_success['radar_scnr_db'].dropna()
            if len(v) > 0:
                print(f"    SCNR:             {v.mean():.2f} ± {v.std():.2f} dB")

    if 'ofdm_mitigation_enabled' in df_success.columns:
        df_mit    = df_success[df_success['ofdm_mitigation_enabled'] == True]
        df_no_mit = df_success[df_success['ofdm_mitigation_enabled'] == False]
        if len(df_mit) > 0:
            print(f"\n  OFDM MITIGATION (NLMS):")
            print(f"    Samples with NLMS: {len(df_mit)}")
            dr_mit = df_mit['radar_detection_matched'].mean() * 100
            print(f"    Detection rate (with): {dr_mit:.1f}%")
            if len(df_no_mit) > 0:
                dr_no = df_no_mit['radar_detection_matched'].mean() * 100
                print(f"    Detection rate (without): {dr_no:.1f}%")
                print(f"    Improvement: {dr_mit - dr_no:+.1f} pp")

    print(f"\n{'='*80}")
    print("✅ PROCESSING COMPLETE!")
    print(f"{'='*80}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())