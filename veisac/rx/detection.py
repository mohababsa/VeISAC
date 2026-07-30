# detection.py
"""
VeISAC — CFAR Target Detection

2D OS/CA/GO/SO-CFAR detection on the range-Doppler power map, with peak extraction and detection statistics for the Sen-RX chain.

Paper: "VeISAC: An End-to-End MIMO-OFDM-FMCW Framework for ISAC
        in 6G Vehicular Networks"
Authors: M. Ababsa, S. Ribouh, Y. El Hillali, A. Rivenq
"""

import numpy as np
from typing import Literal, Tuple, List, Optional, Union
import warnings

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np


class CFARDetector:
    
    def __init__(
        self,
        method: Literal['ca', 'os', 'go', 'so'] = 'os',
        guard_cells: int = 8,
        training_cells: int = 24,
        pfa: float = 1e-5,
        os_index: Optional[int] = None,
        enable_2d: bool = True,
        use_gpu: bool = True,
        verbose: bool = True,
        min_training_cells: int = 8
    ):
        self.method = method.lower()
        self.guard_cells = guard_cells
        self.training_cells = training_cells
        self.pfa = pfa
        self.enable_2d = enable_2d
        self.verbose = True
        self.min_training_cells = min_training_cells
        
        if guard_cells < 0:
            raise ValueError(f"guard_cells must be >= 0, got {guard_cells}")
        if training_cells < 1:
            raise ValueError(f"training_cells must be >= 1, got {training_cells}")
        if not (0 < pfa < 1):
            raise ValueError(f"pfa must be in (0, 1), got {pfa}")
        if self.method not in ['ca', 'os', 'go', 'so']:
            raise ValueError(f"Unknown CFAR method: {method}")
        
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        if os_index is None:
            total_train_cells = 2 * training_cells
            
            # ═══════════════════════════════════════════════════════════════════
            # OS-CFAR k-INDEX SELECTION
            # ═══════════════════════════════════════════════════════════════════
            # Theory: OS-CFAR picks the k-th smallest value from N sorted training cells
            #   - k too LOW (e.g., 25th percentile) → underestimates noise → too many detections
            #   - k too HIGH (e.g., 90th percentile) → overestimates noise → miss weak targets
            #   - OPTIMAL: 50-75th percentile depending on clutter characteristics
            #
            # Recommended values:
            #   - Clean environment (low clutter): k ≈ 0.50N (median)
            #   - Moderate clutter: k ≈ 0.60N to 0.70N
            #   - Heavy clutter: k ≈ 0.75N
            #
            # For Pfa=1e-5 with 48 training cells → k=24 (50th percentile) is standard
            # ═══════════════════════════════════════════════════════════════════
            
            # Use 60th percentile (good balance for multi-target scenes)
            percentile = 0.6
            self.os_index = max(1, int(percentile * total_train_cells))
            
            if self.verbose:
                print(f"  [OS-CFAR] Auto k-index: {self.os_index}/{total_train_cells} "
                    f"({percentile*100:.0f}th percentile)")
        else:
            self.os_index = os_index
        
        self.threshold_multiplier = self._compute_threshold_multiplier()
        self.window_size = 2 * (guard_cells + training_cells) + 1
        
        if self.verbose:
            gpu_status = "GPU (CuPy)" if self.use_gpu else "CPU (NumPy)"
            print(f"\n{'='*80}")
            print(f"[CFAR DETECTOR v4.3 - CORRECTED & PRODUCTION-READY]")
            print(f"{'='*80}")
            print(f"  Method: {self.method.upper()}")
            print(f"  Device: {gpu_status}")
            print(f"  Guard cells: {guard_cells} (each side)")
            print(f"  Training cells: {training_cells} (each side)")
            print(f"  Total window: {self.window_size} × {self.window_size}")
            print(f"  Pfa: {pfa:.1e}")
            print(f"  Threshold multiplier (α): {self.threshold_multiplier:.6f}")
            print(f"  2D CFAR: {'Enabled' if enable_2d else 'Disabled'}")
            print(f"  Min training cells: {min_training_cells}")
            if self.method == 'os':
                print(f"  OS index (k): {self.os_index}")
            print(f"{'='*80}\n")
    
    def _compute_threshold_multiplier(self) -> float:
        if self.method == 'ca':
            N = 2 * self.training_cells
            alpha = self.pfa ** (-1.0 / N) if N > 0 else 1.0
        elif self.method == 'os':
            N = 2 * self.training_cells
            k = min(self.os_index, N - 1)
            alpha = self.pfa ** (-1.0 / (N - k + 1)) if (N - k + 1) > 0 else 1.0
        elif self.method in ['go', 'so']:
            N = self.training_cells
            alpha = self.pfa ** (-1.0 / N) if N > 0 else 1.0
        else:
            alpha = 1.0
        
        return alpha
    
    def detect(
        self,
        range_doppler_map: Union[np.ndarray, 'cp.ndarray'],
        axis: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        xp = self.xp
        
        if self.use_gpu and isinstance(range_doppler_map, np.ndarray):
            rdm = cp.asarray(range_doppler_map)
        elif not self.use_gpu and hasattr(range_doppler_map, 'get'):
            rdm = range_doppler_map.get()
        else:
            rdm = range_doppler_map
        
        if xp.iscomplexobj(rdm):
            power_map = xp.abs(rdm) ** 2
            if self.verbose:
                print(f"[CFAR] Converted complex RDM to power: |.|²")
        else:
            power_map = rdm.astype(xp.float64, copy=True)
        
        n_range, n_doppler = power_map.shape
        
        if self.verbose:
            print(f"\n{'─'*80}")
            print(f"[CFAR DETECTION STARTED]")
            print(f"{'─'*80}")
            print(f"  Input RDM shape: ({n_range}, {n_doppler})")
            print(f"  Method: {self.method.upper()}")
            print(f"  Detection mode: {'2D' if (axis is None and self.enable_2d) else '1D'}")
            print(f"  Total cells: {n_range * n_doppler:,}")
        
        if self.window_size > n_range:
            warnings.warn(
                f"CFAR window ({self.window_size}) exceeds range dimension ({n_range})"
            )
        
        if self.enable_2d and self.window_size > n_doppler:
            warnings.warn(
                f"CFAR window ({self.window_size}) exceeds Doppler dimension ({n_doppler})"
            )
        
        import time
        start_time = time.time()
        
        if self.method == 'ca':
            if self.enable_2d and axis is None:
                detections, threshold_map = self._detect_ca_cfar_2d(power_map)
            else:
                detections, threshold_map = self._detect_ca_cfar_1d(power_map, axis)
        elif self.method == 'os':
            detections, threshold_map = self._detect_os_cfar_2d(power_map)
        elif self.method == 'go':
            detections, threshold_map = self._detect_go_cfar_2d(power_map)
        elif self.method == 'so':
            detections, threshold_map = self._detect_so_cfar_2d(power_map)
        else:
            raise ValueError(f"Unknown CFAR method: {self.method}")
        
        elapsed_time = time.time() - start_time
        
        if self.use_gpu:
            if isinstance(detections, cp.ndarray):
                detections = cp.asnumpy(detections)
            if isinstance(threshold_map, cp.ndarray):
                threshold_map = cp.asnumpy(threshold_map)
        
        n_det = int(detections.sum())
        det_rate = 100 * n_det / detections.size

        if self.verbose:
            print(f"\n  RESULTS:")
            print(f"    Processing time: {elapsed_time*1000:.2f} ms")
            print(f"    Total detections: {n_det}")
            print(f"    Detection rate: {det_rate:.4f}%")
            if n_det > 0:
                det_powers = threshold_map[detections]
                print(f"    Threshold range: {det_powers.min():.2e} to {det_powers.max():.2e}")
    
            # Phase 1: Multi-target awareness
            if n_det > 100:
                print(f"    ⚠️  Dense multi-target scene detected ({n_det} peaks)")
                print(f"    Note: extract_detections() will limit to top {40} by default")
            elif n_det == 0:
                print(f"    ⚠️  No detections! Check CFAR threshold or input signal")
                
        # ═══════════════════════════════════════════════════════════════════
        # DIAGNOSTIC: CFAR DETECTION OUTPUT - WHAT PEAKS DID CFAR DETECT?
        # ═══════════════════════════════════════════════════════════════════
        if self.verbose and n_det > 0:
            print(f"\n{'='*80}")
            print(f"[CFAR DETECTION OUTPUT DIAGNOSTIC]")
            print(f"{'='*80}")
            
            # Find the strongest detected peak
            detected_power_map = power_map * detections
            max_detected_power = float(xp.max(detected_power_map))
            
            if max_detected_power > 0:
                strongest_det_idx = xp.unravel_index(
                    xp.argmax(detected_power_map), 
                    detected_power_map.shape
                )
                strongest_range_bin = int(strongest_det_idx[0])
                strongest_doppler_bin = int(strongest_det_idx[1])
                
                print(f"  Total detections: {n_det}")
                print(f"  ")
                print(f"  STRONGEST DETECTED PEAK:")
                print(f"    Range bin: {strongest_range_bin}")
                print(f"    Doppler bin: {strongest_doppler_bin}")
                print(f"    Power: {max_detected_power:.6e}")
                print(f"  ")
                
                # Check if bin 177 was detected
                if detections.shape[0] > 177:
                    _dc_bin = detections.shape[1] // 2   # DC bin = N_doppler / 2
                    bin_177_detected = bool(detections[177, _dc_bin])
                    if bin_177_detected:
                        bin_177_power = float(power_map[177, _dc_bin])
                        print(f"  BIN 177 CHECK:")
                        print(f"    Detected: ✅ YES")
                        print(f"    Power: {bin_177_power:.6e}")
                        
                        if strongest_range_bin == 177:
                            print(f"    Status: ✅ Bin 177 IS the strongest detected peak")
                        else:
                            print(f"    Status: ⚠️  Bin 177 detected but NOT strongest")
                            print(f"    Strongest is bin {strongest_range_bin} with {max_detected_power:.6e}")
                            print(f"    Bin 177 has {bin_177_power:.6e}")
                            print(f"    Power ratio: {bin_177_power/max_detected_power:.2f}×")
                    else:
                        print(f"  BIN 177 CHECK:")
                        print(f"    Detected: ❌ NO")
                        print(f"    Threshold at 177: "
                              f"{float(threshold_map[177, _dc_bin]):.6e}")
                        print(f"    Power at 177: "
                              f"{float(power_map[177, _dc_bin]):.6e}")
                        print(f"    Status: ❌ CRITICAL - Bin 177 was NOT detected by CFAR!")
                
                # Show top 5 detected peaks
                print(f"  ")
                print(f"  TOP 5 DETECTED PEAKS:")
                det_indices = xp.argwhere(detections)
                det_powers_list = []
                for idx in det_indices:
                    r, d = int(idx[0]), int(idx[1])
                    p = float(power_map[r, d])
                    det_powers_list.append((r, d, p))
                
                det_powers_list.sort(key=lambda x: x[2], reverse=True)
                for i, (r, d, p) in enumerate(det_powers_list[:5], 1):
                    marker = " ← Strongest" if i == 1 else ""
                    marker += " ← BIN 177!" if r == 177 else ""
                    print(f"    {i}. Range bin {r:4d}, Doppler bin {d:3d}, Power {p:.6e}{marker}")
            
            print(f"{'='*80}\n")
        # ═══════════════════════════════════════════════════════════════════
        
            print(f"{'─'*80}\n")

        return detections.astype(bool), threshold_map
    
    def _detect_ca_cfar_2d(
        self,
        power_map: Union[np.ndarray, 'cp.ndarray']
    ) -> Tuple[Union[np.ndarray, 'cp.ndarray'], Union[np.ndarray, 'cp.ndarray']]:
        
        xp = self.xp
        n_range, n_doppler = power_map.shape
        detections = xp.zeros((n_range, n_doppler), dtype=bool)
        threshold_map = xp.zeros((n_range, n_doppler), dtype=xp.float64)
        
        half_win = self.guard_cells + self.training_cells
        
        if self.verbose:
            print(f"  [CA-CFAR 2D] Processing {n_range}×{n_doppler} cells...")
            print(f"    Half window: {half_win} (guard={self.guard_cells}, train={self.training_cells})")
        
        for r_idx in range(n_range):
            for d_idx in range(n_doppler):
                r_start = max(0, r_idx - half_win)
                r_end = min(n_range, r_idx + half_win + 1)
                d_start = max(0, d_idx - half_win)
                d_end = min(n_doppler, d_idx + half_win + 1)
                
                window = power_map[r_start:r_end, d_start:d_end]
                
                mask = xp.ones(window.shape, dtype=bool)
                
                r_cut_local = r_idx - r_start
                d_cut_local = d_idx - d_start
                
                g = self.guard_cells
                r_guard_start = max(0, r_cut_local - g)
                r_guard_end = min(window.shape[0], r_cut_local + g + 1)
                d_guard_start = max(0, d_cut_local - g)
                d_guard_end = min(window.shape[1], d_cut_local + g + 1)
                
                mask[r_guard_start:r_guard_end, d_guard_start:d_guard_end] = False
                
                training_cells = window[mask]
                
                if len(training_cells) >= self.min_training_cells:
                    noise_power = float(xp.mean(training_cells))
                    threshold = self.threshold_multiplier * noise_power
                    threshold_map[r_idx, d_idx] = threshold
                    
                    if power_map[r_idx, d_idx] > threshold:
                        detections[r_idx, d_idx] = True
        
        return detections, threshold_map
    
    def _detect_ca_cfar_1d(
        self,
        power_map: Union[np.ndarray, 'cp.ndarray'],
        axis: Optional[int] = None
    ) -> Tuple[Union[np.ndarray, 'cp.ndarray'], Union[np.ndarray, 'cp.ndarray']]:
        
        xp = self.xp
        n_range, n_doppler = power_map.shape
        detections = xp.zeros_like(power_map, dtype=bool)
        threshold_map = xp.zeros_like(power_map, dtype=xp.float64)
        
        if self.verbose:
            print(f"  [CA-CFAR 1D] Processing along axis={axis}...")
        
        if axis == 0 or axis is None:
            for d_idx in range(n_doppler):
                for r_idx in range(n_range):
                    threshold, _ = self._compute_ca_threshold_1d(
                        power_map[:, d_idx], r_idx
                    )
                    threshold_map[r_idx, d_idx] = threshold
                    
                    if power_map[r_idx, d_idx] > threshold:
                        detections[r_idx, d_idx] = True
        
        elif axis == 1:
            for r_idx in range(n_range):
                for d_idx in range(n_doppler):
                    threshold, _ = self._compute_ca_threshold_1d(
                        power_map[r_idx, :], d_idx
                    )
                    threshold_map[r_idx, d_idx] = threshold
                    
                    if power_map[r_idx, d_idx] > threshold:
                        detections[r_idx, d_idx] = True
        
        return detections, threshold_map
    
    def _compute_ca_threshold_1d(
        self,
        profile: Union[np.ndarray, 'cp.ndarray'],
        idx: int
    ) -> Tuple[float, float]:
        
        xp = self.xp
        n = len(profile)
        
        start_left = max(0, idx - self.guard_cells - self.training_cells)
        end_left = max(0, idx - self.guard_cells)
        
        start_right = min(n, idx + self.guard_cells + 1)
        end_right = min(n, idx + self.guard_cells + self.training_cells + 1)
        
        train_left = profile[start_left:end_left]
        train_right = profile[start_right:end_right]
        
        n_train = len(train_left) + len(train_right)
        
        if n_train >= self.min_training_cells:
            noise_power = (xp.sum(train_left) + xp.sum(train_right)) / n_train
        else:
            noise_power = 0.0
        
        threshold = self.threshold_multiplier * float(noise_power)
        
        return float(threshold), float(noise_power)
    
    def _detect_os_cfar_2d(
        self,
        power_map: Union[np.ndarray, 'cp.ndarray']
    ) -> Tuple[Union[np.ndarray, 'cp.ndarray'], Union[np.ndarray, 'cp.ndarray']]:
        
        xp = self.xp
        n_range, n_doppler = power_map.shape
        detections = xp.zeros((n_range, n_doppler), dtype=bool)
        threshold_map = xp.zeros((n_range, n_doppler), dtype=xp.float64)
        
        half_win = self.guard_cells + self.training_cells
        
        if self.verbose:
            print(f"  [OS-CFAR 2D] Processing {n_range}×{n_doppler} cells...")
            print(f"    OS index (k): {self.os_index}")
        
        for r_idx in range(n_range):
            for d_idx in range(n_doppler):
                r_start = max(0, r_idx - half_win)
                r_end = min(n_range, r_idx + half_win + 1)
                d_start = max(0, d_idx - half_win)
                d_end = min(n_doppler, d_idx + half_win + 1)
                
                window = power_map[r_start:r_end, d_start:d_end]
                
                mask = xp.ones(window.shape, dtype=bool)
                r_cut = r_idx - r_start
                d_cut = d_idx - d_start
                
                g = self.guard_cells
                r_g_start = max(0, r_cut - g)
                r_g_end = min(window.shape[0], r_cut + g + 1)
                d_g_start = max(0, d_cut - g)
                d_g_end = min(window.shape[1], d_cut + g + 1)
                
                mask[r_g_start:r_g_end, d_g_start:d_g_end] = False
                
                train_cells = window[mask]
                
                if len(train_cells) >= self.min_training_cells:
                    train_sorted = xp.sort(train_cells)
                    k = min(self.os_index, len(train_sorted) - 1)
                    noise_estimate = float(train_sorted[k])
                    
                    threshold = self.threshold_multiplier * noise_estimate
                    threshold_map[r_idx, d_idx] = threshold
                    
                    if power_map[r_idx, d_idx] > threshold:
                        detections[r_idx, d_idx] = True
        
        return detections, threshold_map
    
    def _detect_go_cfar_2d(
        self,
        power_map: Union[np.ndarray, 'cp.ndarray']
    ) -> Tuple[Union[np.ndarray, 'cp.ndarray'], Union[np.ndarray, 'cp.ndarray']]:
        
        return self._detect_go_so_2d(power_map, use_max=True)
    
    def _detect_so_cfar_2d(
        self,
        power_map: Union[np.ndarray, 'cp.ndarray']
    ) -> Tuple[Union[np.ndarray, 'cp.ndarray'], Union[np.ndarray, 'cp.ndarray']]:
        
        return self._detect_go_so_2d(power_map, use_max=False)
    
    def _detect_go_so_2d(
        self,
        power_map: Union[np.ndarray, 'cp.ndarray'],
        use_max: bool = True
    ) -> Tuple[Union[np.ndarray, 'cp.ndarray'], Union[np.ndarray, 'cp.ndarray']]:
    
        xp = self.xp
        n_range, n_doppler = power_map.shape
        detections = xp.zeros((n_range, n_doppler), dtype=bool)
        threshold_map = xp.zeros((n_range, n_doppler), dtype=xp.float64)
    
        half_win = self.guard_cells + self.training_cells
        mode_name = "GO-CFAR" if use_max else "SO-CFAR"
    
        if self.verbose:
            print(f"  [{mode_name} 2D] Processing {n_range}×{n_doppler} cells...")
            print(f"    Strategy: {'Greatest-Of' if use_max else 'Smallest-Of'} 4-quadrant split")
    
        for r_idx in range(n_range):
            for d_idx in range(n_doppler):
                # Extract full window
                r_start = max(0, r_idx - half_win)
                r_end = min(n_range, r_idx + half_win + 1)
                d_start = max(0, d_idx - half_win)
                d_end = min(n_doppler, d_idx + half_win + 1)
            
                window = power_map[r_start:r_end, d_start:d_end]
            
                # Create mask for guard region (exclude CUT + guard cells)
                mask = xp.ones(window.shape, dtype=bool)
                r_cut = r_idx - r_start
                d_cut = d_idx - d_start
            
                g = self.guard_cells
                r_g_start = max(0, r_cut - g)
                r_g_end = min(window.shape[0], r_cut + g + 1)
                d_g_start = max(0, d_cut - g)
                d_g_end = min(window.shape[1], d_cut + g + 1)
            
                mask[r_g_start:r_g_end, d_g_start:d_g_end] = False
            
                # Extract all training cells
                train_cells = window[mask]
            
                if len(train_cells) < self.min_training_cells:
                    # Not enough training cells
                    continue
            
                # ========== CRITICAL FIX: PROPER GO/SO LOGIC ==========
                # Split training window into 4 quadrants for robust estimation
            
                # Create quadrant masks (all exclude guard region)
                quadrant_averages = []
            
                # Quadrant 1: Top-left (leading in both dimensions)
                mask_q1 = mask.copy()
                mask_q1[r_cut:, :] = False  # Remove bottom half
                mask_q1[:, d_cut:] = False  # Remove right half
                cells_q1 = window[mask_q1]
                if len(cells_q1) > 0:
                    quadrant_averages.append(float(xp.mean(cells_q1)))
            
                # Quadrant 2: Top-right (leading range, lagging Doppler)
                mask_q2 = mask.copy()
                mask_q2[r_cut:, :] = False  # Remove bottom half
                mask_q2[:, :d_cut+1] = False  # Remove left half
                cells_q2 = window[mask_q2]
                if len(cells_q2) > 0:
                    quadrant_averages.append(float(xp.mean(cells_q2)))
            
                # Quadrant 3: Bottom-left (lagging range, leading Doppler)
                mask_q3 = mask.copy()
                mask_q3[:r_cut+1, :] = False  # Remove top half
                mask_q3[:, d_cut:] = False  # Remove right half
                cells_q3 = window[mask_q3]
                if len(cells_q3) > 0:
                    quadrant_averages.append(float(xp.mean(cells_q3)))
             
                # Quadrant 4: Bottom-right (lagging in both dimensions)
                mask_q4 = mask.copy()
                mask_q4[:r_cut+1, :] = False  # Remove top half
                mask_q4[:, :d_cut+1] = False  # Remove left half
                cells_q4 = window[mask_q4]
                if len(cells_q4) > 0:
                    quadrant_averages.append(float(xp.mean(cells_q4)))
            
                # Need at least 2 quadrants for GO/SO to make sense
                if len(quadrant_averages) < 2:
                    # Fallback to CA-CFAR if window split failed
                    noise_power = float(xp.mean(train_cells))
                else:
                    # Apply GO or SO logic
                    if use_max:
                        # GO-CFAR: Take maximum (most pessimistic estimate)
                        noise_power = float(max(quadrant_averages))
                    else:
                        # SO-CFAR: Take minimum (most optimistic estimate)
                        noise_power = float(min(quadrant_averages))
            
                # ========== END FIX ==========
            
                threshold = self.threshold_multiplier * noise_power
                threshold_map[r_idx, d_idx] = threshold
            
                if power_map[r_idx, d_idx] > threshold:
                    detections[r_idx, d_idx] = True
    
        return detections, threshold_map
    
    def extract_detections(
        self,
        detections: np.ndarray,
        power_map: np.ndarray,
        max_detections: Optional[int] = 40
    ) -> List[Tuple[int, int, float]]:

        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL FIX: Ensure power_map is clean numpy array
        # ═══════════════════════════════════════════════════════════════════
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"[EXTRACT_DETECTIONS - ENTRY DIAGNOSTICS]")
            print(f"{'='*80}")
            print(f"  power_map shape: {power_map.shape}")
            print(f"  power_map dtype: {power_map.dtype}")
            print(f"  power_map type: {type(power_map)}")
            print(f"  power_map max: {float(np.max(power_map)):.6e}")
            print(f"  power_map min: {float(np.min(power_map)):.6e}")
            _dc_bin_ext = power_map.shape[1] // 2
            print(f"  power_map[177, {_dc_bin_ext}]: "
                  f"{float(power_map[177, _dc_bin_ext]):.6e}"
                  if power_map.shape[0] > 177 else "  N/A")
        
        # Force conversion to clean NumPy array (avoid view/memory issues)
        if np.iscomplexobj(power_map):
            if self.verbose:
                print(f"  [EXTRACT FIX] power_map is COMPLEX → computing |·|²")
            power_values = np.abs(power_map).astype(np.float64)
        else:
            if self.verbose:
                print(f"  [EXTRACT FIX] power_map is REAL → using as-is")
            # CRITICAL: Force copy to avoid view issues
            power_values = np.asarray(power_map, dtype=np.float64, order='C')
        
        if self.verbose:
            print(f"  ")
            print(f"  AFTER CONVERSION:")
            print(f"    power_values dtype: {power_values.dtype}")
            print(f"    power_values max: {float(np.max(power_values)):.6e}")
            print(f"    power_values[177, {_dc_bin_ext}]: "
                  f"{float(power_values[177, _dc_bin_ext]):.6e}"
                  if power_values.shape[0] > 177 else "    N/A")
            print(f"{'='*80}\n")
        # ═══════════════════════════════════════════════════════════════════

        det_indices = np.argwhere(detections)

        detection_list = []
        for idx in det_indices:
            r_idx = int(idx[0])  # ← Force to Python int
            d_idx = int(idx[1])  # ← Force to Python int
            power = float(power_values[r_idx, d_idx])
            detection_list.append((r_idx, d_idx, power))
            
        # ═══════════════════════════════════════════════════════════════════
        # DIAGNOSTIC: PRE-SORTING VERIFICATION - ARE POWER VALUES CORRECT?
        # ═══════════════════════════════════════════════════════════════════
        if self.verbose and len(detection_list) > 0:
            print(f"\n{'='*80}")
            print(f"[PRE-SORTING POWER CHECK]")
            print(f"{'='*80}")
            print(f"  Total detections before sorting: {len(detection_list)}")
            
            # ═══════════════════════════════════════════════════════════════
            # NEW: Find ALL bin 177 entries (not just first one)
            # ═══════════════════════════════════════════════════════════════
            bin_177_entries = []
            for entry in detection_list:
                if entry[0] == 177:  # range_idx == 177
                    bin_177_entries.append(entry)
            
            # Find the entry with maximum power in UNSORTED list
            max_power_entry = max(detection_list, key=lambda x: x[2])
            
            print(f"  ")
            print(f"  UNSORTED LIST CHECK:")
            print(f"    Highest power entry: Range bin {max_power_entry[0]}, Doppler bin {max_power_entry[1]}, Power {max_power_entry[2]:.6e}")
            
            if len(bin_177_entries) > 0:
                print(f"  ")
                print(f"  BIN 177 ENTRIES (ALL {len(bin_177_entries)} FOUND):")
                
                # Sort bin 177 entries by power to see range
                bin_177_sorted = sorted(bin_177_entries, key=lambda x: x[2], reverse=True)
                
                for i, (r, d, p) in enumerate(bin_177_sorted[:10], 1):  # Show top 10
                    marker = " ← STRONGEST!" if i == 1 else ""
                    marker += " ← DC BIN!" if d == 128 else ""
                    print(f"    {i}. Bin (177, {d:3d}), Power {p:.6e}{marker}")
                
                if len(bin_177_sorted) > 10:
                    print(f"    ... and {len(bin_177_sorted) - 10} more bin 177 entries")
                
                # Check if strongest bin 177 entry matches max_power_entry
                strongest_177 = bin_177_sorted[0]
                weakest_177 = bin_177_sorted[-1]
                
                print(f"  ")
                print(f"  BIN 177 POWER RANGE:")
                print(f"    Strongest: {strongest_177[2]:.6e} at Doppler {strongest_177[1]}")
                print(f"    Weakest: {weakest_177[2]:.6e} at Doppler {weakest_177[1]}")
                print(f"    Ratio (strongest/weakest): {strongest_177[2]/weakest_177[2]:.2f}×")
                print(f"  ")
                
                if strongest_177[2] == max_power_entry[2]:
                    print(f"  STATUS: ✅ Bin 177 HAS the highest power (before sorting)")
                else:
                    print(f"  STATUS: ❌ Bin 177 does NOT have highest power (before sorting)")
                    print(f"    Max power: {max_power_entry[2]:.6e} at bin ({max_power_entry[0]}, {max_power_entry[1]})")
                    print(f"    Bin 177 max: {strongest_177[2]:.6e}")
                    print(f"    Power ratio (177/max): {strongest_177[2]/max_power_entry[2]:.4f}")
            else:
                print(f"  ")
                print(f"  STATUS: ❌ Bin 177 NOT in detection list (CFAR didn't detect it)")
            
            print(f"{'='*80}\n")
        # ═══════════════════════════════════════════════════════════════════
    
        # Sort by power (strongest first)
        detection_list.sort(key=lambda x: x[2], reverse=True)
    
        # Phase 1 Fix: Limit detections for memory safety & processing time
        if max_detections is not None and len(detection_list) > max_detections:
            if self.verbose:
                print(f"  [EXTRACT] Detected {len(detection_list)} peaks, limiting to top {max_detections}")
            detection_list = detection_list[:max_detections]
            
        # ═══════════════════════════════════════════════════════════════════
        # DIAGNOSTIC: EXTRACTION OUTPUT - DID SORTING PRESERVE BIN 177?
        # ═══════════════════════════════════════════════════════════════════
        if self.verbose and len(detection_list) > 0:
            print(f"\n{'='*80}")
            print(f"[EXTRACTION OUTPUT DIAGNOSTIC]")
            print(f"{'='*80}")
            print(f"  Total extracted peaks: {len(detection_list)}")
            print(f"  ")
            
            # Show top 5 after sorting and limiting
            print(f"  TOP 5 AFTER SORTING & LIMITING:")
            for i in range(min(5, len(detection_list))):
                r_idx, d_idx, power = detection_list[i]
                marker = " ← #1 STRONGEST" if i == 0 else ""
                marker += " ← BIN 177!" if r_idx == 177 else ""
                print(f"    {i+1}. Range bin {r_idx:4d}, Doppler bin {d_idx:3d}, Power {power:.6e}{marker}")
            
            print(f"  ")
            
            # ═══════════════════════════════════════════════════════════════
            # NEW: Find ALL bin 177 entries in SORTED list
            # ═══════════════════════════════════════════════════════════════
            bin_177_in_sorted = []
            for i, (r_idx, d_idx, power) in enumerate(detection_list, 1):
                if r_idx == 177:
                    bin_177_in_sorted.append((i, r_idx, d_idx, power))  # rank, r, d, power
            
            print(f"  BIN 177 CHECK (AFTER SORTING):")
            if len(bin_177_in_sorted) > 0:
                print(f"    Found: ✅ YES ({len(bin_177_in_sorted)} entries)")
                print(f"  ")
                
                # Show top bin 177 entries
                print(f"    BIN 177 ENTRIES IN SORTED LIST:")
                for rank, r, d, p in bin_177_in_sorted[:5]:  # Show top 5
                    marker = " ← #1 OVERALL!" if rank == 1 else ""
                    marker += " ← DC BIN!" if d == 128 else ""
                    print(f"      Rank #{rank:3d}: Doppler bin {d:3d}, Power {p:.6e}{marker}")
                
                if len(bin_177_in_sorted) > 5:
                    print(f"      ... and {len(bin_177_in_sorted) - 5} more bin 177 entries")
                
                print(f"  ")
                
                # Check if bin 177 is #1
                top_rank = bin_177_in_sorted[0][0]
                top_power = bin_177_in_sorted[0][3]
                
                if top_rank == 1:
                    print(f"    Status: ✅ Bin 177 is #1 strongest (CORRECT)")
                else:
                    print(f"    Status: ❌ CRITICAL - Bin 177 is NOT #1!")
                    top_r, top_d, top_p = detection_list[0]
                    print(f"    #1 is actually: Range bin {top_r}, Doppler bin {top_d}")
                    print(f"    #1 power: {top_p:.6e}")
                    print(f"    Bin 177 highest power: {top_power:.6e}")
                    print(f"    Power ratio (177/top): {top_power/top_p:.4f}")
                    print(f"    ")
                    print(f"    🔴 BUG IDENTIFIED: SORTING OR POWER COMPUTATION ERROR!")
            else:
                print(f"    Found: ❌ NO")
                print(f"    Status: ❌ CRITICAL - Bin 177 is NOT in extracted list!")
                print(f"    This means CFAR did not detect bin 177 at all.")
            
            print(f"{'='*80}\n")
        # ═══════════════════════════════════════════════════════════════════
    
        return detection_list
    
    def compute_detection_stats(
        self,
        detections: np.ndarray,
        power_map: np.ndarray
    ) -> dict:
        
        det_list = self.extract_detections(detections, power_map)
        
        if len(det_list) == 0:
            return {
                'n_detections': 0,
                'total_cells': int(detections.size),
                'detection_rate': 0.0
            }
        
        powers = [d[2] for d in det_list]
        
        stats = {
            'n_detections': len(det_list),
            'total_cells': int(detections.size),
            'detection_rate': len(det_list) / detections.size,
            'min_power': float(np.min(powers)),
            'max_power': float(np.max(powers)),
            'mean_power': float(np.mean(powers)),
            'median_power': float(np.median(powers)),
            'min_power_db': float(10 * np.log10(np.min(powers) + 1e-12)),
            'max_power_db': float(10 * np.log10(np.max(powers) + 1e-12)),
            'mean_power_db': float(10 * np.log10(np.mean(powers) + 1e-12))
        }
        
        return stats


if __name__ == "__main__":
    print("\n" + "="*80)
    print("CFAR DETECTION v4.3 - CORRECTED & PRODUCTION-READY")
    print("="*80)