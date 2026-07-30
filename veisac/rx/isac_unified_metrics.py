# isac_unified_metrics.py
"""
VeISAC — Unified ISAC Performance Metrics

Joint communication-sensing evaluation: EIR, Equivalent-MSE, and Capacity-Distortion metrics across monostatic and bistatic topologies.

Paper: "VeISAC: An End-to-End MIMO-OFDM-FMCW Framework for ISAC
        in 6G Vehicular Networks"
Authors: M. Ababsa, S. Ribouh, Y. El Hillali, A. Rivenq
"""

import numpy as np
from typing import Optional, Union

_FLOAT = Union[float, np.ndarray]
_EPS   = 1e-12


def _db2lin(x_db: _FLOAT) -> _FLOAT:
    return 10.0 ** (np.asarray(x_db, dtype=float) / 10.0)


def _lin2db(x_lin: _FLOAT) -> _FLOAT:
    return 10.0 * np.log10(np.asarray(x_lin, dtype=float) + _EPS)


class ISACUnifiedMetrics:

    def __init__(
        self,
        bandwidth_hz:       float = 200e6,
        t_chirp_s:          float = 20e-6,
        n_chirps:           int   = 128,
        max_range_m:        float = 300.0,
        max_velocity_ms:    float = 50.0,
        theta_fov_deg:      float = 90.0,
        n_tx:               int   = 4,
        n_rx:               int   = 4,
        comm_power_factor:  float = np.sqrt(0.5),
        radar_power_factor: float = np.sqrt(0.5),
    ):
        self.B         = bandwidth_hz
        self.T_chirp   = t_chirp_s
        self.N_chirps  = n_chirps
        self.T_frame   = t_chirp_s * n_chirps
        self.R_max     = max_range_m
        self.V_max     = max_velocity_ms
        self.theta_max = theta_fov_deg
        self.N_tx      = n_tx
        self.N_rx      = n_rx
        self.alpha     = comm_power_factor
        self.beta      = radar_power_factor
        self.alpha2    = comm_power_factor ** 2
        self.beta2     = radar_power_factor ** 2
        self.P_R       = (max_range_m     / 2.0) ** 2
        self.P_V       = (max_velocity_ms / 2.0) ** 2
        self.P_theta   = (theta_fov_deg   / 2.0) ** 2

    def compute_all(
        self,
        comm_sinr_eff_db:        _FLOAT,
        radar_range_error_m:     _FLOAT,
        radar_velocity_error_ms: _FLOAT,
        radar_angle_error_deg:   _FLOAT,
        radar_crb_range_m:       _FLOAT,
        radar_crb_velocity_ms:   _FLOAT,
        radar_crb_angle_deg:     _FLOAT,
        radar_scnr_db:           Optional[_FLOAT] = None,
        comm_power_fraction:     Optional[_FLOAT] = None,
        radar_power_fraction:    Optional[_FLOAT] = None,
    ) -> dict:
        out = {}
        out.update(self.metric2_equivalent_mse(
            comm_sinr_eff_db        = comm_sinr_eff_db,
            radar_range_error_m     = radar_range_error_m,
            radar_velocity_error_ms = radar_velocity_error_ms,
            radar_angle_error_deg   = radar_angle_error_deg,
        ))
        out.update(self.metric1_estimation_information_rate(
            radar_range_error_m     = radar_range_error_m,
            radar_velocity_error_ms = radar_velocity_error_ms,
            radar_angle_error_deg   = radar_angle_error_deg,
            radar_crb_range_m       = radar_crb_range_m,
            radar_crb_velocity_ms   = radar_crb_velocity_ms,
            radar_crb_angle_deg     = radar_crb_angle_deg,
        ))
        out.update(self.metric3_capacity_distortion(
            comm_sinr_eff_db      = comm_sinr_eff_db,
            radar_crb_range_m     = radar_crb_range_m,
            radar_crb_velocity_ms = radar_crb_velocity_ms,
            radar_crb_angle_deg   = radar_crb_angle_deg,
        ))
        out.update(self.metric4_sinr_unified(
            comm_sinr_eff_db      = comm_sinr_eff_db,
            radar_scnr_db         = radar_scnr_db,
            radar_crb_range_m     = radar_crb_range_m,
            radar_crb_velocity_ms = radar_crb_velocity_ms,
            radar_crb_angle_deg   = radar_crb_angle_deg,
            comm_power_fraction   = comm_power_fraction,
            radar_power_fraction  = radar_power_fraction,
        ))
        return out

    def metric2_equivalent_mse(
        self,
        comm_sinr_eff_db:        _FLOAT,
        radar_range_error_m:     _FLOAT,
        radar_velocity_error_ms: _FLOAT,
        radar_angle_error_deg:   _FLOAT,
    ) -> dict:
        gamma_c      = _db2lin(comm_sinr_eff_db)
        C_comm       = np.maximum(np.log2(1.0 + gamma_c), 0.0)
        D_equiv      = 2.0 ** (-C_comm)
        D_equiv_norm = D_equiv / (1.0 / (1.0 + _db2lin(30.0)) + _EPS)

        D_R     = np.asarray(radar_range_error_m,    dtype=float) ** 2
        D_V     = np.asarray(radar_velocity_error_ms,dtype=float) ** 2
        D_theta = np.asarray(radar_angle_error_deg,  dtype=float) ** 2

        ratio_R     = D_equiv / (D_R     + _EPS)
        ratio_V     = D_equiv / (D_V     + _EPS)
        ratio_theta = D_equiv / (D_theta + _EPS)

        return {
            'm2_comm_sinr_lin':              gamma_c,
            'm2_comm_capacity_bps_hz':       C_comm,
            'm2_D_equiv':                    D_equiv,
            'm2_D_equiv_norm':               D_equiv_norm,
            'm2_D_range_m2':                 D_R,
            'm2_D_velocity_m2s2':            D_V,
            'm2_D_angle_deg2':               D_theta,
            'm2_ratio_equiv_vs_range':       ratio_R,
            'm2_ratio_equiv_vs_velocity':    ratio_V,
            'm2_ratio_equiv_vs_angle':       ratio_theta,
            'm2_ratio_equiv_vs_range_db':    _lin2db(ratio_R),
            'm2_ratio_equiv_vs_velocity_db': _lin2db(ratio_V),
            'm2_ratio_equiv_vs_angle_db':    _lin2db(ratio_theta),
            'm2_isac_balance_range':    np.where(ratio_R     < 1.0, 'sensing_limited', 'comm_limited'),
            'm2_isac_balance_velocity': np.where(ratio_V     < 1.0, 'sensing_limited', 'comm_limited'),
            'm2_isac_balance_angle':    np.where(ratio_theta < 1.0, 'sensing_limited', 'comm_limited'),
        }

    def metric1_estimation_information_rate(
        self,
        radar_range_error_m:     _FLOAT,
        radar_velocity_error_ms: _FLOAT,
        radar_angle_error_deg:   _FLOAT,
        radar_crb_range_m:       _FLOAT,
        radar_crb_velocity_ms:   _FLOAT,
        radar_crb_angle_deg:     _FLOAT,
    ) -> dict:
        D_R     = np.asarray(radar_range_error_m,    dtype=float) ** 2
        D_V     = np.asarray(radar_velocity_error_ms,dtype=float) ** 2
        D_theta = np.asarray(radar_angle_error_deg,  dtype=float) ** 2

        CRB_R     = np.asarray(radar_crb_range_m,    dtype=float) ** 2
        CRB_V     = np.asarray(radar_crb_velocity_ms,dtype=float) ** 2
        CRB_theta = np.asarray(radar_crb_angle_deg,  dtype=float) ** 2

        D_R_eff     = np.maximum(D_R,     0.1 * CRB_R)
        D_V_eff     = np.maximum(D_V,     0.1 * CRB_V)
        D_theta_eff = np.maximum(D_theta, 0.1 * CRB_theta)

        EIR_R     = np.maximum(0.5 * np.log2(self.P_R     / (D_R_eff     + _EPS)), 0.0)
        EIR_V     = np.maximum(0.5 * np.log2(self.P_V     / (D_V_eff     + _EPS)), 0.0)
        EIR_theta = np.maximum(0.5 * np.log2(self.P_theta / (D_theta_eff + _EPS)), 0.0)

        EIR_total = EIR_R + EIR_V + EIR_theta

        return {
            'm1_EIR_range_bits_obs':     EIR_R,
            'm1_EIR_velocity_bits_obs':  EIR_V,
            'm1_EIR_angle_bits_obs':     EIR_theta,
            'm1_EIR_total_bits_obs':     EIR_total,
            'm1_EIR_range_bps':          EIR_R     / self.T_frame,
            'm1_EIR_velocity_bps':       EIR_V     / self.T_frame,
            'm1_EIR_angle_bps':          EIR_theta / self.T_frame,
            'm1_EIR_total_bps':          EIR_total / self.T_frame,
            'm1_T_frame_s':              self.T_frame,
            'm1_P_R_prior_m2':           self.P_R,
            'm1_P_V_prior_m2s2':         self.P_V,
            'm1_P_theta_prior_deg2':     self.P_theta,
            'm1_D_R_effective_m2':       D_R_eff,
            'm1_D_V_effective_m2s2':     D_V_eff,
            'm1_D_theta_effective_deg2': D_theta_eff,
        }

    def metric3_capacity_distortion(
        self,
        comm_sinr_eff_db:      _FLOAT,
        radar_crb_range_m:     _FLOAT,
        radar_crb_velocity_ms: _FLOAT,
        radar_crb_angle_deg:   _FLOAT,
    ) -> dict:
        gamma_c = _db2lin(comm_sinr_eff_db)
        C_comm  = np.maximum(np.log2(1.0 + gamma_c), 0.0)

        CRB_R     = np.asarray(radar_crb_range_m,    dtype=float)
        CRB_V     = np.asarray(radar_crb_velocity_ms,dtype=float)
        CRB_theta = np.asarray(radar_crb_angle_deg,  dtype=float)

        CRB_R_norm     = CRB_R     / (self.R_max     + _EPS)
        CRB_V_norm     = CRB_V     / (self.V_max     + _EPS)
        CRB_theta_norm = CRB_theta / (self.theta_max + _EPS)

        D_sensing_norm = (CRB_R_norm + CRB_V_norm + CRB_theta_norm) / 3.0
        CD_score       = C_comm * np.exp(-D_sensing_norm)

        return {
            'm3_C_comm_bps_hz':            C_comm,
            'm3_ISAC_spectral_eff_bps_hz': C_comm,
            'm3_CRB_range_m':              CRB_R,
            'm3_CRB_velocity_ms':          CRB_V,
            'm3_CRB_angle_deg':            CRB_theta,
            'm3_CRB_range_norm':           CRB_R_norm,
            'm3_CRB_velocity_norm':        CRB_V_norm,
            'm3_CRB_angle_norm':           CRB_theta_norm,
            'm3_D_sensing_norm':           D_sensing_norm,
            'm3_CD_operating_C':           C_comm,
            'm3_CD_operating_D':           D_sensing_norm,
            'm3_CD_score':                 CD_score,
        }

    def metric4_sinr_unified(
        self,
        comm_sinr_eff_db:      _FLOAT,
        radar_scnr_db:         Optional[_FLOAT],
        radar_crb_range_m:     _FLOAT,
        radar_crb_velocity_ms: _FLOAT,
        radar_crb_angle_deg:   _FLOAT,
        comm_power_fraction:   Optional[_FLOAT] = None,
        radar_power_fraction:  Optional[_FLOAT] = None,
    ) -> dict:
        _cpf   = comm_power_fraction  if comm_power_fraction  is not None else self.alpha2
        _rpf   = radar_power_fraction if radar_power_fraction is not None else self.beta2
        alpha2 = float(np.nanmean(_cpf)) if isinstance(_cpf, (np.ndarray, list)) else float(_cpf)
        beta2  = float(np.nanmean(_rpf)) if isinstance(_rpf, (np.ndarray, list)) else float(_rpf)

        gamma_c = _db2lin(comm_sinr_eff_db)
        R_comm  = np.maximum(np.log2(1.0 + gamma_c), 0.0)

        if radar_scnr_db is not None:
            gamma_s    = _db2lin(radar_scnr_db)
            gamma_s_db = np.asarray(radar_scnr_db, dtype=float)
        else:
            gamma_s    = np.full_like(gamma_c, np.nan)
            gamma_s_db = np.full_like(gamma_c, np.nan)

        CRB_R     = np.asarray(radar_crb_range_m,    dtype=float)
        CRB_V     = np.asarray(radar_crb_velocity_ms,dtype=float)
        CRB_theta = np.asarray(radar_crb_angle_deg,  dtype=float)

        G_proc            = self.N_tx * self.N_rx * 128 * 1664
        power_split_ratio = alpha2 / (beta2 + _EPS)
        isac_product      = gamma_c * gamma_s
        isac_geom_sinr    = np.sqrt(np.abs(isac_product) + _EPS)
        R_comm_norm       = R_comm  / (np.log2(1.0 + _db2lin(30.0)) + _EPS)
        gamma_s_norm      = gamma_s / (_db2lin(30.0) + _EPS)

        return {
            'm4_gamma_c_lin':             gamma_c,
            'm4_gamma_c_db':              np.asarray(comm_sinr_eff_db, dtype=float),
            'm4_gamma_s_lin':             gamma_s,
            'm4_gamma_s_db':              gamma_s_db,
            'm4_R_comm_bps_hz':           R_comm,
            'm4_CRB_range_m':             CRB_R,
            'm4_CRB_velocity_ms':         CRB_V,
            'm4_CRB_angle_deg':           CRB_theta,
            'm4_alpha2':                  alpha2,
            'm4_beta2':                   beta2,
            'm4_power_split_ratio':       power_split_ratio,
            'm4_power_split_ratio_db':    _lin2db(power_split_ratio),
            'm4_processing_gain_lin':     G_proc,
            'm4_processing_gain_db':      _lin2db(G_proc),
            'm4_ISAC_sinr_product_lin':   isac_product,
            'm4_ISAC_sinr_product_db':    _lin2db(np.abs(isac_product) + _EPS),
            'm4_ISAC_effective_sinr_lin': isac_geom_sinr,
            'm4_ISAC_effective_sinr_db':  _lin2db(isac_geom_sinr),
            'm4_ISAC_operating_score':    R_comm_norm * gamma_s_norm,
        }


def compute_unified_metrics_from_row(
    row:       dict,
    config:    ISACUnifiedMetrics,
    angle_col: str = 'radar_angle_error_deg',
) -> dict:
    def _get(key, default=np.nan):
        v = row.get(key, default)
        return default if v is None or (isinstance(v, float) and np.isnan(v)) else v

    return config.compute_all(
        comm_sinr_eff_db        = _get('comm_sinr_eff_db'),
        radar_range_error_m     = abs(_get('radar_range_error_median_m',
                                          _get('radar_range_error_m'))),
        radar_velocity_error_ms = abs(_get('radar_velocity_error_median_ms',
                                          _get('radar_velocity_error_ms'))),
        radar_angle_error_deg   = abs(_get(angle_col)),
        radar_crb_range_m       = _get('radar_crb_range_m'),
        radar_crb_velocity_ms   = _get('radar_crb_velocity_ms'),
        radar_crb_angle_deg     = _get('radar_crb_angle_deg'),
        radar_scnr_db           = _get('radar_scnr_db', None),
        comm_power_fraction     = _get('comm_power_fraction', None),
        radar_power_fraction    = _get('radar_power_fraction', None),
    )


def compute_unified_metrics_from_dataframe(
    df,
    config:    ISACUnifiedMetrics,
    angle_col: str = 'radar_angle_error_deg',
) -> 'pd.DataFrame':
    import pandas as pd

    def _col(name, fallback=None):
        if name in df.columns:
            return df[name].values.astype(float)
        if fallback and fallback in df.columns:
            return df[fallback].values.astype(float)
        return np.full(len(df), np.nan)

    comm_sinr_db = _col('comm_sinr_eff_db')
    range_err    = np.abs(_col('radar_range_error_median_m', 'radar_range_error_m'))
    velocity_err = np.abs(_col('radar_velocity_error_median_ms', 'radar_velocity_error_ms'))
    angle_err    = np.abs(_col(angle_col))
    crb_range    = _col('radar_crb_range_m')
    crb_velocity = _col('radar_crb_velocity_ms')
    crb_angle    = _col('radar_crb_angle_deg')
    scnr_db      = _col('radar_scnr_db')
    comm_pf      = _col('comm_power_fraction')
    radar_pf     = _col('radar_power_fraction')

    comm_sinr_db = np.where(np.isfinite(comm_sinr_db),                        comm_sinr_db,  0.0)
    range_err    = np.where(np.isfinite(range_err)    & (range_err    > 0),   range_err,     1e-3)
    velocity_err = np.where(np.isfinite(velocity_err) & (velocity_err > 0),   velocity_err,  1e-4)
    angle_err    = np.where(np.isfinite(angle_err)    & (angle_err    > 0),   angle_err,     0.1)
    crb_range    = np.where(np.isfinite(crb_range)    & (crb_range    > 0),   crb_range,     1e-4)
    crb_velocity = np.where(np.isfinite(crb_velocity) & (crb_velocity > 0),   crb_velocity,  1e-5)
    crb_angle    = np.where(np.isfinite(crb_angle)    & (crb_angle    > 0),   crb_angle,     0.1)

    results = config.compute_all(
        comm_sinr_eff_db        = comm_sinr_db,
        radar_range_error_m     = range_err,
        radar_velocity_error_ms = velocity_err,
        radar_angle_error_deg   = angle_err,
        radar_crb_range_m       = crb_range,
        radar_crb_velocity_ms   = crb_velocity,
        radar_crb_angle_deg     = crb_angle,
        radar_scnr_db           = scnr_db,
        comm_power_fraction     = comm_pf,
        radar_power_fraction    = radar_pf,
    )

    out_df = df.copy()
    for k, v in results.items():
        if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == len(df):
            out_df[k] = v
        elif not isinstance(v, (np.ndarray, str)):
            out_df[k] = v
    return out_df