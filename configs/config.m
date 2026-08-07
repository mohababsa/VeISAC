%% General Parameters
dv.dataset_folder = 'scenarios';
dv.scenario = 'O1';

dv.scenes = [1075:1099];

dv.basestations = [1,2,3,4]; 
dv.comm.enable = true;
dv.radar.enable = true;

dv.camera = true;
dv.camera_id = ["unit1_cam1", "unit1_cam2", "unit1_cam3", "unit2_cam4", "unit2_cam5", "unit1_cam6", 
                "unit3_cam7", "unit3_cam8", "unit3_cam9", "unit4_cam10", "unit4_cam11", "unit4_cam12"];

dv.lidar = true;
dv.lidar_id = ["unit1_lidar1", "unit2_lidar2", "unit3_lidar3", "unit4_lidar4"];

dv.position = true;

%% ============================================================================
%% COMM - CONFIGURATION 1-MEDIUM (200 MHz)
%% ============================================================================

% Antenna Configuration (unchanged)
dv.comm.bs_antenna.shape = [2, 2];
dv.comm.bs_antenna.rotation = [330, -10, 0];
dv.comm.bs_antenna.spacing = 0.5;
dv.comm.bs_antenna.FoV = [360, 180];

dv.comm.ue_antenna.shape = [2, 1];
dv.comm.ue_antenna.rotation = [130, 0, 0];
dv.comm.ue_antenna.spacing = 0.5;
dv.comm.ue_antenna.FoV = [360, 180];

% OFDM Parameters - MODIFIED FOR 200 MHz BANDWIDTH
dv.comm.OFDM.bandwidth = 200e6;         % 200 MHz 
dv.comm.OFDM.subcarriers = 1638;        % Total subcarriers 
dv.comm.OFDM.selected_subcarriers = [0:1632];  % Data subcarriers 

% Channel Generation
dv.comm.activate_RX_filter = 0;
dv.comm.generate_OFDM_channels = 1;
dv.comm.num_paths = 20;
dv.comm.enable_Doppler = 1;

dv.comm.active_bs = [1,2,3,4];

%% ============================================================================
%% RADAR - CONFIGURATION 1-MEDIUM (200 MHz)
%% ============================================================================

% Antenna Configuration 
dv.radar.tx_antenna.shape = [2, 2];
dv.radar.tx_antenna.rotation = [330, -10, 0];
dv.radar.tx_antenna.spacing = 0.5;
dv.radar.tx_antenna.FoV = [180, 180];

dv.radar.rx_antenna.shape = [2, 1];
dv.radar.rx_antenna.rotation = [330, -10, 0];
dv.radar.rx_antenna.spacing = 0.5;
dv.radar.rx_antenna.FoV = [180, 180];

% FMCW Parameters - MODIFIED FOR 200 MHz BANDWIDTH
dv.radar.FMCW.chirp_slope = 2.4e13;         % 2.4×10^13 Hz/s 
dv.radar.FMCW.Fs = 200e6;                   % 200 MHz sampling 
dv.radar.FMCW.n_samples_per_chirp = 1664;   % Samples per chirp 
dv.radar.FMCW.n_chirps = 128;               % Chirps per frame 

dv.radar.num_paths = 200;