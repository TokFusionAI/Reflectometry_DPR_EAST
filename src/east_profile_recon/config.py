from pathlib import Path

DEFAULT_IT_PATH = "D:/workspace/data/IT.mat"
DEFAULT_DATA_ROOT = "D:/data/refl_data/IQ_data"
DEFAULT_VACUUM_DATA_PATH = "D:/workspace/data/vacuum_data/vacuum_data_new6.h5"
DEFAULT_QVW_DATA_PATH = "D:/workspace/data/QVWdata.h5"
DEFAULT_VACUUM_OUTPUT_DIR = "D:/workspace/data/vacuum_data"
DEFAULT_OUTPUT_ROOT = "D:/workspace/data/output"
DEFAULT_PLOT_OUTPUT_ROOT = "D:/workspace/data/plot"

TIME_DELAY = 17.4e-3
T_SCAN = 40e-6
Q_F_BASE = (8.1406, 14.2073)
V_F_BASE = (11.8427, 19.3105)
W_F_BASE = (11.8313, 19.1936)
FREQ_OFFSET = 0.1
WINDOW_SIZE = 128
FFTPOINT = 256
FS = 62.5e6
DEFAULT_FILTERS = (
    (-2, 15),
    (0, 15),
    (0, 35),
)

ENDPOINT_CUSUM_K = 14.0
ENDPOINT_CUSUM_H = 14.0
ENDPOINT_SNR_CONTRAST_RATIO = 0.12
ENDPOINT_CONSECUTIVE_BELOW = 3
