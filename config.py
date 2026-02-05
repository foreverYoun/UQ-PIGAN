# config.py
"""
PID-GAN Configuration File - Uncertainty Quantification based on Training Error
"""

# ===== Data Paths =====
TRAIN_CSV_PATH = './train1/train_data1.csv'
TEST_CSV_PATH = './train1/test_data1.csv'
MIN_MAX_PATH = 'min_max_values.csv'

# ===== Output Paths =====
OUTPUT_DIR = './generate_data/'
MODEL_OUTPUT_DIR = './saved_models/'  # <--- New: Directory for saving models
LOG_DIR = './training_logs/'          # <--- New: Directory for saving training logs

# ===== Random Seed =====
RANDOM_SEED = 2025

# ===== Feature Columns =====
# Note: Values are kept in Chinese to match CSV headers
IMPACT_FEATURES = ['弹体重量', '弹体直径', '弹体长度', '侵彻速度', '靶体抗压强度', '弹体形状系数']
LABEL_FEATURE = '侵彻深度'

# ===== Physics Parameters =====
TARGET_DENSITY = 2300.0

# ===== Physics Model Velocity Ranges =====
V_FORRESTAL_MAX = 800.0    # Forrestal model upper limit (m/s)
V_JONES_MIN = 800.0        # Jones model lower limit (m/s)
V_JONES_MAX = 1500.0       # Jones model upper limit (m/s)

# ===== Uncertainty Quantification Parameters (Calculated from Training Set) =====
UNCERTAINTY_FORRESTAL = None  # To be calculated from training set before training
UNCERTAINTY_JONES = None      # To be calculated from training set before training
UNCERTAINTY_INVALID = 999.0   # Penalty value for out-of-range samples

# ===== Model Hyperparameters =====
LATENT_DIM = 16
CONDITION_DIM = 6
GENERATOR_HIDDEN_DIMS = [128, 256, 128]
DISCRIMINATOR_HIDDEN_DIMS = [128, 256, 128]

# ===== Training Hyperparameters =====
NUM_EPOCHS = 50000
BATCH_SIZE = 64
LEARNING_RATE_G = 3e-4
LEARNING_RATE_D = 2e-4
LAMBDA_PHYSICS = 1.0

# ===== Diffusion Model Hyperparameters =====
DIFFUSION_HIDDEN_DIM = 128
DIFFUSION_NUM_LAYERS = 3
DIFFUSION_NUM_TIMESTEPS = 1000
DIFFUSION_LEARNING_RATE = 2e-4

# ===== Data Generation Parameters =====
NUM_GENERATE_SAMPLES = 1000

# ===== Device =====
DEVICE = 'cuda'

# ===== Generation Mode =====
GENERATION_ONLY_MODE = False
