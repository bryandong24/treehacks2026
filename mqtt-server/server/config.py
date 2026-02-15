"""Server configuration constants."""

# Model
MODEL_ID = "nvidia/Alpamayo-R1-10B"

# MQTT
MQTT_HOST = "localhost"
MQTT_PORT = 1883

# Inference
INFERENCE_INTERVAL_S = 5.0
NUM_TRAJ_SAMPLES = 1
MAX_GENERATION_LENGTH = 256
TEMPERATURE = 0.6
TOP_P = 0.98

# Ego motion
NUM_HISTORY_STEPS = 16
NUM_FRAMES = 4
TIME_STEP = 0.1  # 100ms between ego samples

# WebSocket
WS_MAX_SIZE = 4 * 1024 * 1024  # 4 MB
MOBILE_VIDEO_FPS = 5
