SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

TRACK_WIDTH = 5760
TRACK_HEIGHT = 3240

CAR_WIDTH = 80
CAR_HEIGHT = 106

ASSETS_PATH = "Assets"

FPS = 10000
CAMERA_SCROLL_SPEED = 0.1

COLOUR_SCHEME = [
    (255, 255, 255),
    (0, 0, 0),
    (31, 102, 31),
    (11, 36, 11),
    (8, 132, 28)
]

BACKGROUND_COLOUR = COLOUR_SCHEME[4]

BUTTON_BORDER_THICKNESS = 5
BUTTON_HOVER_THICKNESS = 7

DEFAULT_TRACK_NAME = "track"

TOTAL_LAPS = 2

CHECKPOINT_REWARD = 10
LAP_REWARD = 100
CRASH_REWARD = -100

TRAINING_TIMESTEP = 1 / 60
MAX_TIME_STEPS_PER_EPISODE = 3600

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4