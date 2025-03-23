import pygame

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
ASPECT_RATIO = (SCREEN_WIDTH / SCREEN_HEIGHT)

TRACK_WIDTH = 5760
TRACK_HEIGHT = 3240

CAR_WIDTH = 80
CAR_HEIGHT = 106

ASSETS_PATH = "Assets"
TRACKS_PATH = "Assets/Tracks"

# Initialising fonts
FONT_16 = pygame.font.Font(f"{ASSETS_PATH}/Fonts/font.otf", 16)
FONT_32 = pygame.font.Font(f"{ASSETS_PATH}/Fonts/font.otf", 32)
FONT_64 = pygame.font.Font(f"{ASSETS_PATH}/Fonts/font.otf", 64)

# Loading car images
BLUE_CAR_IMAGE = pygame.transform.scale(pygame.image.load(f"{ASSETS_PATH}/Cars/BlueCar.png"), (CAR_WIDTH, CAR_HEIGHT))
RED_CAR_IMAGE = pygame.transform.scale(pygame.image.load(f"{ASSETS_PATH}/Cars/RedCar.png"), (CAR_WIDTH, CAR_HEIGHT))

FPS = 60
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

CHECKPOINT_FREQUENCY = 5
TOTAL_LAPS = 2

TRAINING_TIMESTEP = 1 / 5
MAX_IDLE_TIMESTEPS = 100
MAX_TIMESTEPS = 500

CHECKPOINT_REWARD = 100
LAP_REWARD = 100
CRASH_REWARD = -100
SPEED_REWARD = TRAINING_TIMESTEP / 2
IDLE_REWARD = -200

TRAINING_EPISODES = 100000
VISUALISATION_STEP = 100

BATCH_SIZE = 64
GAMMA = 0.95
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
TAU = 0.1
LR = 1e-3