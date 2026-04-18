"""Shared paths and training defaults."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Geographic scope for invasive-risk blurbs and UI copy (iNaturalist export is place_id=829, San Diego area).
REGION_SHORT = "San Diego County"
REGION_LONG = "San Diego County, California"

CSV_PATH = ROOT / "observations-711965" / "observations-711965.csv"
DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
MODELS_DIR = ROOT / "models"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"
CHECKPOINT_PATH = MODELS_DIR / "plant_classifier.pt"

# Only train on taxa with at least this many images (sparse data otherwise).
MIN_SAMPLES_PER_CLASS = 5

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
TRAIN_FRACTION = 0.82
RANDOM_SEED = 42

# --- iNaturalist Computer Vision (optional; used when INAT_JWT is set / BACKEND=inaturalist)
INAT_COMPUTER_VISION_URL = "https://api.inaturalist.org/v1/computervision/score_image"
# Default map hint for CV geo model (San Diego County centroid). Override with INAT_LAT / INAT_LNG env.
DEFAULT_OBS_LAT = 32.7157
DEFAULT_OBS_LNG = -117.1611
# Kingdom Plantae on iNaturalist — narrows vision scores to plants.
PLANTAE_TAXON_ID = 47126
