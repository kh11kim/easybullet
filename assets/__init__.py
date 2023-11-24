from pathlib import Path

ASSETS_PATH = Path(__file__).parent
PANDA_URDF = ASSETS_PATH / "urdfs/panda/franka_panda.urdf"
PANDA_HAND_URDF = ASSETS_PATH / "urdfs/panda/hand.urdf"

def get_assets_path():
    return Path(__file__)