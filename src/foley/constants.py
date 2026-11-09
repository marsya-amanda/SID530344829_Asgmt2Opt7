SR, DURATION, N_SAMPLES = 22050, 4.0, 88200
N_FFT, HOP, N_MELS, FMIN, FMAX = 1024, 256, 80, 0, 11025
N_FRAMES = 1 + N_SAMPLES // HOP
T_PAD = 352
CLASS_NAMES = ["DogBark", "Footstep", "GunShot", "Keyboard",
               "MovingMotorVehicle", "Rain", "Sneeze/Cough"]
N_CLASSES = 7
NULL_ID = 7                            # CFG "no class" token

_KEY = {"dogbark": 0, "footstep": 1, "gunshot": 2, "keyboard": 3,
        "movingmotorvehicle": 4, "rain": 5, "sneezecough": 6}

def folder_to_id(name: str) -> int:
    """dev/eval folder name -> class id. Tolerant to case/underscores
    ('dog_bark', 'DogBark', 'gun_shot' and 'gunshot' all work). Document this mapping in README."""
    key = name.lower().replace("_", "").replace("-", "").replace(" ", "").replace("/", "")
    if key not in _KEY:
        raise ValueError(f"Unknown sound-class folder: {name!r}")
    return _KEY[key]