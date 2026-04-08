import os


def get_float_env(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


NOISE_WEIGHT = get_float_env("FORENSICHECK_NOISE_WEIGHT", 0.35)
VIT_WEIGHT = get_float_env("FORENSICHECK_VIT_WEIGHT", 0.65)
AUTHENTIC_THRESHOLD = get_float_env("FORENSICHECK_AUTHENTIC_THRESHOLD", 50.0)
