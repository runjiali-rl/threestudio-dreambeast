import threestudio
from packaging.version import Version

if hasattr(threestudio, "__version__") and Version(threestudio.__version__) >= Version(
    "0.2.0"
):
    pass
else:
    if hasattr(threestudio, "__version__"):
        print(f"[INFO] threestudio version: {threestudio.__version__}")
    raise ValueError(
        "threestudio version must be >= 0.2.0, please update threestudio by pulling the latest version from github"
    )

from .background import neural_environment_map_background
from .data import uncond_multiview
from .guidance import mvdream_guidance, stable_diffusion_3_guidance, stable_diffusion_3_prompt_processor

from .system import dreambeast
