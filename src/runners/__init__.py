from .episode_runner_meta import EpisodeRunnerMeta
from .parallel_runner import ParallelRunner
from .episode_runner import EpisodeRunner
REGISTRY = {}

REGISTRY["episode"] = EpisodeRunner

REGISTRY["parallel"] = ParallelRunner

REGISTRY["episode_meta"] = EpisodeRunnerMeta
