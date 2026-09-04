"""Sandbox runners: implementations of vidaio.competition.interfaces.SandboxRunner.

DockerSandboxRunner is the report/local implementation. ModalSandboxRunner is the
create-only production implementation for fresh, network-blocked GPU contender
execution; canonical scoring and auditing remain CPU-only. Both implement the
same competition runner interface.
"""

from vidaio.competition.runners import safeio
from vidaio.competition.runners.docker_runner import DockerSandboxRunner
from vidaio.competition.runners.errors import (
    BatchExecutionError,
    BatchTimeout,
    BuildError,
    BuildTimeout,
    CheckoutError,
    CheckoutRejectedError,
    ContenderBuildError,
    ContenderFaultError,
    InputStagingError,
    OutputRejectedError,
    OversizeOutputError,
    RunnerUnavailableError,
    SandboxIsolationError,
    SandboxProbeUnavailableError,
    SandboxRunnerError,
    SolutionExitError,
    UnknownImageError,
    UnsafePathError,
)
from vidaio.competition.runners.modal_runner import (
    FRESH_CREATION_CONFIRMATION,
    ModalRunnerConfig,
    ModalSandboxRunner,
    ModalSdkRuntime,
)
from vidaio.competition.runners.repo import (
    GitRepoProvider,
    LocalRepoProvider,
    PinnedRepoProvider,
    ReleasableRepoProvider,
    RepoProvider,
    checkout_pinned,
    release_checkout,
)

__all__ = [
    "DockerSandboxRunner",
    "ModalSandboxRunner",
    "ModalRunnerConfig",
    "ModalSdkRuntime",
    "FRESH_CREATION_CONFIRMATION",
    "RepoProvider",
    "PinnedRepoProvider",
    "ReleasableRepoProvider",
    "checkout_pinned",
    "release_checkout",
    "LocalRepoProvider",
    "GitRepoProvider",
    "SandboxRunnerError",
    "ContenderFaultError",
    "RunnerUnavailableError",
    "CheckoutError",
    "CheckoutRejectedError",
    "BuildError",
    "ContenderBuildError",
    "BuildTimeout",
    "BatchExecutionError",
    "BatchTimeout",
    "SolutionExitError",
    "OutputRejectedError",
    "OversizeOutputError",
    "UnsafePathError",
    "SandboxIsolationError",
    "SandboxProbeUnavailableError",
    "UnknownImageError",
    "InputStagingError",
    "safeio",
]
