"""Fresh, pinned contender-repository materialization.

``LocalRepoProvider`` is the non-Git fixture/development seam. Production uses
``GitRepoProvider.checkout_pinned``: every call creates a new isolated checkout,
fetches one exact commit over HTTPS with a read-only token supplied through a
private askpass environment, verifies both commit and tree SHA, removes Git
metadata, and rejects unsafe or oversized trees. No existing checkout is listed,
resolved, cached, or reused.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Mapping, Protocol, Sequence, runtime_checkable
from urllib.parse import urlsplit

from vidaio.competition.runners import safeio
from vidaio.competition.runners.errors import (
    CheckoutError,
    CheckoutRejectedError,
    UnsafePathError,
)

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_ENV_ALLOWLIST = (
    "PATH",
    "TMPDIR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)


@runtime_checkable
class RepoProvider(Protocol):
    """Maps a pinned code identity to a local checkout directory."""

    def checkout(self, repo_url: str, commit_sha: str) -> Path:
        """Return a checkout. Production callers use :func:`checkout_pinned`."""
        ...


@runtime_checkable
class PinnedRepoProvider(Protocol):
    """Provider that can attest the enrolled commit and tree together."""

    def checkout_pinned(
        self, repo_url: str, commit_sha: str, tree_sha: str
    ) -> Path: ...


@runtime_checkable
class ReleasableRepoProvider(Protocol):
    """Provider owning temporary checkout storage that must be released."""

    def release(self, checkout: str | Path) -> None: ...


def checkout_pinned(
    provider: RepoProvider,
    repo_url: str,
    commit_sha: str,
    tree_sha: str,
) -> Path:
    """Use exact tree verification when supported, preserving local test seams."""
    if isinstance(provider, PinnedRepoProvider):
        return provider.checkout_pinned(repo_url, commit_sha, tree_sha)
    return provider.checkout(repo_url, commit_sha)


def release_checkout(provider: RepoProvider, checkout: str | Path) -> None:
    """Release one consumed checkout; fixture providers deliberately no-op."""
    if isinstance(provider, ReleasableRepoProvider):
        provider.release(checkout)


class LocalRepoProvider:
    """Explicit local mapping for fixtures and operator-controlled dry runs.

    Local directories do not necessarily contain Git metadata, so this provider
    cannot attest a commit/tree pin. It is never selected by production config.
    """

    def __init__(self, mapping: Mapping[str, str | Path]) -> None:
        self._mapping = {url: Path(path) for url, path in mapping.items()}

    def checkout(self, repo_url: str, commit_sha: str) -> Path:
        path = self._mapping.get(repo_url)
        if path is None:
            raise CheckoutError(f"no local checkout registered for {repo_url!r}")
        if not path.is_dir():
            raise CheckoutError(
                f"registered checkout for {repo_url!r} is not a directory: {path}"
            )
        return path

    def release(self, checkout: str | Path) -> None:
        """Mapped fixture trees are caller-owned and intentionally retained."""


class GitRepoProvider:
    """Create-only HTTPS Git provider for independently pinned submissions.

    ``read_only_token`` must be a provider token whose server-side scope is read
    only; Git cannot attest token scope itself. The token is never placed in the
    repo URL, argv, filesystem, exception, or log. A tiny askpass program reads it
    from the child-only environment. Global/system Git config and redirects are
    disabled so credential helpers, filters and redirect-to-internal-host tricks
    cannot expand the boundary.

    Successful checkouts remain owned until ``release``/``close`` so callers can
    build/archive them. Every call creates a different directory and fetches
    again; there is deliberately no cache or lookup of an earlier checkout.
    """

    def __init__(
        self,
        *,
        scratch_root: str | Path,
        read_only_token: str,
        username: str = "x-access-token",
        allowed_hosts: Sequence[str] = ("github.com",),
        git_path: str = "git",
        timeout_seconds: float = 180.0,
        max_checkout_bytes: int = 512 * 1024 * 1024,
        max_log_bytes: int = 1024 * 1024,
        poll_seconds: float = 0.1,
    ) -> None:
        if not read_only_token or "\x00" in read_only_token or "\n" in read_only_token:
            raise ValueError("read_only_token must be non-empty and single-line")
        if not username or "\x00" in username or "\n" in username:
            raise ValueError("Git username must be non-empty and single-line")
        hosts = frozenset(
            host.strip().lower() for host in allowed_hosts if host.strip()
        )
        if not hosts:
            raise ValueError("allowed_hosts must contain at least one exact HTTPS host")
        if timeout_seconds <= 0 or poll_seconds <= 0:
            raise ValueError("Git timeout and poll cadence must be positive")
        if max_checkout_bytes <= 0 or max_log_bytes <= 0:
            raise ValueError("Git checkout/log byte caps must be positive")
        self._root = Path(scratch_root)
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._token = read_only_token
        self._username = username
        self._allowed_hosts = hosts
        self._git = git_path
        self._timeout = float(timeout_seconds)
        self._max_checkout_bytes = int(max_checkout_bytes)
        self._max_log_bytes = int(max_log_bytes)
        self._poll = float(poll_seconds)
        self._owned: dict[Path, Path] = {}
        self._lock = threading.RLock()

    def checkout(self, repo_url: str, commit_sha: str) -> Path:
        raise CheckoutError(
            "GitRepoProvider requires the enrolled tree SHA; call checkout_pinned "
            "so production can verify both commit and tree identity"
        )

    def checkout_pinned(self, repo_url: str, commit_sha: str, tree_sha: str) -> Path:
        self._validate_pin(commit_sha, what="commit_sha")
        self._validate_pin(tree_sha, what="tree_sha")
        self._validate_url(repo_url)
        # One deadline covers the entire materialization, not each individual Git
        # command.  Otherwise a server can consume ``timeout_seconds`` once per
        # phase and turn a nominal three-minute boundary into a much longer one.
        deadline = time.monotonic() + self._timeout

        session = Path(tempfile.mkdtemp(prefix="vidaio-next-checkout-", dir=self._root))
        session.chmod(0o700)
        checkout = session / "repo"
        checkout.mkdir(mode=0o700)
        askpass = session / "vidaio-next-git-askpass.sh"
        askpass.write_text(
            "#!/bin/sh\n"
            'case "$1" in\n'
            "  *Username*) printf '%s' \"$VIDAIO_NEXT_GIT_USERNAME\" ;;\n"
            "  *Password*) printf '%s' \"$VIDAIO_NEXT_GIT_TOKEN\" ;;\n"
            "  *) exit 1 ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        askpass.chmod(0o700)
        env = self._git_env(askpass)
        try:
            self._run_git(
                ["init", "--quiet", str(checkout)],
                session=session,
                checkout=checkout,
                env=env,
                label="git init",
                deadline=deadline,
            )
            self._run_git(
                ["-C", str(checkout), "remote", "add", "origin", repo_url],
                session=session,
                checkout=checkout,
                env=env,
                label="git remote add",
                deadline=deadline,
            )
            self._run_git(
                [
                    "-C",
                    str(checkout),
                    "-c",
                    "protocol.version=2",
                    "-c",
                    "http.followRedirects=false",
                    "fetch",
                    "--quiet",
                    "--no-tags",
                    "--depth=1",
                    "--filter=blob:none",
                    "origin",
                    commit_sha,
                ],
                session=session,
                checkout=checkout,
                env=env,
                label="git fetch pinned commit",
                deadline=deadline,
            )
            fetched = self._run_git(
                ["-C", str(checkout), "rev-parse", "--verify", "FETCH_HEAD^{commit}"],
                session=session,
                checkout=checkout,
                env=env,
                label="git verify fetched commit",
                deadline=deadline,
            ).strip()
            if fetched != commit_sha:
                raise CheckoutRejectedError(
                    f"fetched commit {fetched!r} does not match enrolled {commit_sha}"
                )
            self._run_git(
                [
                    "-C",
                    str(checkout),
                    "checkout",
                    "--quiet",
                    "--detach",
                    "--force",
                    commit_sha,
                ],
                session=session,
                checkout=checkout,
                env=env,
                label="git checkout pinned commit",
                deadline=deadline,
            )
            head = self._run_git(
                ["-C", str(checkout), "rev-parse", "--verify", "HEAD^{commit}"],
                session=session,
                checkout=checkout,
                env=env,
                label="git verify HEAD",
                deadline=deadline,
            ).strip()
            actual_tree = self._run_git(
                ["-C", str(checkout), "rev-parse", "--verify", "HEAD^{tree}"],
                session=session,
                checkout=checkout,
                env=env,
                label="git verify tree",
                deadline=deadline,
            ).strip()
            if head != commit_sha:
                raise CheckoutRejectedError(
                    f"checked-out HEAD {head!r} does not match enrolled {commit_sha}"
                )
            if actual_tree != tree_sha:
                raise CheckoutRejectedError(
                    f"checked-out tree {actual_tree!r} does not match enrolled {tree_sha}"
                )
            if (checkout / ".gitmodules").exists():
                raise CheckoutRejectedError(
                    "submission contains .gitmodules; independently unpinned submodules "
                    "are forbidden"
                )
            # Never expose origin config, credentials/helpers, or Git object state
            # to the contender build context.
            shutil.rmtree(checkout / ".git")
            try:
                safeio.assert_safe_tree(checkout)
            except UnsafePathError as exc:
                raise CheckoutRejectedError(f"unsafe checkout tree: {exc}") from exc
            final_bytes = safeio.tree_bytes(checkout)
            if final_bytes > self._max_checkout_bytes:
                raise CheckoutRejectedError(
                    f"checkout is {final_bytes} bytes, over the cap of "
                    f"{self._max_checkout_bytes}"
                )
        except BaseException:
            shutil.rmtree(session, ignore_errors=True)
            raise
        with self._lock:
            self._owned[checkout] = session
        return checkout

    def release(self, checkout: str | Path) -> None:
        path = Path(checkout)
        with self._lock:
            session = self._owned.pop(path, None)
        if session is None:
            raise CheckoutError(
                f"refusing to release checkout not owned by this provider: {path}"
            )
        shutil.rmtree(session, ignore_errors=True)

    def close(self) -> None:
        with self._lock:
            sessions = list(self._owned.values())
            self._owned.clear()
        for session in sessions:
            shutil.rmtree(session, ignore_errors=True)

    def __enter__(self) -> GitRepoProvider:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @staticmethod
    def _validate_pin(value: str, *, what: str) -> None:
        if not _SHA1_RE.fullmatch(value):
            raise CheckoutRejectedError(
                f"{what} must be one lowercase 40-character Git SHA-1"
            )

    def _validate_url(self, value: str) -> None:
        if len(value) > 2048:
            raise CheckoutRejectedError("repo_url exceeds 2048 characters")
        parsed = urlsplit(value)
        try:
            port = parsed.port
        except ValueError as exc:
            raise CheckoutRejectedError(
                "repo_url must contain a valid HTTPS port"
            ) from exc
        host = (parsed.hostname or "").lower()
        if (
            parsed.scheme != "https"
            or not host
            or host not in self._allowed_hosts
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or port not in (None, 443)
        ):
            raise CheckoutRejectedError(
                "repo_url must be credential-free HTTPS on an explicitly allowed "
                f"host/port with no query or fragment (allowed: {sorted(self._allowed_hosts)})"
            )

    def _git_env(self, askpass: Path) -> dict[str, str]:
        env = {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}
        env.update(
            {
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_ASKPASS": str(askpass),
                "GIT_ASKPASS_REQUIRE": "force",
                "GIT_ALLOW_PROTOCOL": "https",
                "GIT_CEILING_DIRECTORIES": str(self._root.resolve()),
                "VIDAIO_NEXT_GIT_USERNAME": self._username,
                "VIDAIO_NEXT_GIT_TOKEN": self._token,
            }
        )
        return env

    def _run_git(
        self,
        args: Sequence[str],
        *,
        session: Path,
        checkout: Path,
        env: Mapping[str, str],
        label: str,
        deadline: float,
    ) -> str:
        log_path = session / f".git-command-{uuid.uuid4().hex}.log"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CheckoutError(
                f"checkout exceeded the bounded {self._timeout}s total Git timeout"
            )
        with log_path.open("xb") as log:
            try:
                process = subprocess.Popen(
                    [self._git, *args],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(env),
                    start_new_session=True,
                )
            except OSError as exc:
                raise CheckoutError(f"{label} could not start Git: {exc}") from exc
            violation: CheckoutError | None = None
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    violation = CheckoutError(
                        f"checkout exceeded the bounded {self._timeout}s total Git "
                        f"timeout during {label}"
                    )
                    break
                if safeio.tree_bytes(checkout) > self._max_checkout_bytes:
                    violation = CheckoutRejectedError(
                        f"{label} crossed the checkout cap of "
                        f"{self._max_checkout_bytes} bytes"
                    )
                    break
                try:
                    log_size = log_path.stat().st_size
                except OSError:
                    log_size = 0
                if log_size > self._max_log_bytes:
                    violation = CheckoutRejectedError(
                        f"{label} crossed the Git log cap of {self._max_log_bytes} bytes"
                    )
                    break
                time.sleep(self._poll)
            if violation is not None:
                self._kill_process_group(process)
                raise violation
            returncode = process.wait()
            log.flush()
            os.fsync(log.fileno())
        try:
            data = log_path.read_bytes()
        finally:
            log_path.unlink(missing_ok=True)
        if len(data) > self._max_log_bytes:
            raise CheckoutRejectedError(
                f"{label} wrote {len(data)} Git log bytes, over the cap of "
                f"{self._max_log_bytes}"
            )
        output = data.decode("utf-8", "replace")
        if returncode != 0:
            raise CheckoutError(
                f"{label} exited {returncode}: {output.strip()[-2000:]}"
            )
        if safeio.tree_bytes(checkout) > self._max_checkout_bytes:
            raise CheckoutRejectedError(
                f"{label} completed over the checkout cap of "
                f"{self._max_checkout_bytes} bytes"
            )
        return output

    @staticmethod
    def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
