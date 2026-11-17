"""Miner entries while enrollment is open: versions, pre-deadline checks, summaries.

An entry is one contender row (one hotkey per competition). Before the enrollment
deadline the entrant may replace the pinned ``(repo_url, commit_sha, tree_sha)`` IN
PLACE (same ``contender_id``) or withdraw it; nothing downstream reads an entry
before the deadline, and finalization, validation, building and the epoch evidence
all read the row as it stands after the deadline.

While the competition is ENROLLING the orchestrator checks every entry ahead of the
deadline and only WARNS: a fetch check through the production checkout (the real
reader credentials) and, when enabled, one trial build of that exact version on the
real builder. Neither ever changes a contender's status or halts the competition.

Everything lives in the append-only event log:

``contender_entry_replaced``
    ``{contender_id, hotkey, previous_repo_url, repo_url, previous_commit_sha,
    commit_sha, previous_tree_sha, tree_sha}``
``contender_withdrawn``
    ``{contender_id, hotkey, reason: "withdrawn by the miner"}``
``contender_prechecked``
    ``{contender_id, repo_url, commit_sha, tree_sha, ok, problems: [codes], detail}``
``contender_trial_built``
    ``{contender_id, repo_url, commit_sha, tree_sha, ok, outcome: ok|failed|unavailable,
    error (full builder message when failed), detail (operator-only, unavailable)}``
``enrollment_stake_rechecked``
    ``{floor, checked: [{contender_id, stake}], rejected: [contender_id]}``: the one
    alpha-stake re-check when finalization starts. Before the deadline the fetch check
    only warns (``stake_below_floor``).

A check belongs to the entry VERSION it ran on; a replaced entry starts again at
``pending``. The public summary carries only codes (``ok``/``pending``/a problem code
for the fetch, ``ok``/``pending``/``failed``/``disabled`` for the build): a builder
message can quote a private tree, so the full text is served only to the entrant
(``GET /v1/competitions/{id}/entry``) and the operator.

The public ``entry_commitment`` is ``sha256("<competition_id>\\n<hotkey>\\n<repo_url>\\n
<commit_sha>\\n<tree_sha>")`` (UTF-8, lowercase hex, no trailing newline): a miner
recomputes it to confirm that the listing shows its latest replacement.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable

from vidaio.competition import repository as repo
from vidaio.competition.manifest import CompetitionManifest

EVENT_ENTRY_REPLACED = repo.EVENT_ENTRY_REPLACED
EVENT_WITHDRAWN = repo.EVENT_WITHDRAWN
EVENT_PRECHECKED = "contender_prechecked"
EVENT_TRIAL_BUILT = "contender_trial_built"
#: The one alpha-stake re-check after enrollment closed (start of finalization).
EVENT_STAKE_RECHECKED = "enrollment_stake_rechecked"
#: Warning only before the deadline; the re-check then rejects such an entry.
STAKE_BELOW_FLOOR = "stake_below_floor"
#: ``cause`` of an unavailable trial build that failed fetching the entry's source.
CAUSE_CHECKOUT = "checkout"
WITHDRAWN_REASON = repo.WITHDRAWN_REASON

#: A LICENSE file at the repository root (any of these names).
LICENSE_FILES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING")
#: The third-party component list at the repository root.
LICENSES_MANIFEST = "LICENSES.md"

#: Problem codes a fetch check can report. ``check_unavailable`` and
#: ``fetch_timeout`` may be OUR failure and never blame the miner.
PROBLEM_CODES = (
    "repo_unreadable",
    "commit_missing",
    "tree_mismatch",
    "unsafe_tree",
    "no_dockerfile",
    "no_license",
    "no_licenses_md",
    "over_size_cap",
    "checkout_rejected",
    "fetch_timeout",
    "check_unavailable",
    STAKE_BELOW_FLOOR,
)

TRIAL_OK = "ok"
TRIAL_FAILED = "failed"
TRIAL_UNAVAILABLE = "unavailable"

#: Tracks whose rules require ``LICENSE`` and ``LICENSES.md`` at the root
#: (docs/MINING.md: frame interpolation and both upscaling scorers).
_LICENSED_TRACKS = frozenset({"interpolation", "upscaling"})


def licenses_required(manifest: CompetitionManifest) -> bool:
    """Whether this competition's rules require LICENSE and LICENSES.md."""
    return manifest.track in _LICENSED_TRACKS


def entry_commitment(
    competition_id: str, hotkey: str, repo_url: str, commit_sha: str, tree_sha: str
) -> str:
    """sha256 hex over ``competition_id\\nhotkey\\nrepo_url\\ncommit_sha\\ntree_sha``."""
    preimage = "\n".join((competition_id, hotkey, repo_url, commit_sha, tree_sha))
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def version_of(record: repo.ContenderRecord) -> tuple[str, str, str]:
    return (record.repo_url, record.commit_sha, record.tree_sha)


def _payload_version(payload: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (payload.get("repo_url"), payload.get("commit_sha"), payload.get("tree_sha"))


@dataclass
class EntryChecks:
    """Pre-deadline check state of one entry, for its CURRENT version."""

    #: latest ``contender_prechecked`` payload of the current version, plus ``at``
    precheck: dict[str, Any] | None = None
    #: latest ``contender_trial_built`` payload of the current version, plus ``at``
    trial: dict[str, Any] | None = None
    #: how many times the entry was replaced
    replacements: int = 0
    #: completed (ok/failed) trial builds of this entry, any version, with their time
    trial_times: list[datetime] = field(default_factory=list)

    def trial_builds_since(self, since: datetime) -> int:
        return sum(1 for at in self.trial_times if at >= since)


def _events(
    conn: sqlite3.Connection, competition_id: str, types: Iterable[str]
) -> list[sqlite3.Row]:
    names = tuple(types)
    marks = ",".join("?" for _ in names)
    return conn.execute(
        f"SELECT event_type, payload_json, created_at FROM events"
        f" WHERE competition_id = ? AND event_type IN ({marks}) ORDER BY event_id",
        (competition_id, *names),
    ).fetchall()


def entry_checks(
    conn: sqlite3.Connection,
    competition_id: str,
    contenders: Iterable[repo.ContenderRecord],
) -> dict[int, EntryChecks]:
    """Check state per contender id, read from the event log in one pass."""
    by_id = {c.contender_id: c for c in contenders}
    out = {cid: EntryChecks() for cid in by_id}
    for row in _events(
        conn,
        competition_id,
        (EVENT_PRECHECKED, EVENT_TRIAL_BUILT, EVENT_ENTRY_REPLACED),
    ):
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except ValueError:
            continue
        cid = payload.get("contender_id")
        if not isinstance(cid, int) or cid not in by_id:
            continue
        state = out[cid]
        at = repo.parse_ts(row["created_at"])
        if row["event_type"] == EVENT_ENTRY_REPLACED:
            state.replacements += 1
            continue
        current = _payload_version(payload) == version_of(by_id[cid])
        if row["event_type"] == EVENT_PRECHECKED:
            if current:
                state.precheck = {**payload, "at": row["created_at"]}
        else:  # EVENT_TRIAL_BUILT
            if payload.get("outcome") in (TRIAL_OK, TRIAL_FAILED) and at is not None:
                state.trial_times.append(at)
            if current:
                state.trial = {**payload, "at": row["created_at"]}
    return out


def last_trial_unavailable_at(
    conn: sqlite3.Connection, competition_id: str
) -> datetime | None:
    """When a trial build last could not run because of the runtime (OUR side),
    for the global back-off. A failure to fetch one entry's source
    (``cause == checkout``) backs off that entry only, so it is skipped here."""
    for row in reversed(_events(conn, competition_id, (EVENT_TRIAL_BUILT,))):
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except ValueError:
            continue
        if payload.get("cause") == CAUSE_CHECKOUT:
            continue
        if payload.get("outcome") == TRIAL_UNAVAILABLE:
            return repo.parse_ts(row["created_at"])
        return None
    return None


def fetch_summary(checks: EntryChecks | None) -> str:
    """``ok``, ``pending`` or the first problem code of the current version."""
    if checks is None or checks.precheck is None:
        return "pending"
    if checks.precheck.get("ok") is True:
        return "ok"
    problems = checks.precheck.get("problems") or []
    first = problems[0] if problems else "check_unavailable"
    return first if first in PROBLEM_CODES else "check_unavailable"


def fetch_passed(checks: EntryChecks | None) -> bool:
    """The current version fetched and inspected clean (a stake warning aside):
    what a trial build of it needs."""
    if checks is None or checks.precheck is None:
        return False
    problems = set(checks.precheck.get("problems") or [])
    return checks.precheck.get("ok") is True or problems == {STAKE_BELOW_FLOOR}


def build_summary(checks: EntryChecks | None, *, trial_enabled: bool) -> str:
    """``ok``, ``failed``, ``pending`` or ``disabled`` for the current version."""
    if not trial_enabled:
        return "disabled"
    if checks is None or checks.trial is None:
        return "pending"
    outcome = checks.trial.get("outcome")
    if outcome == TRIAL_OK:
        return "ok"
    if outcome == TRIAL_FAILED:
        return "failed"
    return "pending"  # unavailable: our side, retried


def checks_summary(checks: EntryChecks | None, *, trial_enabled: bool) -> dict[str, str]:
    return {
        "fetch": fetch_summary(checks),
        "build": build_summary(checks, trial_enabled=trial_enabled),
    }


def precheck_interval(
    now: datetime,
    deadline: datetime,
    *,
    interval_seconds: float,
    final_interval_seconds: float,
    final_window_seconds: float,
) -> timedelta:
    """Re-check cadence: every ``interval`` until the final window before the
    deadline, then every ``final_interval`` (also after the deadline)."""
    if now >= deadline - timedelta(seconds=final_window_seconds):
        return timedelta(seconds=final_interval_seconds)
    return timedelta(seconds=interval_seconds)


__all__ = [
    "EVENT_ENTRY_REPLACED",
    "EVENT_PRECHECKED",
    "EVENT_STAKE_RECHECKED",
    "EVENT_TRIAL_BUILT",
    "EVENT_WITHDRAWN",
    "EntryChecks",
    "LICENSES_MANIFEST",
    "LICENSE_FILES",
    "PROBLEM_CODES",
    "STAKE_BELOW_FLOOR",
    "TRIAL_FAILED",
    "TRIAL_OK",
    "TRIAL_UNAVAILABLE",
    "WITHDRAWN_REASON",
    "build_summary",
    "checks_summary",
    "entry_checks",
    "entry_commitment",
    "fetch_passed",
    "fetch_summary",
    "last_trial_unavailable_at",
    "licenses_required",
    "precheck_interval",
    "version_of",
]
