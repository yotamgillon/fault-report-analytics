"""Final scan of the repository and its git history.

Nothing in the files or the history may carry a local path, an email address (other than the two
allowed ones), a real name, or a domain term from the original project. Every check is strict.

This file and tests/test_readme.py are excluded from the scans because they necessarily contain
the words they search for.
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOREPLY = "340698105+yotamgillon@users.noreply.github.com"
ALLOWED_EMAILS = {NOREPLY, "noreply@anthropic.com"}
ALLOWED_AUTHOR = "yotamgillon"
EXCLUDE = [
    ":(exclude)tests/test_repo_scan.py",
    ":(exclude)tests/test_readme.py",
    ":(exclude)uv.lock",
]

# Aviation / UAV / military terms. Extended regular expressions (what `git grep -E` understands).
DOMAIN = [
    r"\buavs?\b", r"\bdrones?\b", r"aircraft", r"airframe", r"avionic", r"gimbal", r"propeller",
    r"landing gear", r"hard landing", r"payload", r"\bflight", r"\bflying\b", r"hangar",
    r"sortie", r"\bmission", r"waypoint", r"geofenc", r"return to home",
    r"ground (control|ctrl|station)", r"sensor pod", r"\bgps\b", r"compass", r"\bimu\b",
    r"satellite", r"lidar", r"barometer", r"magnetometer", r"altitude", r"military", r"\barmy\b",
    r"(^|[^-A-Za-z0-9_])idf([^-A-Za-z0-9_]|$)",  # not the "IDF" inside "TF-IDF"
    r"reserve duty", r"reservist", r"squadron",
]  # fmt: skip
LOCAL_PATHS = [r"/Users/", r"/home/[A-Za-z]", r"[A-Za-z]:\\", r"/private/var", r"/var/folders",
               r"/tmp/claude"]  # fmt: skip
OLD_IDENTITY = [r"yotamg150", r"gmail\.com", r"icloud\.com", r"yotamgillon"]
EMAIL = r"[A-Za-z0-9._+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]*[A-Za-z]"
PERSONAL = [*LOCAL_PATHS, *OLD_IDENTITY, EMAIL]


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=check)


@pytest.fixture(scope="module", autouse=True)
def inside_a_git_checkout():
    if git("rev-parse", "--git-dir", check=False).returncode != 0:
        pytest.skip("not a git checkout")


@pytest.fixture(scope="module")
def full_history() -> list[str]:
    """Every revision. A shallow clone cannot be scanned, and must not pass by accident."""
    if git("rev-parse", "--is-shallow-repository").stdout.strip() == "true":
        pytest.skip("shallow clone: the history scan needs full history (CI uses fetch-depth: 0)")
    return git("rev-list", "--all").stdout.split()


def grep_files(patterns: list[str], revisions: list[str] | None = None) -> list[str]:
    """`file:line:text` hits for any pattern in tracked files (or in the given revisions)."""
    result = git("grep", "-I", "-n", "-i", "-E", "|".join(patterns), *(revisions or []), "--", ".",
                 *EXCLUDE, check=False)  # fmt: skip
    assert result.returncode in (0, 1), result.stderr
    return [line for line in result.stdout.splitlines() if line]


def unexpected_emails(text: str) -> list[str]:
    return [e for e in re.findall(EMAIL, text) if e not in ALLOWED_EMAILS]


def personal_hits(lines: list[str]) -> list[str]:
    """Drop the allowed addresses before judging an e-mail-like hit."""
    kept = []
    for line in lines:
        text = line.split(":", 3)[-1]
        only_allowed_email = not unexpected_emails(text) and not any(
            re.search(p, text, re.I) for p in [*LOCAL_PATHS, *OLD_IDENTITY]
        )
        if not only_allowed_email:
            kept.append(line)
    return kept


# --- the files as they are now ----------------------------------------------------------------
def test_files_have_no_domain_terms():
    assert grep_files(DOMAIN) == []


def test_files_have_no_local_paths_emails_or_names():
    assert personal_hits(grep_files(PERSONAL)) == []


# --- the history ------------------------------------------------------------------------------
def test_every_commit_has_exactly_one_identity():
    out = git("log", "--all", "--format=%an%x09%ae%x09%cn%x09%ce").stdout.splitlines()
    assert out and set(out) == {f"{ALLOWED_AUTHOR}\t{NOREPLY}\t{ALLOWED_AUTHOR}\t{NOREPLY}"}


def test_commit_messages_carry_only_the_allowed_trailers():
    messages = git("log", "--all", "--format=%B%x00").stdout
    assert not unexpected_emails(messages)
    for pattern in [*LOCAL_PATHS, *OLD_IDENTITY, *DOMAIN]:
        assert not re.search(pattern, messages, re.I | re.M), pattern
    names = re.findall(r"^Co-Authored-By: (.+?) <", messages, re.M)
    assert names and all(n.startswith("Claude") for n in names)  # an AI assistant, not a person


def test_history_has_no_local_paths_emails_or_names(full_history):
    assert personal_hits(grep_files(PERSONAL, full_history)) == []


def test_history_has_no_domain_terms(full_history):
    assert grep_files(DOMAIN, full_history) == []


# --- the patterns themselves can fail ---------------------------------------------------------
@pytest.mark.parametrize(
    ("patterns", "sample"),
    [
        (DOMAIN, "the gimbal reset"),
        (DOMAIN, "return to home failed"),
        (DOMAIN, "reserve duty"),
        (LOCAL_PATHS, "see /Users/someone/project"),
        (LOCAL_PATHS, r"C:\Users\someone"),
        (OLD_IDENTITY, "an address at gmail.com"),
        ([EMAIL], "contact someone@example.org"),
    ],
)
def test_the_patterns_catch_what_they_are_meant_to(patterns, sample):
    assert any(re.search(p, sample, re.I) for p in patterns)


def test_the_patterns_do_not_flag_ordinary_text():
    ordinary = "a TF-IDF model, the Operator UI, a power harness and the alignment module"
    assert not any(re.search(p, ordinary, re.I) for p in [*DOMAIN, *LOCAL_PATHS, *OLD_IDENTITY])
    assert unexpected_emails(f"Co-Authored-By: Claude <noreply@anthropic.com> {NOREPLY}") == []
