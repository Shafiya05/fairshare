"""
github_data.py – GitHub REST API collector for FairShare.

Fetches commits, pull-requests, issues and comments for a repository.
Preserves all existing fields (date, person, type) and adds `title`.
Supports authenticated requests via a GitHub personal access token.
"""

import requests
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_headers(token: str | None) -> dict:
    h = {"Accept": "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get_all(url: str, params: dict | None = None,
             headers: dict | None = None, max_pages: int = 5) -> list:
    """Paginate a GitHub API endpoint (up to max_pages × 100 items)."""
    items: list = []
    for page in range(1, max_pages + 1):
        p = {"per_page": 100, "page": page}
        if params:
            p.update(params)
        try:
            r = requests.get(url, params=p, headers=headers, timeout=20)
        except requests.exceptions.Timeout:
            raise RuntimeError("GitHub API request timed out.")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Could not reach GitHub. Check your internet connection.")

        if r.status_code == 401:
            raise RuntimeError("GitHub token is invalid or expired (HTTP 401).")
        if r.status_code == 403:
            msg = r.json().get("message", "")
            if "rate limit" in msg.lower():
                raise RuntimeError("GitHub API rate limit exceeded. Try again later.")
            raise RuntimeError(f"GitHub access forbidden (HTTP 403): {msg}")
        if r.status_code == 404:
            raise RuntimeError("Repository not found. Check the URL.")
        r.raise_for_status()

        data = r.json()
        if not data:
            break
        items.extend(data)
    return items


def _is_bot(user: dict) -> bool:
    return user.get("type") == "Bot" or user.get("login", "").endswith("[bot]")


# ---------------------------------------------------------------------------
# Public collector
# ---------------------------------------------------------------------------

def collect(repo: str, token: str | None = None) -> pd.DataFrame:
    """
    Collect activity for `repo` (format: "owner/name").

    Returns a DataFrame with columns:
        date     – ISO-8601 timestamp string
        person   – GitHub login
        type     – "commit" | "pull_request" | "issue" | "comment"
        title    – first line of message / PR title / issue title / ""
    """
    if not repo or "/" not in repo:
        raise ValueError(f"Invalid repository identifier: {repo!r}")

    base = f"https://api.github.com/repos/{repo}"
    headers = _build_headers(token)
    rows: list[dict] = []

    # 1. Commits – skip merges and bots
    for c in _get_all(f"{base}/commits", headers=headers):
        if len(c.get("parents", [])) > 1:          # merge commit
            continue
        user = c.get("author")
        if user and _is_bot(user):
            continue
        name = user["login"] if user else c["commit"]["author"]["name"]
        message = c["commit"]["message"]
        title = message.split("\n")[0].strip()      # first line only
        rows.append({"person": name, "type": "commit",
                     "date": c["commit"]["author"]["date"],
                     "title": title})

    # 2. Pull requests
    for p in _get_all(f"{base}/pulls", {"state": "all"}, headers=headers):
        if p.get("user") is None or _is_bot(p["user"]):
            continue
        rows.append({"person": p["user"]["login"], "type": "pull_request",
                     "date": p["created_at"],
                     "title": p.get("title", "")})

    # 3. Issues (GitHub mixes PRs into /issues, skip those)
    for i in _get_all(f"{base}/issues", {"state": "all"}, headers=headers):
        if "pull_request" in i:
            continue
        if i.get("user") is None or _is_bot(i["user"]):
            continue
        rows.append({"person": i["user"]["login"], "type": "issue",
                     "date": i["created_at"],
                     "title": i.get("title", "")})

    # 4. Review comments and issue comments
    for endpoint in ["pulls/comments", "issues/comments"]:
        for cm in _get_all(f"{base}/{endpoint}",
                           {"sort": "created", "direction": "desc"},
                           headers=headers):
            if cm.get("user") is None or _is_bot(cm["user"]):
                continue
            rows.append({"person": cm["user"]["login"], "type": "comment",
                         "date": cm["created_at"],
                         "title": ""})   # comments have no meaningful title

    if not rows:
        return pd.DataFrame(columns=["date", "person", "type", "title"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    # De-duplicate: keep one record per (person, type, date, title)
    df = df.drop_duplicates(subset=["person", "type", "date", "title"])
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    return df
