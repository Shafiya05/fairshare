import requests
import pandas as pd

def get_all(url, params=None, max_pages=3):
    """Get up to max_pages pages (100 items each) from a GitHub API address."""
    items = []
    for page in range(1, max_pages + 1):
        p = {"per_page": 100, "page": page}
        if params:
            p.update(params)
        r = requests.get(url, params=p)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        items.extend(data)
    return items

def is_bot(user):
    return user["type"] == "Bot" or user["login"].endswith("[bot]")

def collect(repo):
    base = f"https://api.github.com/repos/{repo}"
    rows = []

    # 1. Commits (skip merge commits and bots)
    for c in get_all(f"{base}/commits"):
        if len(c["parents"]) > 1:
            continue
        user = c["author"]
        if user and is_bot(user):
            continue
        name = user["login"] if user else c["commit"]["author"]["name"]
        rows.append({"person": name, "type": "commit",
                     "date": c["commit"]["author"]["date"]})

    # 2. Pull requests
    for p in get_all(f"{base}/pulls", {"state": "all"}):
        if p["user"] is None or is_bot(p["user"]):
            continue
        rows.append({"person": p["user"]["login"], "type": "pull_request",
                     "date": p["created_at"]})

    # 3. Issues (GitHub mixes PRs into issues, so skip those)
    for i in get_all(f"{base}/issues", {"state": "all"}):
        if "pull_request" in i or i["user"] is None or is_bot(i["user"]):
            continue
        rows.append({"person": i["user"]["login"], "type": "issue",
                     "date": i["created_at"]})

    # 4. Review comments and discussion comments
    for endpoint in ["pulls/comments", "issues/comments"]:
        for cm in get_all(f"{base}/{endpoint}", {"sort": "created", "direction": "desc"}):
            if cm["user"] is None or is_bot(cm["user"]):
                continue
            rows.append({"person": cm["user"]["login"], "type": "comment",
                         "date": cm["created_at"]})

    return pd.DataFrame(rows)