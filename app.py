"""
app.py – FairShare: Team Contribution Intelligence
Streamlit dashboard for GitHub repository activity analysis.
"""

import html
import io
import json
import random
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import altair as alt
import pandas as pd
import streamlit as st

from github_data import collect
from summarizer import summarize, resolve_model, is_model_404, PREFERRED_MODELS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FairShare",
    page_icon="📊",
    layout="wide",
)

INACTIVE_DAYS = 30
TOP_N = 10                      # top-N contributors for statistics
TOP_SUMMARY = 5                 # contributors to summarise with AI
SUMMARY_TTL = 3600              # seconds to cache successful AI summaries
SUMMARY_ERR_TTL = 60            # seconds to cache failed AI summaries
ACTIVITY_TYPES = ["commit", "pull_request", "issue", "comment"]
ACTIVITY_COLORS = ["#2563EB", "#16A34A", "#D97706", "#9333EA"]


# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------
def _get_secret(key: str, default: str = "") -> str:
    """Read from st.secrets first, return default on any error."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


# REFRESH_INTERVAL may be overridden by REFRESH_SECONDS in secrets (e.g. 60 for demo)
def _refresh_interval() -> int:
    raw = _get_secret("REFRESH_SECONDS", "")
    try:
        return max(10, int(raw))
    except (ValueError, TypeError):
        return 300


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "live_mode": False,
        "last_refresh": 0.0,
        "df": None,
        "repo": "",
        "identity_map": {},         # {display_name: [login1, login2, ...]}
        "ai_summaries": {},         # {person: (summary_text, timestamp, is_error)}
        "ai_model": None,
        "analysis_done": False,
        "is_sample": False,
        "repo_url": "https://github.com/Shafiya05/fairshare",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Utility: parse a GitHub URL to owner/repo
# ---------------------------------------------------------------------------
def parse_repo(link: str) -> str:
    link = link.strip().rstrip("/")
    if not link:
        raise ValueError("Please enter a repository URL.")
    if "github.com/" not in link:
        raise ValueError("Not a valid GitHub URL. It must contain 'github.com/'.")
    parts = link.split("github.com/")[-1].split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError("URL must be in the form https://github.com/owner/repo")
    return f"{parts[0]}/{parts[1]}"


# ---------------------------------------------------------------------------
# GitHub data loading (cached per REFRESH_INTERVAL)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=_refresh_interval(), show_spinner=False)
def load_data(repo: str, token: str) -> pd.DataFrame:
    return collect(repo, token or None)


# ---------------------------------------------------------------------------
# Identity mapping helpers
# ---------------------------------------------------------------------------
def apply_identity_map(df: pd.DataFrame, identity_map: dict) -> pd.DataFrame:
    """
    Replace GitHub logins with display names according to identity_map.
    identity_map format: { "Display Name": ["login1", "login2"], ... }
    Original logins are preserved in an 'original_person' column.
    """
    if df is None or df.empty or not identity_map:
        return df
    df = df.copy()
    if "original_person" not in df.columns:
        df["original_person"] = df["person"]
    login_to_display = {}
    for display, logins in identity_map.items():
        for login in logins:
            login_to_display[login] = display
    df["person"] = df["original_person"].map(login_to_display).fillna(df["original_person"])
    return df


# ---------------------------------------------------------------------------
# AI summary cache helpers
# ---------------------------------------------------------------------------
def _summary_cache_ok(person: str) -> bool:
    """True if the cached summary (success or error) is still fresh."""
    cache = st.session_state.ai_summaries
    if person not in cache:
        return False
    text, ts, is_error = cache[person]
    ttl = SUMMARY_ERR_TTL if is_error else SUMMARY_TTL
    return (time.time() - ts) < ttl


def _get_cached_summary(person: str) -> str:
    text, _, _ = st.session_state.ai_summaries[person]
    return text


def _store_summary(person: str, text: str):
    is_error = text.startswith("Error:")
    st.session_state.ai_summaries[person] = (text, time.time(), is_error)


# ---------------------------------------------------------------------------
# CSV export helpers
# ---------------------------------------------------------------------------
def df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


def build_summary_df(top_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Add original usernames column to contributor summary."""
    id_map = st.session_state.identity_map
    rows = []
    for person in top_df.index:
        originals = []
        for display, logins in id_map.items():
            if display == person:
                originals = logins
                break
        orig_str = ", ".join(originals) if originals else person
        row = top_df.loc[person].to_dict()
        row["display_name"] = person
        row["original_usernames"] = orig_str
        rows.append(row)
    return pd.DataFrame(rows).set_index("display_name")


def build_inactive_df(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    last = df.groupby("person")["date"].max()
    inactive_rows = []
    for person in summary.index:
        d = int((now - last.get(person, now)).days)
        status = "INACTIVE" if d > INACTIVE_DAYS else "Active"
        inactive_rows.append({
            "contributor": person,
            "last_activity_date": last.get(person, pd.NaT),
            "days_since_last_activity": d,
            "status": status,
        })
    return pd.DataFrame(inactive_rows)


# ---------------------------------------------------------------------------
# Parallel AI summary generation
# ---------------------------------------------------------------------------
def _generate_summaries_parallel(top5, df, summary, gemini_key, ai_model):
    """
    Generate summaries for top-5 contributors in parallel.

    st.session_state is NOT thread-safe. We snapshot the cache as a plain
    dict on the main thread before entering the pool, use only that snapshot
    inside worker threads, and write results back to session_state on the
    main thread after each future completes.
    """
    # --- snapshot on main thread (safe) ---
    cache_snapshot: dict = dict(st.session_state.ai_summaries)
    now = time.time()

    def _is_cached(person: str) -> bool:
        if person not in cache_snapshot:
            return False
        _text, _ts, _is_err = cache_snapshot[person]
        ttl = SUMMARY_ERR_TTL if _is_err else SUMMARY_TTL
        return (now - _ts) < ttl

    def _task(person: str) -> tuple[str, str]:
        # Only plain dicts/values used here — no st.session_state access.
        if _is_cached(person):
            cached_text, _, _ = cache_snapshot[person]
            return person, cached_text
        person_counts = {
            k: int(v)
            for k, v in summary.loc[person].to_dict().items()
            if k != "total"
        }
        person_df = df[df["person"] == person].head(40)
        titles = (
            person_df["title"]
            .dropna()
            .loc[person_df["title"].str.strip() != ""]
            .tolist()[:15]
        ) if "title" in person_df.columns else []
        result = summarize(
            person=person,
            counts=person_counts,
            titles=titles,
            api_key=gemini_key,
            model=ai_model,
        )
        return person, result

    with ThreadPoolExecutor(max_workers=TOP_SUMMARY) as pool:
        futures = {pool.submit(_task, person): person for person in top5.index}
        for future in as_completed(futures):
            try:
                person, text = future.result()
            except Exception as exc:
                person = futures[future]
                text = f"Error: Unexpected error ({exc})."
            # Write back to session_state on the main thread (safe).
            _store_summary(person, text)


def _format_error_hint(message: str) -> str:
    """Append a plain-language hint to common GitHub/input error messages."""
    low = message.lower()
    if "404" in low or "not found" in low:
        return f"{message} — Check the URL and make sure the repository exists and is accessible."
    if "rate limit" in low or "403" in low or "429" in low:
        return f"{message} — Rate limit reached: try again later or configure a GITHUB_TOKEN."
    if "401" in low or "unauthorized" in low or "bad credentials" in low:
        return f"{message} — Check your GITHUB_TOKEN in .streamlit/secrets.toml."
    if "timed out" in low or "timeout" in low or "connection" in low:
        return f"{message} — Network issue: check your connection and try again."
    return message


def inject_css():
    """Inject theme-neutral custom CSS that adapts to both light and dark modes."""
    st.markdown(
        """
        <style>
        .stApp {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        /* Hero banner */
        .fs-hero {
            background: linear-gradient(135deg, #4f46e5 0%, #0d9488 100%);
            border-radius: 14px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.25rem;
            color: #ffffff;
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.18);
        }
        .fs-hero-title {
            font-size: 1.85rem;
            font-weight: 700;
            margin: 0 0 0.3rem 0;
            letter-spacing: -0.02em;
            color: #ffffff;
        }
        .fs-hero-sub {
            font-size: 0.98rem;
            margin: 0 0 0.85rem 0;
            color: rgba(255, 255, 255, 0.9);
        }
        .fs-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .fs-badge {
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            background: rgba(255, 255, 255, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.28);
            color: #ffffff;
        }
        /* Metric cards */
        .fs-metric-card {
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            height: 100%;
        }
        .fs-metric-top {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            opacity: 0.72;
            margin-bottom: 0.35rem;
        }
        .fs-metric-value {
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.25;
            word-break: break-word;
        }
        .fs-metric-sub {
            font-size: 0.78rem;
            opacity: 0.7;
            margin-top: 0.25rem;
        }
        /* Leaderboard cards */
        .fs-leader-card {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.6rem;
        }
        .fs-rank {
            font-size: 0.9rem;
            font-weight: 700;
            min-width: 1.9rem;
            opacity: 0.75;
        }
        .fs-avatar {
            position: relative;
            width: 38px;
            height: 38px;
            min-width: 38px;
            border-radius: 50%;
            background: rgba(79, 70, 229, 0.22);
            border: 1px solid rgba(128, 128, 128, 0.25);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.82rem;
            overflow: hidden;
        }
        .fs-avatar-img {
            position: absolute;
            inset: 0;
            background-size: cover;
            background-position: center;
            border-radius: 50%;
        }
        .fs-leader-body {
            flex: 1;
            min-width: 0;
        }
        .fs-leader-head {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 0.5rem;
            margin-bottom: 0.35rem;
        }
        .fs-leader-name {
            font-weight: 600;
            font-size: 0.95rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .fs-leader-count {
            font-size: 0.82rem;
            opacity: 0.8;
            white-space: nowrap;
        }
        .fs-progress-track {
            width: 100%;
            height: 7px;
            border-radius: 999px;
            background: rgba(128, 128, 128, 0.18);
            overflow: hidden;
        }
        .fs-progress-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #4f46e5, #0d9488);
        }
        /* Inactive member cards & badges */
        .fs-inactive-card {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 10px;
            padding: 0.7rem 1rem;
            margin-bottom: 0.55rem;
        }
        .fs-status-badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            white-space: nowrap;
        }
        .fs-status-amber {
            background: rgba(217, 119, 6, 0.16);
            border: 1px solid rgba(217, 119, 6, 0.45);
            color: #d97706;
        }
        .fs-status-red {
            background: rgba(220, 38, 38, 0.16);
            border: 1px solid rgba(220, 38, 38, 0.45);
            color: #ef4444;
        }
        /* AI summary cards */
        .fs-ai-card {
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 12px;
            padding: 0.95rem 1.1rem;
            margin-bottom: 0.75rem;
        }
        .fs-ai-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            margin-bottom: 0.45rem;
        }
        .fs-ai-name {
            font-weight: 600;
            font-size: 0.98rem;
        }
        .fs-ai-pill {
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            padding: 0.15rem 0.5rem;
            border-radius: 999px;
            background: rgba(79, 70, 229, 0.18);
            border: 1px solid rgba(79, 70, 229, 0.4);
        }
        .fs-ai-body {
            font-size: 0.92rem;
            line-height: 1.5;
            opacity: 0.92;
        }
        /* Home hook: animated gradient + contribution grid */
        @keyframes fs-gradient-shift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes fs-cell-pulse {
            0%, 100% { opacity: 0.72; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.08); }
        }
        .fs-hook {
            background: linear-gradient(125deg, rgba(79, 70, 229, 0.22), rgba(13, 148, 136, 0.22), rgba(37, 99, 235, 0.18));
            background-size: 200% 200%;
            animation: fs-gradient-shift 14s ease infinite;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 14px;
            padding: 1.35rem 1.5rem;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 1.25rem;
        }
        .fs-hook-text {
            flex: 1 1 280px;
        }
        .fs-hook-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin: 0 0 0.3rem 0;
            letter-spacing: -0.02em;
        }
        .fs-hook-sub {
            font-size: 0.94rem;
            opacity: 0.85;
            margin: 0;
            line-height: 1.45;
        }
        .fs-contrib-grid {
            display: grid;
            grid-template-columns: repeat(20, 10px);
            grid-template-rows: repeat(7, 10px);
            gap: 3px;
        }
        .fs-cell {
            width: 10px;
            height: 10px;
            border-radius: 2px;
            animation: fs-cell-pulse 4s ease-in-out infinite;
        }
        .fs-c0 { background: rgba(128, 128, 128, 0.16); }
        .fs-c1 { background: rgba(13, 148, 136, 0.32); }
        .fs-c2 { background: rgba(13, 148, 136, 0.54); }
        .fs-c3 { background: rgba(13, 148, 136, 0.78); }
        .fs-c4 { background: rgba(79, 70, 229, 0.85); }
        @media (prefers-reduced-motion: reduce) {
            .fs-hook, .fs-cell {
                animation: none !important;
            }
        }
        /* Preview card */
        .fs-preview-card {
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 12px;
            padding: 1rem 1.15rem;
            margin-bottom: 0.9rem;
        }
        .fs-preview-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }
        .fs-preview-title {
            font-size: 1rem;
            font-weight: 700;
        }
        .fs-sample-pill {
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.18rem 0.6rem;
            border-radius: 999px;
            background: rgba(217, 119, 6, 0.16);
            border: 1px solid rgba(217, 119, 6, 0.45);
            color: #d97706;
        }
        .fs-home-note {
            font-size: 0.84rem;
            opacity: 0.75;
            margin-top: 0.9rem;
        }
        /* Clearer, larger tab labels */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.98rem;
            font-weight: 600;
            padding: 0.55rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_card_html(icon: str, label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="fs-metric-sub">{html.escape(sub)}</div>' if sub else ""
    return (
        f'<div class="fs-metric-card">'
        f'<div class="fs-metric-top"><span>{icon}</span><span>{html.escape(label)}</span></div>'
        f'<div class="fs-metric-value">{html.escape(value)}</div>'
        f'{sub_html}'
        f'</div>'
    )


def _initials(name: str) -> str:
    parts = [p for p in name.replace("-", " ").replace("_", " ").split() if p]
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    return name[:2].upper() if name else "?"


def _github_login_for(person: str, df: pd.DataFrame) -> str:
    if "original_person" in df.columns:
        matches = df.loc[df["person"] == person, "original_person"]
        if not matches.empty:
            return str(matches.iloc[0])
    return person


def _build_contribution_bar_chart(summary_df: pd.DataFrame) -> alt.Chart:
    chart_src = summary_df.drop(columns="total", errors="ignore").reindex(
        columns=ACTIVITY_TYPES, fill_value=0
    )
    long_df = (
        chart_src.reset_index()
        .melt(id_vars="person", var_name="type", value_name="count")
    )
    return (
        alt.Chart(long_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "person:N",
                sort=list(summary_df.index),
                title="Contributor",
                axis=alt.Axis(labelAngle=-25),
            ),
            y=alt.Y("count:Q", title="Activities"),
            color=alt.Color(
                "type:N",
                scale=alt.Scale(domain=ACTIVITY_TYPES, range=ACTIVITY_COLORS),
                legend=alt.Legend(title="Activity Type", orient="top"),
            ),
            order=alt.Order("type:N", sort="ascending"),
            tooltip=[
                alt.Tooltip("person:N", title="Contributor"),
                alt.Tooltip("type:N", title="Type"),
                alt.Tooltip("count:Q", title="Activities"),
            ],
        )
        .properties(height=340, background="transparent")
        .configure_axis(gridOpacity=0.18)
        .configure_view(stroke=None)
    )


def _build_weekly_line_chart(weekly_df: pd.DataFrame) -> alt.Chart:
    src = weekly_df.reindex(columns=ACTIVITY_TYPES, fill_value=0).reset_index()
    date_col = src.columns[0]
    long_df = src.melt(id_vars=date_col, var_name="type", value_name="count")
    return (
        alt.Chart(long_df)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X(f"{date_col}:T", title="Week"),
            y=alt.Y("count:Q", title="Activities"),
            color=alt.Color(
                "type:N",
                scale=alt.Scale(domain=ACTIVITY_TYPES, range=ACTIVITY_COLORS),
                legend=alt.Legend(title="Activity Type", orient="top"),
            ),
            tooltip=[
                alt.Tooltip(f"{date_col}:T", title="Week"),
                alt.Tooltip("type:N", title="Type"),
                alt.Tooltip("count:Q", title="Activities"),
            ],
        )
        .properties(height=340, background="transparent")
        .configure_axis(gridOpacity=0.18)
        .configure_view(stroke=None)
    )


inject_css()


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.header("⚙️ Settings")

    REFRESH_INTERVAL = _refresh_interval()

    # Live mode
    live_mode = st.checkbox(
        f"🔴 Live Mode (every {REFRESH_INTERVAL}s)",
        value=st.session_state.live_mode,
        help="Automatically re-fetch repository activity on the configured interval.",
    )
    st.session_state.live_mode = live_mode

    if live_mode and st.session_state.last_refresh > 0:
        last_ts = datetime.datetime.fromtimestamp(
            st.session_state.last_refresh
        ).strftime("%H:%M:%S")
        st.caption(f"🕐 Last updated at {last_ts}")

    st.divider()

    # Identity mapping management
    st.subheader("👤 Identity Mapping")
    st.caption("Merge multiple GitHub usernames into one contributor identity.")

    id_map = st.session_state.identity_map

    # Add new mapping
    with st.expander("➕ Add / Edit Mapping"):
        display_name = st.text_input("Display name", key="new_display")
        logins_raw = st.text_input(
            "GitHub usernames (comma-separated)",
            key="new_logins",
            help="e.g.  alice, alice-work, alice2",
        )
        if st.button("Save mapping", key="save_map"):
            if display_name and logins_raw:
                logins = [l.strip() for l in logins_raw.split(",") if l.strip()]
                id_map[display_name] = logins
                st.session_state.identity_map = id_map
                st.session_state.df = None
                st.success(f"Saved: {display_name} → {logins}")
            else:
                st.warning("Enter both a display name and at least one username.")

    # Show existing mappings with delete option
    if id_map:
        st.write("**Current mappings:**")
        for name, logins in list(id_map.items()):
            col_a, col_b = st.columns([3, 1])
            col_a.write(f"**{name}** → {', '.join(logins)}")
            if col_b.button("✕", key=f"del_{name}"):
                del id_map[name]
                st.session_state.identity_map = id_map
                st.session_state.df = None
                st.rerun()
    else:
        st.caption("No mappings configured.")

    st.divider()
    st.caption("FairShare v2.0 · Streamlit 1.67")


# ===========================================================================
# MAIN AREA
# ===========================================================================
st.markdown(
    """
    <div class="fs-hero">
        <div class="fs-hero-title">📊 FairShare</div>
        <div class="fs-hero-sub">Team contribution intelligence powered by real GitHub repository activity.</div>
        <div class="fs-badges">
            <span class="fs-badge">GitHub API</span>
            <span class="fs-badge">Gemini AI</span>
            <span class="fs-badge">Live</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

def _set_example_url(repo_slug: str):
    """Fill the URL input box with an example repository without auto-analysing."""
    st.session_state["repo_url"] = f"https://github.com/{repo_slug}"


def _build_sample_df() -> pd.DataFrame:
    """Return a deterministic sample DataFrame matching real repository columns."""
    rng = random.Random(42)
    now = pd.Timestamp.now(tz="UTC")
    specs = [
        ("Aarav", 52, 0, 21, [
            "Implement OAuth session refresh",
            "Refactor API retry helper",
            "Add unit tests for parser",
            "Review PR #42 edge cases",
        ]),
        ("Meera", 27, 1, 24, [
            "Design responsive dashboard cards",
            "Fix dark mode contrast",
            "Update contributor chart tooltips",
            "Review layout PR #38",
        ]),
        ("Kabir", 14, 2, 28, [
            "Add CSV export headers",
            "Document setup steps in README",
            "Investigate rate limit handling",
            "Comment on issue #19",
        ]),
        ("Zoya", 7, 45, 58, [
            "Initial project scaffolding",
            "Configure CI workflow",
            "Open issue #4 for test fixtures",
            "Comment on architecture notes",
        ]),
    ]
    rows = []
    for person, count, min_days_ago, max_days_ago, titles in specs:
        for i in range(count):
            days_ago = min_days_ago if i == 0 else rng.randint(min_days_ago, max_days_ago)
            hours_offset = rng.randint(0, 20)
            ts = now - pd.Timedelta(days=days_ago, hours=hours_offset)
            act_type = rng.choices(
                ACTIVITY_TYPES,
                weights=[0.5, 0.2, 0.15, 0.15],
                k=1,
            )[0]
            rows.append({
                "date": ts,
                "person": person,
                "type": act_type,
                "title": titles[i % len(titles)],
            })
    return pd.DataFrame(rows, columns=["date", "person", "type", "title"])


def _build_contrib_grid_html() -> str:
    """Build a decorative 7x20 GitHub-style contribution grid with staggered pulse."""
    rng = random.Random(42)
    cells = []
    for col in range(20):
        for row in range(7):
            level = rng.choices([0, 1, 2, 3, 4], weights=[35, 25, 20, 13, 7], k=1)[0]
            delay = ((col + row) % 10) * 0.35
            cells.append(
                f'<div class="fs-cell fs-c{level}" style="animation-delay: {delay:.2f}s;"></div>'
            )
    return f'<div class="fs-contrib-grid" aria-hidden="true">{"".join(cells)}</div>'


# ---------------------------------------------------------------------------
# Repository input row
# ---------------------------------------------------------------------------
col_url, col_btn, col_clear = st.columns([5, 1.5, 1], vertical_alignment="bottom")
url = col_url.text_input(
    "🔗 GitHub Repository URL",
    key="repo_url",
    placeholder="https://github.com/owner/repo",
)
analyse_clicked = col_btn.button("🔍 Analyse", type="primary", use_container_width=True)
if col_clear.button("🗑️ Clear", type="secondary", use_container_width=True):
    st.session_state.df = None
    st.session_state.analysis_done = False
    st.session_state.is_sample = False
    st.session_state.ai_summaries = {}
    st.rerun()

# ---------------------------------------------------------------------------
# Credentials (read once at top level)
# ---------------------------------------------------------------------------
token = _get_secret("GITHUB_TOKEN")
gemini_key = _get_secret("GEMINI_API_KEY")
gemini_model_cfg = _get_secret("GEMINI_MODEL", "")

# ---------------------------------------------------------------------------
# Manual Analyse button
# ---------------------------------------------------------------------------
if analyse_clicked:
    try:
        repo_id = parse_repo(url)
    except ValueError as e:
        st.error(f"⚠️ {e} Check the URL format (e.g. https://github.com/owner/repo).")
        st.stop()

    with st.spinner(f"Fetching activity for **{repo_id}**…"):
        try:
            raw_df = load_data(repo_id, token)
        except RuntimeError as e:
            st.error(f"⚠️ {_format_error_hint(str(e))}")
            st.stop()
        except Exception as e:
            st.error(f"⚠️ {_format_error_hint(f'Unexpected error fetching data: {e}')}")
            st.stop()

    if raw_df is None or raw_df.empty:
        st.info(
            "📭 This repository has no recorded commits, pull requests, issues, or comments yet. "
            "Check the URL or try a repository with recent activity."
        )
        st.stop()

    df = apply_identity_map(raw_df, st.session_state.identity_map)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    st.session_state.df = df
    st.session_state.repo = repo_id
    st.session_state.analysis_done = True
    st.session_state.is_sample = False
    st.session_state.last_refresh = time.time()

# ---------------------------------------------------------------------------
# Live-mode: countdown fragment + auto-refresh logic
# ---------------------------------------------------------------------------
if live_mode and st.session_state.analysis_done and not st.session_state.get("is_sample", False):
    REFRESH_INTERVAL = _refresh_interval()

    elapsed = time.time() - st.session_state.last_refresh

    if elapsed >= REFRESH_INTERVAL:
        # Time to refresh: re-fetch data, never set df to None
        repo_id = st.session_state.repo
        st.cache_data.clear()
        try:
            raw_df = load_data(repo_id, token)
            if raw_df is not None and not raw_df.empty:
                df_new = apply_identity_map(raw_df, st.session_state.identity_map)
                df_new["date"] = pd.to_datetime(df_new["date"], utc=True)
                st.session_state.df = df_new
        except Exception as exc:
            # Keep old data visible; show a warning (rendered below in dashboard)
            st.session_state["_refresh_error"] = str(exc)
        else:
            st.session_state.pop("_refresh_error", None)
        st.session_state.last_refresh = time.time()
        st.rerun()

    else:
        # Show a countdown that updates every 10 seconds (not every second)
        @st.fragment(run_every=10)
        def _live_countdown():
            _elapsed = time.time() - st.session_state.last_refresh
            _remaining = max(0, _refresh_interval() - _elapsed)
            st.sidebar.progress(
                min(1.0, _elapsed / _refresh_interval()),
                text=f"🔴 Live – next refresh in {int(_remaining)}s",
            )
            if _elapsed >= _refresh_interval():
                st.rerun()

        _live_countdown()

# ---------------------------------------------------------------------------
# Render dashboard if data is available
# ---------------------------------------------------------------------------
if st.session_state.analysis_done and st.session_state.df is not None:
    df = st.session_state.df
    repo_id = st.session_state.repo

    # Show auto-refresh error banner if present
    if st.session_state.get("_refresh_error"):
        st.warning(
            f"⚠️ Auto-refresh failed (showing previous data): "
            f"{_format_error_hint(st.session_state['_refresh_error'])}"
        )

    if live_mode and st.session_state.last_refresh > 0 and not st.session_state.get("is_sample", False):
        last_ts = datetime.datetime.fromtimestamp(
            st.session_state.last_refresh
        ).strftime("%H:%M:%S")
        st.caption(f"🔴 Live Mode active · Last updated at {last_ts}")

    if st.session_state.get("is_sample"):
        st.info(
            "🧪 **Sample data** — Viewing a built-in demo team (`sample/team-project`). "
            "Paste a GitHub repository URL above and click **Analyse** to inspect a real repo."
        )

    # Pre-compute contributor summary & weekly timeline
    summary = pd.crosstab(df["person"], df["type"])
    summary = summary.reindex(columns=ACTIVITY_TYPES, fill_value=0)
    summary["total"] = summary.sum(axis=1)
    summary = summary.sort_values("total", ascending=False).head(TOP_N)

    weekly = (
        df.set_index("date")
        .groupby([pd.Grouper(freq="W"), "type"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ACTIVITY_TYPES, fill_value=0)
    )

    inactive_df = build_inactive_df(df, summary)

    # --- Top KPI metric row (5 cards) ---
    top_person = str(summary.index[0]) if not summary.empty else "—"
    top_total = int(summary.iloc[0]["total"]) if not summary.empty else 0
    date_range = (
        f"{df['date'].min().date()} → {df['date'].max().date()}"
        if not df.empty
        else "—"
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.markdown(_metric_card_html("📁", "Repository", repo_id), unsafe_allow_html=True)
    m2.markdown(_metric_card_html("📊", "Total Activities", f"{len(df):,}"), unsafe_allow_html=True)
    m3.markdown(_metric_card_html("👥", "Contributors", str(df["person"].nunique())), unsafe_allow_html=True)
    m4.markdown(_metric_card_html("📅", "Date Range", date_range), unsafe_allow_html=True)
    m5.markdown(
        _metric_card_html("🏆", "Top Contributor", top_person, f"{top_total:,} activities"),
        unsafe_allow_html=True,
    )

    st.divider()

    # --- Tabbed results ---
    tab_overview, tab_timeline, tab_members, tab_ai, tab_reports = st.tabs([
        "📊 Overview",
        "📅 Timeline",
        "👥 Members",
        "🤖 AI Summaries",
        "📥 Reports",
    ])

    # -----------------------------------------------------------------------
    # Tab 1: Overview (leaderboard + Altair bar chart + raw numbers)
    # -----------------------------------------------------------------------
    with tab_overview:
        st.subheader("🏅 Top Contributor Leaderboard")
        st.caption(
            f"Ranks the top {TOP_N} contributors by their share of all {len(df):,} recorded activities."
        )
        total_all = max(1, len(df))
        lb_cols = st.columns(2)
        for idx, person in enumerate(summary.index):
            person_total = int(summary.loc[person, "total"])
            pct = min(100.0, (person_total / total_all) * 100.0)
            login = _github_login_for(person, df)
            avatar_url = f"https://github.com/{html.escape(login)}.png"
            initials = html.escape(_initials(person))
            card_html = (
                f'<div class="fs-leader-card">'
                f'<div class="fs-rank">#{idx + 1}</div>'
                f'<div class="fs-avatar"><span>{initials}</span>'
                f'<div class="fs-avatar-img" style="background-image: url(\'{avatar_url}\');"></div></div>'
                f'<div class="fs-leader-body">'
                f'<div class="fs-leader-head">'
                f'<span class="fs-leader-name">{html.escape(str(person))}</span>'
                f'<span class="fs-leader-count">{person_total:,} ({pct:.1f}%)</span>'
                f'</div>'
                f'<div class="fs-progress-track">'
                f'<div class="fs-progress-fill" style="width: {pct:.1f}%;"></div>'
                f'</div></div></div>'
            )
            lb_cols[idx % 2].markdown(card_html, unsafe_allow_html=True)

        st.divider()
        st.subheader("📊 Contribution Split & Statistics")
        st.caption(
            f"Shows the top {TOP_N} contributors broken down by commits, pull requests, "
            "issues, and comments."
        )
        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            if not summary.empty:
                st.altair_chart(_build_contribution_bar_chart(summary), use_container_width=True)
        with col_table:
            st.dataframe(summary, use_container_width=True)

    # -----------------------------------------------------------------------
    # Tab 2: Timeline (Altair weekly chart + table side-by-side)
    # -----------------------------------------------------------------------
    with tab_timeline:
        st.subheader("📅 Weekly Activity Timeline")
        st.caption(
            "Tracks how commits, pull requests, issues, and comments change week by week."
        )
        col_wchart, col_wtable = st.columns([3, 2])
        with col_wchart:
            if not weekly.empty:
                st.altair_chart(_build_weekly_line_chart(weekly), use_container_width=True)
        with col_wtable:
            st.dataframe(weekly, use_container_width=True)

    # -----------------------------------------------------------------------
    # Tab 3: Members (inactive alerts + identity mapping result)
    # -----------------------------------------------------------------------
    with tab_members:
        st.subheader(f"🚨 Inactive Members ({INACTIVE_DAYS}-day threshold)")
        st.caption(
            f"Checks contributors with no activity in the last {INACTIVE_DAYS} days (up to the top {TOP_N})."
        )

        inactive_only = inactive_df[inactive_df["status"] == "INACTIVE"]
        if inactive_only.empty:
            n_checked = len(summary)
            noun = "contributor has" if n_checked == 1 else "contributors have"
            st.success(f"✅ All {n_checked} {noun} been active in the last {INACTIVE_DAYS} days.")
        else:
            for _, row in inactive_only.iterrows():
                days_ago = int(row["days_since_last_activity"])
                last_date = (
                    str(row["last_activity_date"].date())
                    if pd.notna(row["last_activity_date"])
                    else "unknown"
                )
                badge_cls = "fs-status-red" if days_ago >= 60 else "fs-status-amber"
                badge_label = f"{days_ago}d inactive"
                st.markdown(
                    f'<div class="fs-inactive-card">'
                    f'<div><strong>{html.escape(str(row["contributor"]))}</strong> '
                    f'<span style="opacity:0.75;font-size:0.85rem;">· last active {html.escape(last_date)}</span></div>'
                    f'<span class="fs-status-badge {badge_cls}">{html.escape(badge_label)}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        st.subheader("👤 Identity Mapping Result")
        st.caption(
            "Shows GitHub usernames merged under a single contributor name using the sidebar rules."
        )
        if st.session_state.identity_map:
            map_rows = []
            for display, logins in st.session_state.identity_map.items():
                matched = int(df[df["person"] == display].shape[0])
                map_rows.append({
                    "Display Name": display,
                    "Mapped Usernames": ", ".join(logins),
                    "Matched Activities": matched,
                })
            st.dataframe(pd.DataFrame(map_rows), use_container_width=True, hide_index=True)
        else:
            st.info(
                "ℹ️ No identity mappings configured. Use **Identity Mapping** in the sidebar to merge duplicate usernames."
            )

    # -----------------------------------------------------------------------
    # Tab 4: AI Summaries
    # -----------------------------------------------------------------------
    with tab_ai:
        st.subheader("🤖 AI Contributor Summaries")
        st.caption(
            f"Generates concise two-sentence summaries for the top {TOP_SUMMARY} contributors based on their recorded activity."
        )

        if not gemini_key:
            st.info(
                "ℹ️ Gemini API key not configured. "
                "Add `GEMINI_API_KEY` to `.streamlit/secrets.toml` to enable AI summaries."
            )
        else:
            # Always normalise the cached value (guards against stale mixed-case
            # values stored before the lower-case fix was deployed).
            if st.session_state.ai_model is not None:
                st.session_state.ai_model = st.session_state.ai_model.strip().lower()

            if st.session_state.ai_model is None:
                with st.spinner("Verifying Gemini model availability…"):
                    resolved = resolve_model(gemini_key, gemini_model_cfg or None)
                st.session_state.ai_model = resolved

            ai_model = st.session_state.ai_model

            if not ai_model:
                st.error("No suitable Gemini model found for your API key.")
            else:
                st.caption(f"Model: `{ai_model}` · Summaries cached for 1 hour (errors 60s)")

                top5 = summary.head(TOP_SUMMARY)

                # Generate all summaries in parallel before rendering cards
                needs_generation = [p for p in top5.index if not _summary_cache_ok(p)]
                if needs_generation:
                    with st.spinner(f"Generating {len(needs_generation)} AI summary/summaries…"):
                        _generate_summaries_parallel(top5, df, summary, gemini_key, ai_model)

                generated = [
                    p for p in top5.index
                    if p in st.session_state.ai_summaries
                ]
                all_404 = bool(generated) and all(
                    is_model_404(st.session_state.ai_summaries[p][0])
                    for p in generated
                )

                if all_404:
                    st.warning(
                        "AI summaries are unavailable: check GEMINI_MODEL in secrets.toml"
                    )
                else:
                    for person in top5.index:
                        text = (
                            _get_cached_summary(person)
                            if _summary_cache_ok(person)
                            else "Error: Summary unavailable."
                        )
                        if is_model_404(text):
                            text = "Error: Gemini model not found – check GEMINI_MODEL in secrets.toml."
                        total_c = int(summary.loc[person, "total"])
                        if text.startswith("Error:"):
                            st.warning(f"**{person}** — {text}")
                        else:
                            card_html = (
                                f'<div class="fs-ai-card">'
                                f'<div class="fs-ai-head">'
                                f'<span class="fs-ai-name">🧑‍💻 {html.escape(str(person))} '
                                f'<span style="opacity:0.65;font-weight:400;font-size:0.85rem;">({total_c:,} contributions)</span></span>'
                                f'<span class="fs-ai-pill">AI</span>'
                                f'</div>'
                                f'<div class="fs-ai-body">{html.escape(text)}</div>'
                                f'</div>'
                            )
                            st.markdown(card_html, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Tab 5: Reports (CSV downloads)
    # -----------------------------------------------------------------------
    with tab_reports:
        st.subheader("📥 Downloadable Reports")
        st.caption("Export contributor statistics, weekly trends, inactivity alerts, or AI summaries as CSV files.")

        r1, r2, r3, r4 = st.columns(4)

        # A. Contributor statistics CSV
        summary_export = build_summary_df(summary, df)
        r1.download_button(
            label="📄 Contributor Stats (CSV)",
            data=df_to_csv(summary_export),
            file_name=f"fairshare_{repo_id.replace('/', '_')}_contributors.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # B. Weekly activity CSV
        r2.download_button(
            label="📅 Weekly Activity (CSV)",
            data=df_to_csv(weekly),
            file_name=f"fairshare_{repo_id.replace('/', '_')}_weekly.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # C. Inactive members CSV
        r3.download_button(
            label="🚨 Inactive Members (CSV)",
            data=inactive_df.to_csv(index=False).encode("utf-8"),
            file_name=f"fairshare_{repo_id.replace('/', '_')}_inactive.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # D. AI summaries CSV (only successfully generated ones)
        ai_rows = []
        for person, (text, _ts, is_error) in st.session_state.ai_summaries.items():
            if not is_error:
                ai_rows.append({"contributor": person, "ai_summary": text})
        if ai_rows:
            ai_df = pd.DataFrame(ai_rows)
            r4.download_button(
                label="🤖 AI Summaries (CSV)",
                data=ai_df.to_csv(index=False).encode("utf-8"),
                file_name=f"fairshare_{repo_id.replace('/', '_')}_ai_summaries.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            r4.info("Generate AI summaries first.")

        # Combined report
        with st.expander("📦 Combined Report (all data)"):
            st.caption("Download a single CSV containing all report sections.")
            combined_parts = []
            combined_parts.append("=== CONTRIBUTOR STATISTICS ===")
            combined_parts.append(summary_export.to_csv(index=True))
            combined_parts.append("\n=== WEEKLY ACTIVITY ===")
            combined_parts.append(weekly.to_csv(index=True))
            combined_parts.append("\n=== INACTIVE MEMBERS ===")
            combined_parts.append(inactive_df.to_csv(index=False))
            if ai_rows:
                combined_parts.append("\n=== AI SUMMARIES ===")
                combined_parts.append(pd.DataFrame(ai_rows).to_csv(index=False))
            combined_csv = "\n".join(combined_parts).encode("utf-8")
            st.download_button(
                label="⬇️ Download Combined Report (CSV)",
                data=combined_csv,
                file_name=f"fairshare_{repo_id.replace('/', '_')}_combined.csv",
                mime="text/csv",
                use_container_width=True,
            )

elif not st.session_state.analysis_done:
    st.markdown(
        f"""
        <div class="fs-hook">
            <div class="fs-hook-text">
                <div class="fs-hook-title">Who really did the work?</div>
                <p class="fs-hook-sub">
                    See every commit, pull request, issue, and review comment in one fair view —
                    without guesswork or noisy merge commits.
                </p>
            </div>
            {_build_contrib_grid_html()}
        </div>
        """,
        unsafe_allow_html=True,
    )

    preview_members = [
        ("Aarav", "AA", 52),
        ("Meera", "ME", 27),
        ("Kabir", "KA", 14),
        ("Zoya", "ZO", 7),
    ]
    preview_rows_html = []
    for idx, (name, initials, pct) in enumerate(preview_members):
        preview_rows_html.append(
            f'<div class="fs-leader-card">'
            f'<div class="fs-rank">#{idx + 1}</div>'
            f'<div class="fs-avatar"><span>{initials}</span></div>'
            f'<div class="fs-leader-body">'
            f'<div class="fs-leader-head">'
            f'<span class="fs-leader-name">{name}</span>'
            f'<span class="fs-leader-count">{pct} activities ({pct}%)</span>'
            f'</div>'
            f'<div class="fs-progress-track">'
            f'<div class="fs-progress-fill" style="width: {pct}%;"></div>'
            f'</div></div></div>'
        )

    st.markdown(
        f"""
        <div class="fs-preview-card">
            <div class="fs-preview-head">
                <span class="fs-preview-title">👀 See it in action</span>
                <span class="fs-sample-pill">Sample data</span>
            </div>
            {"".join(preview_rows_html)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚀 Try the sample team", type="primary", use_container_width=True):
        sample_df = apply_identity_map(_build_sample_df(), st.session_state.identity_map)
        sample_df["date"] = pd.to_datetime(sample_df["date"], utc=True)
        st.session_state.df = sample_df
        st.session_state.repo = "sample/team-project"
        st.session_state.analysis_done = True
        st.session_state.is_sample = True
        st.session_state.ai_summaries = {}
        st.session_state.last_refresh = time.time()
        st.rerun()

    st.caption("Or fill the URL box with an example repository:")
    ex1, ex2, ex3 = st.columns(3)
    ex1.button(
        "pallets/click",
        on_click=_set_example_url,
        args=("pallets/click",),
        use_container_width=True,
    )
    ex2.button(
        "octocat/Hello-World",
        on_click=_set_example_url,
        args=("octocat/Hello-World",),
        use_container_width=True,
    )
    ex3.button(
        "Shafiya05/fairshare",
        on_click=_set_example_url,
        args=("Shafiya05/fairshare",),
        use_container_width=True,
    )

    st.markdown(
        '<p class="fs-home-note">Counts commits, pull requests, issues and comments. '
        'Ignores bots and merge commits. Raw numbers beside every chart.</p>',
        unsafe_allow_html=True,
    )
