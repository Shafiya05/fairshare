"""
app.py – FairShare: Team Contribution Intelligence
Streamlit dashboard for GitHub repository activity analysis.
"""

import io
import json
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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
st.title("📊 FairShare")
st.caption("Team contribution intelligence powered by real GitHub activity.")

# ---------------------------------------------------------------------------
# Repository input row
# ---------------------------------------------------------------------------
col_url, col_btn, col_clear = st.columns([5, 1.5, 1], vertical_alignment="bottom")
url = col_url.text_input(
    "🔗 GitHub Repository URL",
    value="https://github.com/Shafiya05/fairshare",
    placeholder="https://github.com/owner/repo",
)
analyse_clicked = col_btn.button("🔍 Analyse", type="primary", use_container_width=True)
if col_clear.button("🗑️ Clear", use_container_width=True):
    st.session_state.df = None
    st.session_state.analysis_done = False
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
    st.session_state.last_refresh = time.time()

# ---------------------------------------------------------------------------
# Live-mode: countdown fragment + auto-refresh logic
# ---------------------------------------------------------------------------
if live_mode and st.session_state.analysis_done:
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

    if live_mode and st.session_state.last_refresh > 0:
        last_ts = datetime.datetime.fromtimestamp(
            st.session_state.last_refresh
        ).strftime("%H:%M:%S")
        st.caption(f"🔴 Live Mode active · Last updated at {last_ts}")

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
    m1.metric("📁 Repository", repo_id)
    m2.metric("📊 Total Activities", f"{len(df):,}")
    m3.metric("👥 Contributors", df["person"].nunique())
    m4.metric("📅 Date Range", date_range)
    m5.metric("🏆 Top Contributor", top_person, delta=f"{top_total} activities", delta_color="off")

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
    # Tab 1: Overview (bar chart + raw numbers side-by-side)
    # -----------------------------------------------------------------------
    with tab_overview:
        st.subheader("📊 Contribution Split & Statistics")
        st.caption(
            f"Shows the top {TOP_N} contributors broken down by commits, pull requests, "
            "issues, and comments."
        )
        chart_data = summary.drop(columns="total").reindex(columns=ACTIVITY_TYPES, fill_value=0)
        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            st.bar_chart(chart_data, color=ACTIVITY_COLORS, use_container_width=True)
        with col_table:
            st.dataframe(summary, use_container_width=True)

    # -----------------------------------------------------------------------
    # Tab 2: Timeline (weekly chart + table side-by-side)
    # -----------------------------------------------------------------------
    with tab_timeline:
        st.subheader("📅 Weekly Activity Timeline")
        st.caption(
            "Tracks how commits, pull requests, issues, and comments change week by week."
        )
        col_wchart, col_wtable = st.columns([3, 2])
        with col_wchart:
            st.line_chart(weekly, color=ACTIVITY_COLORS, use_container_width=True)
        with col_wtable:
            st.dataframe(weekly, use_container_width=True)

    # -----------------------------------------------------------------------
    # Tab 3: Members (inactive alerts + identity mapping result)
    # -----------------------------------------------------------------------
    with tab_members:
        st.subheader(f"🚨 Inactive Members ({INACTIVE_DAYS}-day threshold)")
        st.caption(
            f"Highlights top contributors who have no recorded activity in the past {INACTIVE_DAYS} days."
        )

        inactive_only = inactive_df[inactive_df["status"] == "INACTIVE"]
        if inactive_only.empty:
            st.success("✅ Everyone in the top 10 has been active recently.")
        else:
            for _, row in inactive_only.iterrows():
                st.warning(
                    f"⚠️ **{row['contributor']}** – last activity "
                    f"{row['days_since_last_activity']} days ago "
                    f"({row['last_activity_date'].date() if pd.notna(row['last_activity_date']) else 'unknown'})"
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

                # Generate all summaries in parallel before rendering expanders
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
                        label = f"🧑‍💻 {person} — {summary.loc[person, 'total']} total contributions"
                        with st.expander(label):
                            if text.startswith("Error:"):
                                st.warning(text)
                            else:
                                st.write(text)
                                st.caption("_(cached)_")

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
    st.info(
        "👋 Paste a public or accessible GitHub repository URL above and click **Analyse** to inspect contributor activity, timelines, inactivity alerts, and AI summaries."
    )
    with st.expander("ℹ️ About FairShare"):
        st.markdown("""
**FairShare** analyses real GitHub repository activity to give a clear, fair picture of each team member's contributions.

**What you get:**
- 📊 **Overview**: Contributor bar chart and raw counts across commits, PRs, issues, and comments
- 📅 **Timeline**: Weekly activity trends and data table
- 👥 **Members**: 30-day inactivity alerts and username identity mapping
- 🤖 **AI Summaries**: Two-sentence contributor summaries powered by Gemini
- 📥 **Reports**: Individual and combined CSV downloads
        """)
