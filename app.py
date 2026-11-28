"""
app.py – FairShare: Team Contribution Intelligence
Streamlit dashboard for GitHub repository activity analysis.
"""

import io
import json
import time
import datetime

import pandas as pd
import streamlit as st

from github_data import collect
from summarizer import summarize, resolve_model

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FairShare – Team Contribution Intelligence",
    page_icon="📊",
    layout="wide",
)

INACTIVE_DAYS = 30
REFRESH_INTERVAL = 300          # seconds between live-mode data refreshes
SUMMARY_TTL = 3600              # seconds to cache AI summaries
TOP_N = 10                      # top-N contributors for statistics
TOP_SUMMARY = 5                 # contributors to summarise with AI


# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------
def _get_secret(key: str, default: str = "") -> str:
    """Read from st.secrets first, fall back to st.session_state override."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


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
        "ai_summaries": {},         # {person: (summary_text, timestamp)}
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
# GitHub data loading (cached 5 min)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=REFRESH_INTERVAL, show_spinner=False)
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
def _summary_is_fresh(person: str) -> bool:
    cache = st.session_state.ai_summaries
    if person not in cache:
        return False
    _, ts = cache[person]
    return (time.time() - ts) < SUMMARY_TTL


def _get_cached_summary(person: str) -> str:
    text, _ = st.session_state.ai_summaries[person]
    return text


def _store_summary(person: str, text: str):
    st.session_state.ai_summaries[person] = (text, time.time())


# ---------------------------------------------------------------------------
# CSV / Excel export helpers
# ---------------------------------------------------------------------------
def df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


def build_summary_df(top_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Add original usernames column to contributor summary."""
    id_map = st.session_state.identity_map
    rows = []
    for person in top_df.index:
        # find original logins
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
    days_ago = (now - last).dt.days
    inactive_rows = []
    for person in summary.index:
        d = int(days_ago.get(person, 999))
        status = "INACTIVE" if d > INACTIVE_DAYS else "Active"
        inactive_rows.append({
            "contributor": person,
            "last_activity_date": last.get(person, pd.NaT),
            "days_since_last_activity": d,
            "status": status,
        })
    return pd.DataFrame(inactive_rows)


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.header("⚙️ Settings")

    # Live mode
    live_mode = st.checkbox(
        "🔴 Live Mode (auto-refresh every 5 min)",
        value=st.session_state.live_mode,
    )
    st.session_state.live_mode = live_mode
    if live_mode:
        elapsed = time.time() - st.session_state.last_refresh
        remaining = max(0, REFRESH_INTERVAL - elapsed)
        st.info(f"Next refresh in {int(remaining)}s")

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
                # Invalidate cached data so stats recalculate
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
st.title("📊 FairShare – Team Contribution Intelligence")
st.caption("Analyse real GitHub repository activity to understand team contributions.")

# ---------------------------------------------------------------------------
# Repository input
# ---------------------------------------------------------------------------
url = st.text_input(
    "🔗 Paste a GitHub repository URL",
    value="https://github.com/Shafiya05/fairshare",
    placeholder="https://github.com/owner/repo",
)

col_btn, col_clear = st.columns([2, 1])
analyse_clicked = col_btn.button("🔍 Analyse Repository", type="primary", use_container_width=True)
if col_clear.button("🗑️ Clear", use_container_width=True):
    st.session_state.df = None
    st.session_state.analysis_done = False
    st.session_state.ai_summaries = {}
    st.rerun()

# ---------------------------------------------------------------------------
# Live-mode auto-refresh trigger
# ---------------------------------------------------------------------------
if live_mode and st.session_state.analysis_done:
    elapsed = time.time() - st.session_state.last_refresh
    if elapsed >= REFRESH_INTERVAL:
        st.cache_data.clear()
        st.session_state.df = None
        st.session_state.last_refresh = time.time()
        st.rerun()
    else:
        # Schedule a rerun after remaining time using st.fragment / query_params trick
        # We use a countdown display and rely on the user's browser to trigger rerun
        # via the meta-refresh equivalent: st.empty + time-based logic.
        # Streamlit 1.67 supports st.rerun(); we schedule via a short sleep in a
        # non-blocking fragment-like approach using session time tracking.
        pass

# ---------------------------------------------------------------------------
# Load / trigger analysis
# ---------------------------------------------------------------------------
token = _get_secret("GITHUB_TOKEN")
gemini_key = _get_secret("GEMINI_API_KEY")
gemini_model_cfg = _get_secret("GEMINI_MODEL", "")

if analyse_clicked:
    try:
        repo_id = parse_repo(url)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    with st.spinner(f"Fetching data for **{repo_id}** …"):
        try:
            raw_df = load_data(repo_id, token)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error fetching data: {e}")
            st.stop()

    if raw_df is None or raw_df.empty:
        st.warning("No activity data found for this repository.")
        st.stop()

    # Apply identity mapping
    df = apply_identity_map(raw_df, st.session_state.identity_map)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    st.session_state.df = df
    st.session_state.repo = repo_id
    st.session_state.analysis_done = True
    st.session_state.last_refresh = time.time()

# ---------------------------------------------------------------------------
# Render dashboard if data is available
# ---------------------------------------------------------------------------
if st.session_state.analysis_done and st.session_state.df is not None:
    df = st.session_state.df
    repo_id = st.session_state.repo

    # --- Repository overview ---
    st.divider()
    st.subheader(f"📁 Repository: `{repo_id}`")
    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.metric("Total Activities", len(df))
    ov2.metric("Contributors", df["person"].nunique())
    ov3.metric("Activity Types", df["type"].nunique())
    if not df.empty:
        date_range = f"{df['date'].min().date()} → {df['date'].max().date()}"
        ov4.metric("Date Range", date_range)

    # --- Contributor statistics ---
    st.divider()
    st.subheader("👥 Contributor Statistics")

    summary = pd.crosstab(df["person"], df["type"])
    summary["total"] = summary.sum(axis=1)
    summary = summary.sort_values("total", ascending=False).head(TOP_N)

    st.dataframe(summary, use_container_width=True)

    # --- Contribution split ---
    st.subheader("📊 Contribution Split")
    chart_data = summary.drop(columns="total")
    st.bar_chart(chart_data, use_container_width=True)

    # --- Weekly activity timeline ---
    st.divider()
    st.subheader("📅 Activity Timeline (per week)")
    weekly = (
        df.set_index("date")
        .groupby([pd.Grouper(freq="W"), "type"])
        .size()
        .unstack(fill_value=0)
    )
    st.line_chart(weekly, use_container_width=True)
    with st.expander("📋 Weekly activity table"):
        st.dataframe(weekly, use_container_width=True)

    # --- Inactive member alerts ---
    st.divider()
    st.subheader(f"🚨 Inactive Members (no activity in {INACTIVE_DAYS} days)")
    now_utc = pd.Timestamp.now(tz="UTC")
    last_activity = df.groupby("person")["date"].max()
    inactive_df = build_inactive_df(df, summary)

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

    # --- AI Contributor Summaries ---
    st.divider()
    st.subheader("🤖 AI Contributor Summaries")

    if not gemini_key:
        st.info(
            "ℹ️ Gemini API key not configured. "
            "Add `GEMINI_API_KEY` to `.streamlit/secrets.toml` to enable AI summaries."
        )
    else:
        # Resolve model once per session
        if st.session_state.ai_model is None:
            with st.spinner("Verifying Gemini model availability…"):
                resolved = resolve_model(gemini_key, gemini_model_cfg or None)
            st.session_state.ai_model = resolved

        ai_model = st.session_state.ai_model

        if not ai_model:
            st.error("No suitable Gemini model found for your API key.")
        else:
            st.caption(f"Model: `{ai_model}` · Summaries cached for 1 hour")

            top5 = summary.head(TOP_SUMMARY)

            for person in top5.index:
                person_counts = {
                    k: int(v)
                    for k, v in summary.loc[person].to_dict().items()
                    if k != "total"
                }

                # Collect recent meaningful titles
                person_df = df[df["person"] == person].head(40)
                titles = (
                    person_df["title"]
                    .dropna()
                    .loc[person_df["title"].str.strip() != ""]
                    .tolist()[:15]
                ) if "title" in person_df.columns else []

                with st.expander(f"🧑‍💻 {person} — {summary.loc[person, 'total']} total contributions"):
                    if _summary_is_fresh(person):
                        cached_text = _get_cached_summary(person)
                        st.write(cached_text)
                        st.caption("_(cached)_")
                    else:
                        with st.spinner(f"Generating summary for {person}…"):
                            result = summarize(
                                person=person,
                                counts=person_counts,
                                titles=titles,
                                api_key=gemini_key,
                                model=ai_model,
                            )
                        if result.startswith("Error:"):
                            st.warning(result)
                        else:
                            _store_summary(person, result)
                            st.write(result)

    # --- Downloadable Reports ---
    st.divider()
    st.subheader("📥 Downloadable Reports")

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
    for person, (text, _ts) in st.session_state.ai_summaries.items():
        if not text.startswith("Error:"):
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

    # Combined report (all sheets as CSV)
    with st.expander("📦 Combined Report (all data)"):
        st.caption("Download a combined CSV containing all report data.")
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

    # --- Live mode status ---
    if live_mode:
        elapsed = time.time() - st.session_state.last_refresh
        remaining = max(0, REFRESH_INTERVAL - elapsed)
        st.sidebar.progress(
            min(1.0, elapsed / REFRESH_INTERVAL),
            text=f"Refresh in {int(remaining)}s",
        )
        # Use st.rerun() with a short sleep for live refresh signalling
        # This avoids blocking – Streamlit re-runs the entire script on rerun()
        time.sleep(1)
        st.rerun()

elif not st.session_state.analysis_done:
    st.info("👆 Enter a GitHub repository URL above and click **Analyse Repository** to begin.")
    with st.expander("ℹ️ About FairShare"):
        st.markdown("""
**FairShare – Team Contribution Intelligence** analyses real GitHub repository activity 
to give a fair picture of each team member's contributions.

**Features:**
- 📊 Contributor statistics (commits, PRs, issues, comments)  
- 📅 Weekly activity timeline  
- 🚨 Inactive member alerts (30-day threshold)  
- 🤖 AI-generated contributor summaries (requires Gemini API key)  
- 👤 Duplicate username identity mapping  
- 🔴 Live auto-refresh every 5 minutes  
- 📥 Downloadable CSV reports  

**Setup:** Add your credentials to `.streamlit/secrets.toml`:
```toml
GITHUB_TOKEN = "ghp_..."
GEMINI_API_KEY = "AIza..."
GEMINI_MODEL = "gemini-2.0-flash"
```
        """)
