import streamlit as st
import pandas as pd
from github_data import collect

INACTIVE_DAYS = 30

st.title("FairShare – Team Contribution Intelligence")

url = st.text_input("Paste a public GitHub repo link",
                    "https://github.com/pallets/flask")

def parse_repo(link):
    parts = link.strip().rstrip("/").split("github.com/")[-1].split("/")
    return f"{parts[0]}/{parts[1]}"

@st.cache_data(ttl=300)
def load(repo):
    return collect(repo)

if st.button("Analyse"):
    try:
        repo = parse_repo(url)
        df = load(repo)
        df["date"] = pd.to_datetime(df["date"], utc=True)

        summary = pd.crosstab(df["person"], df["type"])
        summary["total"] = summary.sum(axis=1)
        summary = summary.sort_values("total", ascending=False).head(10)

        st.subheader("Raw numbers")
        st.dataframe(summary)

        st.subheader("Contribution split")
        st.bar_chart(summary.drop(columns="total"))

        st.subheader("Activity timeline (per week)")
        weekly = (df.set_index("date")
                    .groupby([pd.Grouper(freq="W"), "type"])
                    .size().unstack(fill_value=0))
        st.line_chart(weekly)
        st.dataframe(weekly)

        st.subheader(f"Inactive members (no activity in {INACTIVE_DAYS} days)")
        last = df.groupby("person")["date"].max()[summary.index]
        now = pd.Timestamp.now(tz="UTC")
        days_ago = (now - last).dt.days
        inactive = days_ago[days_ago > INACTIVE_DAYS]
        if inactive.empty:
            st.success("Everyone in the top 10 has been active recently.")
        else:
            for person, d in inactive.items():
                st.warning(f"{person}: last activity {d} days ago")
    except Exception as e:
        st.error(f"Could not analyse this repo: {e}")