import streamlit as st
import pandas as pd
from github_data import collect

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
        summary = pd.crosstab(df["person"], df["type"])
        summary["total"] = summary.sum(axis=1)
        summary = summary.sort_values("total", ascending=False).head(10)

        st.subheader("Raw numbers")
        st.dataframe(summary)

        st.subheader("Contribution split")
        st.bar_chart(summary.drop(columns="total"))
    except Exception as e:
        st.error(f"Could not analyse this repo: {e}")