from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from churn.features import prepare_scoring_frame

st.set_page_config(page_title="Churn Risk Studio", page_icon="📉", layout="wide")
st.title("Customer Churn Risk Studio")
st.caption("Upload customer records to prioritize retention outreach.")
model_path=Path("models/churn_model.joblib")
if not model_path.exists(): st.warning("Train a model first: `python -m churn.train`"); st.stop()
artifact=joblib.load(model_path)
uploaded=st.file_uploader("Telco customer CSV", type="csv")
if uploaded:
    raw=pd.read_csv(uploaded); X=prepare_scoring_frame(raw).reindex(columns=artifact.feature_columns); p=artifact.pipeline.predict_proba(X)[:,1]
    out=raw.copy(); out["churn_probability"]=p; out["risk_band"]=pd.cut(p,[-1,.3,.6,1],labels=["Low","Medium","High"])
    a,b,c=st.columns(3); a.metric("Customers",len(out)); b.metric("High risk",int((p>=.6).sum())); c.metric("Average risk",f"{p.mean():.1%}")
    st.bar_chart(out["risk_band"].value_counts()); st.dataframe(out.sort_values("churn_probability",ascending=False),use_container_width=True)
    st.download_button("Download scored customers",out.to_csv(index=False),"churn_predictions.csv","text/csv")

