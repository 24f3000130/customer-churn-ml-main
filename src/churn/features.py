from __future__ import annotations
import pandas as pd

TARGET = "Churn"
DROP_COLUMNS = ["customerID"]

def prepare_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    work = df.copy()
    work["TotalCharges"] = pd.to_numeric(work["TotalCharges"], errors="coerce")
    work["AvgMonthlySpend"] = work["TotalCharges"] / work["tenure"].replace(0, 1)
    work["IsNewCustomer"] = (work["tenure"] <= 6).astype(int)
    work["ServiceCount"] = sum(
        (work[c] == "Yes").astype(int)
        for c in ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        if c in work
    )
    y = work.pop(TARGET).map({"Yes": 1, "No": 0}).fillna(df[TARGET]).astype(int)
    return work.drop(columns=DROP_COLUMNS, errors="ignore"), y

def prepare_scoring_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["TotalCharges"] = pd.to_numeric(work["TotalCharges"], errors="coerce")
    work["AvgMonthlySpend"] = work["TotalCharges"] / work["tenure"].replace(0, 1)
    work["IsNewCustomer"] = (work["tenure"] <= 6).astype(int)
    work["ServiceCount"] = sum(
        (work[c] == "Yes").astype(int)
        for c in ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        if c in work
    )
    return work.drop(columns=[TARGET, *DROP_COLUMNS], errors="ignore")

