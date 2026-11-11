import pandas as pd
from churn.data import validate
from churn.features import prepare_frame, prepare_scoring_frame
from churn.model import build_pipeline, candidates, choose_threshold

def sample():
    rows=[]
    for i in range(16):
        rows.append({"customerID":f"C{i}","gender":"Male" if i%2 else "Female","SeniorCitizen":i%2,"Partner":"Yes","Dependents":"No","tenure":i+1,"PhoneService":"Yes","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"Yes" if i%3 else "No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month" if i%2 else "Two year","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":30+i,"TotalCharges":str((30+i)*(i+1)),"Churn":"Yes" if i%3==0 else "No"})
    return pd.DataFrame(rows)

def test_features_and_training():
    df=sample(); validate(df); X,y=prepare_frame(df)
    assert {"AvgMonthlySpend","IsNewCustomer","ServiceCount"}.issubset(X.columns)
    pipe=build_pipeline(X,candidates()["logistic_regression"]); pipe.fit(X,y)
    assert pipe.predict_proba(X).shape==(len(X),2)

def test_scoring_and_threshold():
    df=sample(); X=prepare_scoring_frame(df.drop(columns="Churn"))
    assert "customerID" not in X and 0 < choose_threshold([0,0,1,1],[.1,.4,.6,.9]) < 1

