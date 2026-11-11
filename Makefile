.PHONY: setup data train test dashboard clean
setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
data:
	.venv/bin/python -m churn.data
train:
	.venv/bin/python -m churn.train
test:
	.venv/bin/pytest -q
dashboard:
	.venv/bin/streamlit run app.py
clean:
	rm -rf models/* reports/figures/* data/processed/*

