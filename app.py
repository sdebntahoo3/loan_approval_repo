import gradio as gr
import pandas as pd
import pickle

with open("loan_approval_XGBoost_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_loan(education, self_employed,
                 income_annum, loan_amount, loan_term, cibil_score,
                 residential_assets_value, commercial_assets_value,
                 luxury_assets_value, bank_asset_value):

    input_df = pd.DataFrame([{
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }])

    pred = int(model.predict(input_df)[0]) 

    if pred == 1:
        return "Approved"
    else:
        return "Rejected"

inputs = [
    gr.Dropdown([" Select", " Graduate", " Not Graduate"], label="Education"),   # space থাকলে dataset অনুযায়ী
    gr.Dropdown([" Select"," Yes", " No"], label="Self Employed"),
    gr.Number(label="Income (Annual)", value=5000000),
    gr.Number(label="Loan Amount", value=15000000),
    gr.Number(label="Loan Term", value=12),
    gr.Number(label="CIBIL Score", value=750),
    gr.Number(label="Residential Assets Value", value=2000000),
    gr.Number(label="Commercial Assets Value", value=5000000),
    gr.Number(label="Luxury Assets Value", value=5000000),
    gr.Number(label="Bank Asset Value", value=2000000),
]

app = gr.Interface(
    fn=predict_loan,
    inputs=inputs,
    outputs="text",
    title="Loan Approval Predictor"
)

app.launch(share=True)