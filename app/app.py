from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("../models/loan_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# Encoding Maps
gender_map = {
    "Male": 1,
    "Female": 0
}

married_map = {
    "Yes": 1,
    "No": 0
}

education_map = {
    "Graduate": 0,
    "Not Graduate": 1
}

self_employed_map = {
    "Yes": 1,
    "No": 0
}

property_area_map = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        gender = gender_map[request.form["Gender"]]
        married = married_map[request.form["Married"]]
        dependents = int(request.form["Dependents"])
        education = education_map[request.form["Education"]]
        self_employed = self_employed_map[request.form["Self_Employed"]]
        applicant_income = float(request.form["ApplicantIncome"])
        coapplicant_income = float(request.form["CoapplicantIncome"])
        loan_amount = float(request.form["LoanAmount"])
        loan_term = float(request.form["Loan_Amount_Term"])
        credit_history = int(request.form["Credit_History"])
        property_area = property_area_map[request.form["Property_Area"]]

        features = np.array([[
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history,
            property_area
        ]])

        features = scaler.transform(features)

        prediction = model.predict(features)[0]

        probability = model.predict_proba(features)[0][1] * 100

        if prediction == 1:
            result = "✅ Loan Approved"
        else:
            result = "❌ Loan Rejected"

        return render_template(
            "index.html",
            prediction=result,
            probability=round(probability, 2)
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction=f"Error : {e}"
        )


if __name__ == "__main__":
    app.run(debug=True)