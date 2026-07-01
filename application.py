from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

car = pd.read_csv("Cleaned_Car.csv")
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))


@app.route("/")
def index():
    companies = sorted(car["company"].unique())
    car_models = sorted(car["name"].unique())
    years = sorted(car["year"].unique(), reverse=True)
    fuel_types = sorted(car["fuel_type"].unique())

    return render_template(
        "index.html",
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types
    )


@app.route("/predict", methods=["POST"])
def predict():
    company = request.form.get("company")
    car_model = request.form.get("car_models")
    year = int(request.form.get("year"))
    fuel_type = request.form.get("fuel_type")
    kilo_driven = int(request.form.get("kilo_driven"))

    input_data = pd.DataFrame(
        [[car_model, company, year, kilo_driven, fuel_type]],
        columns=["name", "company", "year", "kms_driven", "fuel_type"]
    )

    prediction = model.predict(input_data)

    return str(round(prediction[0], 2))


if __name__ == "__main__":
    app.run(debug=True)