from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load trained pipeline model
model = joblib.load("best_rf_model.joblib")

app = Flask(__name__)

# -------------------------
# Home page (UI)
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# -------------------------
# UI prediction
# -------------------------
@app.route("/predict-ui", methods=["POST"])
def predict_ui():
    data = {
        "carat": float(request.form["carat"]),
        "depth": float(request.form["depth"]),
        "table": float(request.form["table"]),
        "x": float(request.form["x"]),
        "y": float(request.form["y"]),
        "z": float(request.form["z"]),
        "cut": request.form["cut"],
        "color": request.form["color"],
        "clarity": request.form["clarity"]
    }

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    return render_template(
        "index.html",
        prediction=round(prediction, 2)
    )

# -------------------------
# REST API prediction
# -------------------------
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    return jsonify({
        "predicted_price": float(prediction)
    })

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
