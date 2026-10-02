import os
import numpy as np
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Relative paths that work on any computer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ames_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "ames_preprocessor.pkl")

# Load once at startup
model = joblib.load(MODEL_PATH)
prep = joblib.load(PREPROCESSOR_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    error = None

    if request.method == "POST":
        try:
            # 1. Form data -> single-row DataFrame
            data = request.form.to_dict()
            df = pd.DataFrame([data])

            # 2. Coerce every column to numeric (blank -> NaN)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # 3. Align columns with what the preprocessor expects.
            #    Missing columns become NaN so the pipeline can impute them.
            expected_cols = list(prep.feature_names_in_)
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[expected_cols]

            # 4. Debug prints (remove later if you want)
            print("=== DEBUG ===")
            print("Form data received:", data)
            print("DataFrame values:", df.to_dict(orient="records"))
            print("--------------")

            # 5. Transform and predict ONCE
            X_ready = prep.transform(df)
            raw_pred = model.predict(X_ready)[0]

            # 6. Reverse the log1p transform (matches your training)
            real_price = float(np.expm1(raw_pred))

            print("Raw model output:", raw_pred)
            print("After expm1:", real_price)

            # 7. Format
            prediction_text = f"${real_price:,.0f}"

        except Exception as e:
            error = f"Prediction failed: {e}"
            prediction_text = None

    return render_template("index.html", prediction_text=prediction_text, error=error)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)