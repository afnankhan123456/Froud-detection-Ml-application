from flask import Flask, render_template_string, request
import pickle
import numpy as np
import pandas as pd
import shap
import time
import os

app = Flask(__name__)

# --- HTML path ---
html_path = os.path.join(os.path.dirname(__file__), "back.html")
with open(html_path, "r", encoding="utf-8") as f:
    HTML_FORM = f.read()

# --- Model load ---
model_path = os.path.join(os.path.dirname(__file__), "fraud_model.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)

# --- SHAP explainer ---
explainer = shap.Explainer(model)

# --- TPS counter ---
request_count = 0
start_time = time.time()

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global request_count, start_time
    result = None
    ratios = {}

    if request.method == 'POST':
        try:
            request_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1:
                tps = request_count / elapsed
                print(f"⚙️ TPS: {tps:.2f}")
                request_count = 0
                start_time = time.time()

            # --- Main fields ---
            data = {k: float(request.form.get(k, 0)) for k in [
                "Transaction_Amount", "IP_Risk_Score", "VPN_Flag",
                "Geo_Match", "Avg_Amt_1h", "Velocity_Score"
            ]}

            # --- Ratio calculation when OK pressed ---
            if "calc" in request.form:
                ratios["Amt_vs_AvgRatio"] = data["Transaction_Amount"] / (data["Avg_Amt_1h"] + 1e-6)
                ratios["Velocity_IP_Interaction"] = data["Velocity_Score"] * data["IP_Risk_Score"]
                ratios["Txn_Foreign_Ratio"] = data["VPN_Flag"] / (data["Geo_Match"] + 1e-6)
                return render_template_string(HTML_FORM, result=None, ratios=ratios)

            # --- Ratios for prediction ---
            data["Amt_vs_AvgRatio"] = float(request.form.get("Amt_vs_AvgRatio", 0))
            data["Velocity_IP_Interaction"] = float(request.form.get("Velocity_IP_Interaction", 0))
            data["Txn_Foreign_Ratio"] = float(request.form.get("Txn_Foreign_Ratio", 0))

            x_sample = pd.DataFrame([data])
            feature_names = list(data.keys())

            # --- Prediction ---
            prediction = model.predict(x_sample[feature_names])[0]

            shap_values = explainer(x_sample[feature_names])
            reason_codes = pd.DataFrame()
            reason_codes["Risk_Score"] = np.abs(shap_values.values).mean(axis=1)

            top_features = np.argsort(-np.abs(shap_values.values), axis=1)[:, :3]
            top_reasons = [
                ", ".join(x_sample.columns[top_features[i]].tolist())
                for i in range(len(x_sample))
            ]
            reason_codes["Top_Reason_Codes"] = top_reasons

            conditions = [
                reason_codes["Risk_Score"] < 0.05,
                (reason_codes["Risk_Score"] >= 0.05) & (reason_codes["Risk_Score"] < 0.10),
                reason_codes["Risk_Score"] >= 0.10
            ]
            choices = ["Low", "Medium", "High"]
            reason_codes["Risk_Level"] = np.select(conditions, choices, default="Low")

            if prediction == 1:
                result = (
                    f"⚠️ <b>Fraud Detected!</b>"
                    f"<br><b>Risk Level:</b> {reason_codes['Risk_Level'][0]}"
                    f"<br><b>Top Factors:</b> {reason_codes['Top_Reason_Codes'][0]}"
                    f"<br><b>Risk Score:</b> {reason_codes['Risk_Score'][0]:.3f}"
                )
            else:
                result = (
                    f"✅ <b>Transaction Safe.</b>"
                    f"<br><b>Risk Level:</b> {reason_codes['Risk_Level'][0]}"
                    f"<br><b>Top Factors:</b> {reason_codes['Top_Reason_Codes'][0]}"
                    f"<br><b>Risk Score:</b> {reason_codes['Risk_Score'][0]:.3f}"
                )

        except Exception as e:
            result = f"❌ Error: {str(e)}"

    return render_template_string(HTML_FORM, result=result, ratios=ratios)


# --- Run the app on Render ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
