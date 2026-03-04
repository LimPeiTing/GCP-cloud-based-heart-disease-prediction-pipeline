from flask import Flask, request, jsonify
from google.cloud import bigquery
import os

app = Flask(__name__)

# BigQuery client
PROJECT_ID = "cardio-predict-project"
DATASET = "cardio_analytics_dataset"
MODEL = "log_reg_model_small"

client = bigquery.Client(project=PROJECT_ID)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Expecting JSON input with 8 features
        query = f"""
        SELECT predicted_HadHeartAttackInt, predicted_HadHeartAttackInt_probs
        FROM ML.PREDICT(MODEL `{PROJECT_ID}.{DATASET}.{MODEL}`,
          (SELECT
            {', '.join(data.keys())}
          ))
        """

        job = client.query(query)
        results = [dict(row) for row in job.result()]

        return jsonify({"predictions": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
