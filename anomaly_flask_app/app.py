# app.py

from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- Load Model ----------------
MODEL_PATH = "model.pkl"

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)


# ---------------- Home ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- Upload ----------------
@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) == 0:
            return "No numeric columns found"

        column = numeric_cols[0]
        threshold = float(request.form["threshold"])

        # ---------------- Prediction ----------------
        if model is not None:
            preds = model.predict(df[[column]])
            df["prediction"] = preds
            anomalies = df[df["prediction"] == -1]

        else:
            mean = df[column].mean()
            std = df[column].std()

            if std == 0:
                std = 1

            df["z_score"] = (df[column] - mean) / std
            anomalies = df[np.abs(df["z_score"]) > threshold]

        if len(anomalies) == 0:
            anomalies = df.head(10)

        anomalies.to_csv("anomalies_report.csv", index=False)

        plt.style.use("dark_background")

        # ---------------- Chart 1 Line Graph ----------------
        plt.figure(figsize=(7,3))
        plt.plot(df.index, df[column], linewidth=2, color="cyan")
        plt.scatter(anomalies.index, anomalies[column], color="red", s=25)
        plt.title("Line Graph")
        plt.tight_layout()
        plt.savefig("static/chart1.png", dpi=120)
        plt.close()

        # ---------------- Chart 2 Trend Line ----------------
        x = np.arange(len(df))
        y = df[column].values

        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)

        plt.figure(figsize=(7,3))
        plt.plot(x, y, color="deepskyblue", alpha=0.4)
        plt.plot(x, p(x), color="yellow", linewidth=3)
        plt.title("Trend Line")
        plt.tight_layout()
        plt.savefig("static/chart2.png", dpi=120)
        plt.close()

        # ---------------- Chart 3 Histogram ----------------
        plt.figure(figsize=(7,3))
        plt.hist(df[column], bins=20, color="orange", edgecolor="white")
        plt.axvline(df[column].mean(), color="cyan", linestyle="--", linewidth=2)
        plt.title("Histogram")
        plt.tight_layout()
        plt.savefig("static/chart3.png", dpi=120)
        plt.close()

        table_data = anomalies.head(20).to_dict(orient="records")
        columns = anomalies.columns.tolist()

        avg_score = 0
        if "z_score" in anomalies.columns:
            avg_score = round(anomalies["z_score"].abs().mean(), 2)

        return render_template(
            "index.html",
            show_result=True,
            total=len(df),
            anomalies=len(anomalies),
            avg=avg_score,
            column=column,
            table_data=table_data,
            columns=columns
        )

    except Exception as e:
        return str(e)


# ---------------- Download ----------------
@app.route("/download")
def download():
    return send_file("anomalies_report.csv", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
