
from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():

    file = request.files.get("file")

    if not file or file.filename == "":
        return "No file uploaded!"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)

        # Numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) == 0:
            return "No numeric columns found!"

        # Select best column
        best_col = None
        max_std = 0

        for col in numeric_cols:
            std = df[col].std()
            if std > max_std:
                max_std = std
                best_col = col

        column = best_col

        # Clean data
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=[column])

        mean = df[column].mean()
        std = df[column].std()

        if std == 0:
            return "No variation in data!"

        # Default threshold
        threshold = request.form.get("threshold")

        if threshold == "" or threshold is None:
            threshold = 2.0
        else:
            threshold = float(threshold)

        # Z-score
        df["z_score"] = (df[column] - mean) / std

        anomalies = df[np.abs(df["z_score"]) > threshold]

        # Auto fallback
        if len(anomalies) == 0:
            threshold = 1.5
            anomalies = df[np.abs(df["z_score"]) > threshold]

        # Totals
        total_z = round(df["z_score"].abs().sum(), 2)
        anomaly_z = round(anomalies["z_score"].abs().sum(), 2)

        # Save CSV
        anomalies.to_csv("anomalies_report.csv", index=False)

        # -------------------------
        # Plot 1 Scatter Plot
        # -------------------------
        plt.figure(figsize=(10,4))
        plt.scatter(df.index, df[column], s=5, label="Data")
        plt.scatter(anomalies.index, anomalies[column], color="red", s=20, label="Anomalies")
        plt.title("Scatter Plot")
        plt.xlabel("Row Index")
        plt.ylabel(column)
        plt.legend()
        plt.tight_layout()
        plt.savefig("static/chart1.png")
        plt.close()

        # -------------------------
        # Plot 2 Histogram
        # -------------------------
        plt.figure(figsize=(10,4))
        plt.hist(df[column], bins=30, alpha=0.7, label="All Data")
        plt.hist(anomalies[column], bins=20, alpha=0.8, label="Anomalies")
        plt.title("Histogram")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig("static/chart2.png")
        plt.close()

        # -------------------------
        # Plot 3 Box Plot
        # -------------------------
        plt.figure(figsize=(8,4))
        plt.boxplot(df[column], vert=False)
        plt.title("Box Plot")
        plt.xlabel(column)
        plt.tight_layout()
        plt.savefig("static/chart3.png")
        plt.close()

        return render_template(
            "index.html",
            total=len(df),
            anomalies=len(anomalies),
            column=column,
            filename=file.filename,
            total_z=total_z,
            anomaly_z=anomaly_z,
            threshold=threshold,
            show_result=True
        )

    except Exception as e:
        return f"Error: {str(e)}"


@app.route("/download")
def download_file():
    return send_file("anomalies_report.csv", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)

