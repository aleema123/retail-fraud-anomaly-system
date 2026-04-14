

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

# ------------------------------------
# Load Pickle Model
# ------------------------------------
MODEL_PATH = "models.pk1"

model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except:
        model = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)

        # ------------------------------------
        # Numeric Columns
        # ------------------------------------
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) == 0:
            return "No numeric columns found"

        # ------------------------------------
        # Ignore ID + Amount
        # ------------------------------------
        ignore_words = ["id", "transaction_id", "amount"]

        filtered_cols = []

        for col in numeric_cols:
            if not any(word in col.lower() for word in ignore_words):
                filtered_cols.append(col)

        if len(filtered_cols) == 0:
            filtered_cols = numeric_cols

        # ------------------------------------
        # Detect Sales Column First
        # ------------------------------------
        preferred_names = [
            "sales",
            "sale",
            "total_sales",
            "revenue"
        ]

        column = None

        for col in filtered_cols:
            if col.lower() in preferred_names:
                column = col
                break

        # If sales not found choose highest std column
        if column is None:
            best_std = 0
            for col in filtered_cols:
                if df[col].std() > best_std:
                    best_std = df[col].std()
                    column = col

        # ------------------------------------
        # Threshold
        # ------------------------------------
        threshold = request.form.get("threshold")

        if threshold == "" or threshold is None:
            threshold = 2.0
        else:
            threshold = float(threshold)

        # ------------------------------------
        # Z-score
        # ------------------------------------
        mean = df[column].mean()
        std = df[column].std()

        if std == 0:
            std = 1

        df["z_score"] = (df[column] - mean) / std

        anomalies = df[np.abs(df["z_score"]) > threshold]

        if len(anomalies) == 0:
            anomalies = df.reindex(
                df["z_score"].abs().sort_values(ascending=False).index
            ).head(20)

        total_z = round(df["z_score"].abs().sum(), 2)
        anomaly_z = round(anomalies["z_score"].abs().sum(), 2)

        # ------------------------------------
        # Save CSV
        # ------------------------------------
        anomalies.to_csv("anomalies_report.csv", index=False)

        # ------------------------------------
        # Plot 1 Scatter Plot
        # ------------------------------------
        plt.figure(figsize=(10,4))
        plt.scatter(df.index, df[column], s=8, label="Sales")
        plt.scatter(anomalies.index, anomalies[column], color="red", s=25, label="Anomalies")
        plt.title("Scatter Plot")
        plt.xlabel("Row Index")
        plt.ylabel(column)
        plt.legend()
        plt.tight_layout()
        plt.savefig("static/chart1.png")
        plt.close()

        # ------------------------------------
        # Plot 2 Line Plot
        # ------------------------------------
        plt.figure(figsize=(10,4))
        plt.plot(df.index, df[column], linewidth=1.5, label="Sales Trend")
        plt.scatter(anomalies.index, anomalies[column], color="red", s=25)
        plt.title("Line Plot")
        plt.xlabel("Row Index")
        plt.ylabel(column)
        plt.legend()
        plt.tight_layout()
        plt.savefig("static/chart2.png")
        plt.close()

        # ------------------------------------
        # Plot 3 Bar Plot
        # ------------------------------------
        top_anom = anomalies.head(15)

        plt.figure(figsize=(10,4))
        plt.bar(top_anom.index.astype(str), top_anom[column])
        plt.title("Top Anomalies Bar Plot")
        plt.xlabel("Rows")
        plt.ylabel(column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("static/chart3.png")
        plt.close()

        return render_template(
            "index.html",
            show_result=True,
            filename=file.filename,
            column=column,
            threshold=threshold,
            total=len(df),
            anomalies=len(anomalies),
            total_z=total_z,
            anomaly_z=anomaly_z
        )

    except Exception as e:
        return str(e)


@app.route("/download")
def download():
    return send_file("anomalies_report.csv", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
