# app.py

from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# -----------------------------------------
# Folders
# -----------------------------------------
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# -----------------------------------------
# Home Page
# -----------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------------------
# Upload Dataset
# -----------------------------------------
@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    try:

        # -----------------------------------------
        # Read CSV / Excel
        # -----------------------------------------
        if file.filename.lower().endswith(".csv"):

            try:
                df = pd.read_csv(filepath, encoding="utf-8")

            except:
                try:
                    df = pd.read_csv(filepath, encoding="latin1")

                except:
                    df = pd.read_csv(filepath, encoding="cp1252")

        elif file.filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(filepath)

        else:
            return "Upload CSV or Excel file only"

        # -----------------------------------------
        # Detect Numeric Columns
        # -----------------------------------------
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) == 0:
            return "No numeric columns found"

        column = numeric_cols[0]

        # If Quantity + Price exists
        if "Quantity" in df.columns and "Price" in df.columns:
            df["TotalAmount"] = df["Quantity"] * df["Price"]
            column = "TotalAmount"

        threshold = float(request.form["threshold"])

        # -----------------------------------------
        # Z Score
        # -----------------------------------------
        mean = df[column].mean()
        std = df[column].std()

        if std == 0:
            std = 1

        df["ZScore"] = (df[column] - mean) / std

        df["Is_Anomaly"] = np.where(
            abs(df["ZScore"]) > threshold,
            True,
            False
        )

        anomalies = df[df["Is_Anomaly"] == True]

        if len(anomalies) == 0:
            anomalies = df.head(10)

        anomalies.to_csv("anomalies_report.csv", index=False)

        # -----------------------------------------
        # Graph Style
        # -----------------------------------------
        plt.style.use("dark_background")

        # -----------------------------------------
        # Graph 1 Line Chart
        # -----------------------------------------
        plt.figure(figsize=(8,4))
        plt.plot(df.index, df[column], color="cyan", linewidth=2)

        plt.scatter(
            anomalies.index,
            anomalies[column],
            color="red",
            s=40
        )

        plt.title("Z-Score Anomaly Detection")
        plt.tight_layout()
        plt.savefig(
            "static/chart1.png",
            dpi=150,
            bbox_inches="tight"
        )
        plt.close()

        # -----------------------------------------
        # Graph 2 Histogram
        # -----------------------------------------
        plt.figure(figsize=(8,4))
        plt.hist(
            df[column],
            bins=25,
            color="orange",
            edgecolor="white"
        )

        plt.axvline(
            mean,
            color="cyan",
            linestyle="--",
            linewidth=2
        )

        plt.title("Histogram")
        plt.tight_layout()
        plt.savefig(
            "static/chart2.png",
            dpi=150,
            bbox_inches="tight"
        )
        plt.close()

        # -----------------------------------------
        # Graph 3 Boxplot
        # -----------------------------------------
        plt.figure(figsize=(8,4))
        plt.boxplot(df[column], vert=False)
        plt.title("Boxplot")
        plt.tight_layout()
        plt.savefig(
            "static/chart3.png",
            dpi=150,
            bbox_inches="tight"
        )
        plt.close()

        # -----------------------------------------
        # Table
        # -----------------------------------------
        table_data = anomalies.head(20).to_dict(orient="records")
        columns = anomalies.columns.tolist()

        avg_score = round(
            anomalies["ZScore"].abs().mean(),
            2
        )

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


# -----------------------------------------
# Download Report
# -----------------------------------------
@app.route("/download")
def download():
    return send_file(
        "anomalies_report.csv",
        as_attachment=True
    )


# -----------------------------------------
# Run App
# -----------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
