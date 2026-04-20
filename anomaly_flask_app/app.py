# ===============================================
# app.py
# SMART RETAIL ANOMALY DETECTION - PRO VERSION
# ===============================================

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


# ===============================================
# HOME
# ===============================================
@app.route("/")
def home():
    return render_template("index.html")


# ===============================================
# UPLOAD
# ===============================================
@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:

        # ---------------------------------------
        # READ FILE
        # ---------------------------------------
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
            return "Upload CSV or Excel only"

        # ---------------------------------------
        # NUMERIC COLUMN
        # ---------------------------------------
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) == 0:
            return "No numeric columns found"

        column = numeric_cols[0]

        if "Quantity" in df.columns and "Price" in df.columns:
            df["TotalAmount"] = df["Quantity"] * df["Price"]
            column = "TotalAmount"

        threshold = float(request.form["threshold"])

        # ---------------------------------------
        # Z SCORE
        # ---------------------------------------
        mean = df[column].mean()
        std = df[column].std()

        if std == 0:
            std = 1

        df["ZScore"] = ((df[column] - mean) / std).round(2)

        # ---------------------------------------
        # TRUE / FALSE
        # ---------------------------------------
        df["Is_Anomaly"] = np.where(
            abs(df["ZScore"]) >= threshold,
            "True",
            "False"
        )

        # ---------------------------------------
        # RISK LEVEL
        # ---------------------------------------
        def risk_level(z):
            z = abs(z)

            if z >= 4:
                return "High"
            elif z >= 2:
                return "Medium"
            else:
                return "Low"

        df["Risk"] = df["ZScore"].apply(risk_level)

        anomalies = df[df["Is_Anomaly"] == "True"]

        # ---------------------------------------
        # ACCURACY
        # ---------------------------------------
        total = len(df)
        anomaly_count = len(anomalies)

        normal = total - anomaly_count

        accuracy = round((normal / total) * 100, 2)

        # ---------------------------------------
        # SAVE REPORT
        # ---------------------------------------
        df.to_csv("anomalies_report.csv", index=False)

        # =======================================
        # CHARTS
        # =======================================
        plt.style.use("dark_background")

        # ---------------------------------------
        # Chart 1 Line
        # ---------------------------------------
        plt.figure(figsize=(10,4))

        plt.plot(df.index, df[column], color="cyan")

        plt.scatter(
            anomalies.index,
            anomalies[column],
            color="red",
            s=50
        )

        plt.title("Anomaly Detection")
        plt.tight_layout()
        plt.savefig("static/chart1.png", dpi=150)
        plt.close()

        # ---------------------------------------
        # Chart 2 Histogram
        # ---------------------------------------
        plt.figure(figsize=(10,4))

        plt.hist(df[column], bins=30, color="orange", edgecolor="white")

        plt.axvline(mean, color="cyan", linestyle="--")

        plt.title("Distribution")
        plt.tight_layout()
        plt.savefig("static/chart2.png", dpi=150)
        plt.close()

        # ---------------------------------------
        # Chart 3 Pie
        # ---------------------------------------
        plt.figure(figsize=(7,7))

        plt.pie(
            [normal, anomaly_count],
            labels=["Normal", "Anomaly"],
            autopct="%1.1f%%"
        )

        plt.title("Normal vs Anomaly")
        plt.tight_layout()
        plt.savefig("static/chart3.png", dpi=150)
        plt.close()

        # ---------------------------------------
        # ONLY TRUE ROWS TABLE
        # ---------------------------------------
        table_data = anomalies.head(50).to_dict(orient="records")
        columns = anomalies.columns.tolist()

        avg_score = round(df["ZScore"].abs().mean(), 2)

        return render_template(
            "index.html",
            show_result=True,
            total=total,
            anomalies=anomaly_count,
            avg=avg_score,
            accuracy=accuracy,
            column=column,
            table_data=table_data,
            columns=columns
        )

    except Exception as e:
        return str(e)


# ===============================================
# DOWNLOAD
# ===============================================
@app.route("/download")
def download():
    return send_file(
        "anomalies_report.csv",
        as_attachment=True
    )


# ===============================================
# RUN
# ===============================================
if __name__ == "__main__":
    app.run(debug=True)
    
