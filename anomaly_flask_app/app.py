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
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)

        # Numeric columns only
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) == 0:
            return "No numeric columns found"

        column = numeric_cols[0]

        threshold = float(request.form["threshold"])

        # Z-score
        mean = df[column].mean()
        std = df[column].std()

        if std == 0:
            std = 1

        df["z_score"] = (df[column] - mean) / std

        anomalies = df[np.abs(df["z_score"]) > threshold]

        if len(anomalies) == 0:
            anomalies = df.head(20)

        # Save CSV
        anomalies.to_csv("anomalies_report.csv", index=False)

        plt.style.use("dark_background")

        # -------------------
        # Chart 1 Scatter
        # -------------------
        plt.figure(figsize=(7,3))
        plt.scatter(df.index, df[column], s=8, color="deepskyblue")
        plt.scatter(anomalies.index, anomalies[column], s=22, color="red")
        plt.title("Scatter Plot")
        plt.tight_layout()
        plt.savefig("static/chart1.png", dpi=120)
        plt.close()

        # -------------------
        # Chart 2 Line
        # -------------------
        plt.figure(figsize=(7,3))
        plt.plot(df.index, df[column], color="lime", linewidth=1.5)
        plt.scatter(anomalies.index, anomalies[column], s=18, color="red")
        plt.title("Trend Line")
        plt.tight_layout()
        plt.savefig("static/chart2.png", dpi=120)
        plt.close()

        # -------------------
        # Chart 3 Bar
        # -------------------
        top = anomalies.head(10)

        plt.figure(figsize=(7,3))
        plt.bar(top.index.astype(str), top[column], color="orange")
        plt.title("Top Anomalies")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("static/chart3.png", dpi=120)
        plt.close()

        table_data = anomalies.head(20).to_dict(orient="records")
        columns = anomalies.columns.tolist()

        return render_template(
            "index.html",
            show_result=True,
            total=len(df),
            anomalies=len(anomalies),
            avg=round(anomalies["z_score"].abs().mean(), 2),
            column=column,
            table_data=table_data,
            columns=columns
        )

    except Exception as e:
        return str(e)


@app.route("/download")
def download():
    return send_file("anomalies_report.csv", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
