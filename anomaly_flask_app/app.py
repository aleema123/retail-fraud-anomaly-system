from flask import Flask, render_template, request, send_file, redirect, url_for
import pandas as pd
import numpy as np
import os

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    IsolationForest = None

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", show_result=False)


# ---------------- UPLOAD ----------------
@app.route("/upload", methods=["POST"])
def upload_file():

    file = request.files.get("file")
    threshold = float(request.form.get("threshold", 0))

    if not file or file.filename == "":
        return render_template("index.html", error="Please select a file to upload.", show_result=False)

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    try:
        if IsolationForest is None:
            return render_template("index.html", error="Server error: scikit-learn is not installed.", show_result=False)

        if plt is None:
            return render_template("index.html", error="Server error: matplotlib is not installed.", show_result=False)

        # Load file
        if file.filename.endswith(".csv"):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        # Select numeric columns
        numeric_cols = [
            col for col in df.select_dtypes(include=np.number).columns
            if col.lower() not in ["phone", "zipcode"]
        ]

        if len(numeric_cols) == 0:
            return render_template("index.html", error="No useful numeric columns found in the uploaded file.", show_result=False)

        # Select best column
        column = numeric_cols[0]

        df[column] = pd.to_numeric(df[column], errors='coerce')
        df = df.dropna(subset=[column])

        # Compute z-score for the selected column
        mean = df[column].mean()
        std = df[column].std()
        df["z_score"] = (df[column] - mean) / std

        # ---------------- ML MODEL ----------------
        model = IsolationForest(contamination=0.05, random_state=42)
        df["anomaly"] = model.fit_predict(df[[column]])

        anomalies = df[df["anomaly"] == -1]

        # fallback if empty
        if len(anomalies) == 0:
            anomalies = df[np.abs(df["z_score"]) > threshold]

        z_score_sum = df["z_score"].abs().sum()
        anomaly_z_score_sum = anomalies["z_score"].abs().sum() if len(anomalies) > 0 else 0.0

        # Save report
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], "anomalies_report.csv")
        anomalies.to_csv(output_path, index=False)

        # ---------------- PLOT ----------------
        os.makedirs("static", exist_ok=True)

        plot_filename1 = "plot_scatter.png"
        plot_path1 = os.path.join("static", plot_filename1)
        plt.figure(figsize=(10, 4))
        plt.scatter(df.index, df[column], s=10, alpha=0.5, label="Data")
        plt.scatter(anomalies.index, anomalies[column], s=10, color="orange", label="Anomalies")
        plt.title("Value plot with anomalies")
        plt.xlabel("Row index")
        plt.ylabel(column)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path1)
        plt.close()

        plot_filename2 = "plot_hist.png"
        plot_path2 = os.path.join("static", plot_filename2)
        plt.figure(figsize=(10, 4))
        plt.hist(df[column].dropna(), bins=40, alpha=0.5, label="All values")
        if len(anomalies) > 0:
            plt.hist(anomalies[column].dropna(), bins=40, alpha=0.7, label="Anomalies")
        plt.title("Value distribution")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path2)
        plt.close()

        return render_template(
            "index.html",
            total=len(df),
            anomalies=len(anomalies),
            column=column,
            filename=file.filename,
            show_result=True,
            plot_url1=url_for('static', filename=plot_filename1),
            plot_url2=url_for('static', filename=plot_filename2),
            z_score_sum=round(z_score_sum, 2),
            anomaly_z_score_sum=round(anomaly_z_score_sum, 2),
            threshold=threshold
        )

    except Exception as e:
        return render_template("index.html", error=f"Error: {str(e)}", show_result=False)


# ---------------- DOWNLOAD ----------------
@app.route("/download")
def download_file():
    return send_file(os.path.join(app.config["UPLOAD_FOLDER"], "anomalies_report.csv"), as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
    