# app.py

from flask import Flask, render_template, request, send_file
from io import BytesIO
import pandas as pd
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

processed_df = None

# Static folder auto detect
GRAPH_FOLDER = os.path.join(
    app.root_path,
    "static"
)

os.makedirs(
    GRAPH_FOLDER,
    exist_ok=True
)


@app.route("/")
def home():

    return render_template(
        "index.html",
        show_result=False,
        generated_graphs=[],
        cache_id=int(time.time())
    )


@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    global processed_df

    try:

        file = request.files.get("file")

        if not file:
            return "No file uploaded"

        # Read CSV
        df = pd.read_csv(file)

        df.columns = df.columns.str.strip()

        numeric_cols = df.select_dtypes(
            include="number"
        ).columns

        if len(numeric_cols) == 0:
            return "No numeric columns found"

        amount_col = numeric_cols[0]

        threshold = float(
            request.form.get(
                "threshold",
                1.5
            )
        )

        # -------------------------
        # Z Score
        # -------------------------
        mean_val = df[amount_col].mean()
        std_val = df[amount_col].std()

        if std_val == 0:
            std_val = 1

        df["ZScore"] = abs(
            (df[amount_col] - mean_val)
            / std_val
        )

        cutoff = df["ZScore"].quantile(0.95)

        df["Is_Anomaly"] = df[
            "ZScore"
        ].apply(
            lambda x:
            "TRUE"
            if x > threshold
            or x >= cutoff
            else "FALSE"
        )

        def risk(z):

            if z >= cutoff:
                return "High"

            elif z > 1:
                return "Medium"

            return "Low"

        df["Risk"] = df["ZScore"].apply(risk)

        # -------------------------
        # Metrics
        # -------------------------
        total = len(df)

        anomalies = len(
            df[
                df["Is_Anomaly"]
                == "TRUE"
            ]
        )

        avg_z = round(
            df["ZScore"].mean(),
            2
        )

        accuracy = round(
            ((total - anomalies) / total) * 100,
            2
        )

        processed_df = df.copy()

        # -------------------------
        # Delete old png files
        # -------------------------
        for f in os.listdir(GRAPH_FOLDER):

            if f.endswith(".png"):

                try:
                    os.remove(
                        os.path.join(
                            GRAPH_FOLDER,
                            f
                        )
                    )
                except:
                    pass

        generated_graphs = []

        # -------------------------
        # Graph 1 ZScore
        # -------------------------
        plt.figure(figsize=(8,4))
        df["ZScore"].plot(
            color="cyan"
        )
        plt.title("Detected Anomalies")
        plt.grid(True)
        plt.tight_layout()

        name = "chart1.png"

        plt.savefig(
            os.path.join(
                GRAPH_FOLDER,
                name
            ),
            dpi=120
        )
        plt.close()

        generated_graphs.append(name)

        # -------------------------
        # Graph 2 Histogram
        # -------------------------
        plt.figure(figsize=(8,4))
        df[amount_col].hist(
            bins=30,
            color="orange"
        )
        plt.title("Distribution")
        plt.tight_layout()

        name = "chart2.png"

        plt.savefig(
            os.path.join(
                GRAPH_FOLDER,
                name
            ),
            dpi=120
        )
        plt.close()

        generated_graphs.append(name)

        # -------------------------
        # Graph 3 Bar
        # -------------------------
        plt.figure(figsize=(7,4))
        df["Is_Anomaly"]\
            .value_counts()\
            .plot(
                kind="bar",
                color=["green","red"]
            )

        plt.title("Normal vs Anomaly")
        plt.tight_layout()

        name = "chart3.png"

        plt.savefig(
            os.path.join(
                GRAPH_FOLDER,
                name
            ),
            dpi=120
        )
        plt.close()

        generated_graphs.append(name)

        suspicious = df.sort_values(
            by="ZScore",
            ascending=False
        ).head(20)

        return render_template(
            "index.html",

            show_result=True,

            total=total,
            anomalies=anomalies,
            avg=avg_z,
            mean=round(mean_val,2),
            std=round(std_val,2),
            accuracy=accuracy,

            columns=suspicious.columns,

            table_data=suspicious.to_dict(
                orient="records"
            ),

            generated_graphs=generated_graphs,

            cache_id=int(time.time())
        )

    except Exception as e:

        return f"Error: {str(e)}"


@app.route("/download")
def download():

    global processed_df

    if processed_df is None:
        return "No report available"

    output = BytesIO()

    processed_df.to_csv(
        output,
        index=False
    )

    output.seek(0)

    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="anomaly_report.csv"
    )


if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )
