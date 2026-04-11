This project is a Retail Anomaly Detection Web Application developed using Python, Flask, Pandas, NumPy, HTML, and CSS. The main purpose of this system is to identify unusual or suspicious records in retail transaction datasets. Users can upload a CSV file through a simple web interface, and the application automatically analyzes the data to detect anomalies using the Z-score statistical method.

The system first reads the uploaded dataset and identifies all numeric columns. It then selects the most suitable column based on data variation and calculates the mean and standard deviation. Using these values, it computes the Z-score for each record. Transactions with values significantly higher or lower than normal are marked as anomalies.

After processing, the application displays useful results such as the uploaded file name, selected column, total number of transactions, and number of anomalies detected. Users can also download a CSV report containing all detected anomalies for further analysis.

This project is useful for retail businesses to monitor unusual sales patterns, identify errors, detect fraud, and improve decision-making through data analysis. It also demonstrates practical implementation of data science concepts in a real-time web application.
