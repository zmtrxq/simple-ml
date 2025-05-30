# Simple ML - Automated Model Runner & Analyzer

Simple ML is a user-friendly web application designed to help you quickly train, evaluate, and understand basic machine learning models on your own CSV datasets. 
Upload your training and testing data, select your target variable, choose columns to drop, specify the problem type (classification or regression), and instantly get back model performance metrics, insightful visualizations, and downloadable predictions.

## Features

-   **Easy Data Upload:** Upload custom training and testing data in CSV format.
-   **Interactive Data Preview:** View the first 5 rows (`.head()`) of your training data to help with column selection.
-   **Column Management:**
    -   Select your target variable from a list of columns.
    -   Multi-select columns to drop/ignore before modeling (e.g., ID columns, irrelevant features).
-   **Problem Type Selection:** Clearly choose between "Classification" or "Regression" tasks.
-   **Automated Model Training & Evaluation:**
    -   **Classification Models:** Logistic Regression, Random Forest Classifier, Gradient Boosting Classifier.
    -   **Regression Models:** Linear Regression, Random Forest Regressor, Gradient Boosting Regressor.
-   **Robust Metrics:**
    -   **Cross-Validation Metrics:** Detailed CV scores (Accuracy, F1, Precision, Recall for classification; R2, RMSE for regression) are displayed for each model.
    -   **Summary Table:** A consolidated table showing key cross-validation metrics for all models, sorted by performance (Accuracy for classification, R2/RMSE for regression).
-   **Insightful Visualizations:**
    -   **Correlation Matrix:** Heatmap showing correlations between numerical features.
    -   **Feature Importances:** Bar plot displaying the most important features for tree-based models.
    -   **Prediction Plot (Regression):** Scatter plot of Actual vs. Predicted values to visually assess regression model performance.
-   **Downloadable Predictions:** For each model, download a single-column CSV file of its predictions on your test dataset.
-   **User-Friendly Interface:** Clean, minimalist web interface with a dark neon theme for clear readability.

## Technologies Used

-   **Backend:** Python, Flask
-   **Machine Learning & Data Handling:** Scikit-learn, Pandas, NumPy
-   **Plotting:** Matplotlib, Seaborn
-   **Frontend:** HTML, CSS, Vanilla JavaScript

## Setup and Installation

To run Simple ML locally on your machine, you will need Python (version 3.7 or newer is recommended) and pip (Python's package installer).

1.  **Download or Clone the Repository:**
    Obtain the project files (e.g., by downloading the ZIP from GitHub or cloning).

2.  **Navigate to the Project Directory:**
    Open your terminal or command prompt and navigate to the root folder of the project (where `app.py` and `requirements.txt` are located).
    ```bash
    cd path/to/simple-ml 
    ```

3.  **Create and Activate a Virtual Environment (Highly Recommended):**
    This keeps your project dependencies isolated.
    ```bash
    # For Linux/macOS:
    python3 -m venv venv
    source venv/bin/activate

    # For Windows:
    python -m venv venv
    venv\Scripts\activate
    ```
    You should see `(venv)` at the beginning of your terminal prompt.

4.  **Install Dependencies:**
    With the virtual environment activated, install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Application:**
    Ensure you are still in the project's root directory and your virtual environment is active. Then, run:
    ```bash
    python app.py
    ```

6.  **Access in Browser:**
    Open your web browser (e.g., Chrome, Firefox, Edge) and navigate to the address shown in your terminal. This will typically be:
    `http://127.0.0.1:5000`
    or
    `http://localhost:5000`

    You should now see the Simple ML application interface!

## How to Use Simple ML

1.  **Step 1: Load Training Data**
    -   Click the "Load Training CSV" button and select your training data file (must be in CSV format).

2.  **Step 2: Preview Data, Select Target & Columns to Drop**
    -   A preview of the first 5 rows of your training data will appear.
    -   Below the preview, you'll see a table listing all columns.
        -   **Target:** Select **one** radio button for the column you want to predict.
        -   **Drop:** Check the checkboxes for any columns you wish to exclude from the modeling process.

3.  **Step 3: Define Problem & Load Test Data**
    -   **Problem Type:** Select either "Classification" or "Regression" based on your target variable.
    -   Click the "Load Test CSV" button and select your test data file.

4.  **Step 4: Run Analysis**
    -   Click the "Run Analysis" button.

5.  **View Results:**
    -   The application will process your data and display:
        -   A **Model Summary** table with key cross-validation metrics.
        -   An overall **Correlation Matrix** plot.
        -   **Detailed Model Results** for each model (CV metrics, feature importance, regression prediction plot).
        -   A **"Download Predictions"** button for each model.

## Troubleshooting

-   **"No module named 'flask' (or other library)":** Ensure your virtual environment is activated and you have run `pip install -r requirements.txt`.
-   **Error messages:** Check the terminal running `python app.py` for detailed Python tracebacks.

## Future Enhancements (Ideas)

-   Support for more file types.
-   Advanced preprocessing options (e.g., date feature engineering).
-   Wider range of models and hyperparameter tuning.
-   More interactive visualizations.
-   Saving and loading analysis sessions.

## License

Distributed under the MIT License. See `LICENSE` file for more information.
