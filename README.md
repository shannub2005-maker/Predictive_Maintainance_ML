# Predictive Maintenance in RStudio — Beginner Project

## What is included
- `main.R` — the complete beginner-friendly R script.
- `data/` — place the AI4I dataset CSV here. The script automatically downloads it from UCI the first time it runs, if it is not present.
- `output/` — model metrics, confusion matrix, plots and saved models are written here.

## Dataset
AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository.
- 10,000 observations and 14 columns.
- Target: `Machine failure`.
- License: CC BY 4.0.
- DOI: https://doi.org/10.24432/C5HS5C
- Dataset page: https://archive.ics.uci.edu/dataset/601/ai4i

The dataset is downloaded by `main.R` from the official UCI archive when required. Internet access is needed for the first run. If automatic download is blocked, download `ai4i2020.csv` from the UCI dataset page and place it in the `data` folder.

## How to run in RStudio
1. Extract this ZIP file to a folder on your computer.
2. Open RStudio.
3. Choose **File → Open File...** and open `main.R` from the extracted folder.
4. Set the RStudio working directory to the extracted project folder:
   **Session → Set Working Directory → To Source File Location**.
5. Click **Source** (or run the script from top to bottom).
6. If prompted, allow R to install the `rpart` package.
7. When it finishes, open the `output` folder.

## Models used
- Logistic Regression — a basic statistical classification model.
- Decision Tree — a rule-based model that is easy to visualize and explain.

No SMOTE, recipe pipelines, or advanced tuning are used in this simplified version. The script uses a stratified 80:20 train/test split and reports accuracy, precision, recall, specificity, F1-score, and a confusion matrix.

## Important notes
- The target `Machine failure` is converted to `Normal` and `Failure`.
- Identifier columns (`UDI`, `Product ID`) and failure-mode labels (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are not used as predictors.
- The script calculates mechanical power as torque × angular speed (in radians/second).
- Metrics are generated when you run the script; they are not pre-filled or guaranteed results.
- Because failures are rare in this dataset, do not judge the model by accuracy alone. Pay attention to recall and the confusion matrix.
