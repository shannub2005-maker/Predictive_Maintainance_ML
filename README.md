# Predictive Maintenance Using Machine Learning

## Project Overview
This project uses machine learning to predict machine failures based on operational and process parameters. The objective is to identify potential failures in advance and support preventive maintenance.

## Dataset
This project uses the **AI4I 2020 Predictive Maintenance Dataset**.

Dataset source: [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

Download the dataset and select the CSV file when prompted by the R script.

## Technologies Used
- R
- RStudio
- tidyverse
- caret
- randomForest
- rpart
- recipes
- themis
- pROC
- corrplot

## Machine Learning Models
The project trains and compares the following models:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Random Forest with SMOTE

## Project Workflow
1. Load the dataset
2. Perform exploratory data analysis (EDA)
3. Preprocess the data
4. Perform feature engineering
5. Split the data into training and testing sets
6. Train machine learning models
7. Evaluate model performance
8. Apply SMOTE to address class imbalance
9. Compare models and generate visualizations

## Evaluation Metrics
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Repository Structure
```text
Predictive-Maintenance-ML/
├── Predictive_Maintenance_Complete_Updated_Code.R
├── README.md
├── Predictive_Maintenance_Project_Report.pdf
├── visualizations/
├── results/
└── .gitignore
```

## How to Run
1. Install R and RStudio.
2. Download the AI4I 2020 dataset from the source linked above.
3. Open the R script in RStudio.
4. Run the script.
5. When prompted, select the downloaded dataset CSV file.
6. The script will create folders for data, results, and visualizations in the current working directory.

## Outputs
The script generates:
- Exploratory data analysis plots
- Model evaluation results
- Model comparison tables
- ROC and confusion matrix visualizations
- Feature importance visualization
- Saved model files

## Results
Add your final model comparison results here after running the complete script. Report the metrics from your latest RStudio output.

## Future Enhancements
- Build an interactive dashboard using Shiny.
- Deploy the trained model as a web application.
- Explore additional machine learning and deep learning approaches.
- Integrate real-time machine sensor data.

## Author
**B. Shanmukha Reddy**  
B.Tech – Data Science  
Presidency University, Bengaluru
