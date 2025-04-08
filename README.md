# Machine Learning Projects in Python

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains Python-based machine learning projects, exploring different domains and techniques.

## Project 1: Insurance Claim Prediction

### Project Overview

This project, focuses on building a machine learning model to predict insurance claim status for an insurance firm providing tour insurance. The firm is facing higher claim frequency, and the goal is to develop a predictive model to identify potential claims and provide recommendations to the management to mitigate losses.

### Dataset

The dataset used for this project contains information about insurance policies, including customer demographics, trip details, and claim status. It includes the following features:

* **Target Variable:**
    * `Claimed`: Claim Status (Claimed - Yes/No or 1/0)
* **Predictor Variables:**
    * `Agency_Code`: Code of tour firm (Alphanumeric code)
    * `Type`: Type of tour insurance firms (Categorical)
    * `Channel`: Distribution channel (Categorical)
    * `Product Name`: Name of the tour insurance products (Text)
    * `Duration`: Duration of the tour (Days, Numeric)
    * `Destination`: Destination of the tour (Text/Categorical)
    * `Sales`: Amount worth of sales per customer (Rupees in 100's, Numeric)
    * `Commission`: Commission received for tour insurance firm (Percentage of sales, Numeric)
    * `Age`: Age of insured (Years, Numeric)

### Methodology
The project follows a typical machine learning workflow:

1. **Data Loading and Cleaning:** Loading the dataset, handling missing values, and removing duplicates.
2. **Exploratory Data Analysis (EDA):** Performing univariate and bivariate analysis to understand data patterns, distributions, and relationships between variables.
3. **Feature Engineering:** Converting categorical variables to numerical using one-hot encoding.
4. **Model Building:** Training and evaluating three machine learning models:
    * Decision Tree Classifier (CART)
    * Random Forest Classifier (RFC)
    * Artificial Neural Network (ANN)
5. **Model Evaluation:** Assessing model performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
6. **Feature Importance Analysis:** Identifying the most influential features for predicting claim status.
7. **Business Insights and Recommendations:** Providing actionable insights and recommendations to the insurance firm based on the model results.

### Results
The Random Forest Classifier emerged as the most promising model for insurance claim prediction, achieving high accuracy and robust performance in this analysis.

**Key Findings:**

* **Important Features:** `Agency_Code`, `Duration`, and `Sales` were identified as the most important features for predicting claim likelihood.
* **Agency Risk:** Certain agencies exhibited higher claim frequencies, suggesting the need for risk-based pricing and risk management strategies.
* **Trip Duration:** Longer trips were associated with a greater chance of claims, prompting the need for tailored insurance products for different trip durations.
* **Sales Value:** Higher sales values were linked to increased claim amounts, indicating the importance of tiered coverage options based on sales value.

### Recommendations

The project provides several recommendations to the insurance firm, including:

* Implementing risk assessment strategies for agencies with higher claim ratios.
* Developing customized insurance products for various trip durations.
* Offering tiered coverage options based on sales value.
* Leveraging customer segmentation for targeted marketing and risk management.

### Conclusion
This project by **Aravindan Natarajan** demonstrates the potential of machine learning in predicting insurance claims and guiding business decisions. By utilizing the insights and recommendations from the analysis, the insurance firm can take proactive steps to mitigate losses and improve overall profitability.

### Usage

1. Clone the repository: `git clone <repository_url>`
2. Open the Jupyter notebook `insurance_claim_prediction.ipynb` in Google Colab.
3. Run the notebook cells sequentially to execute the analysis and model building steps.
4. Refer to the README for detailed explanations and insights.

---

## Project 2: Election Data Analysis and Prediction

### Project Overview

This project, aims to analyze election data and build a model to predict voter preferences. It uses a dataset of 1525 voters with 9 variables to predict which party a voter will vote for. The goal is to create an exit poll that can help predict the overall win and seats covered by a particular party.

### Project Structure

The project is organized as follows:

- **Election_Data.xlsx:** The dataset containing voter information.
- **Election_Prediction.ipynb:** The Jupyter Notebook containing the data analysis, model building, and evaluation.
- **README.md:** This file, providing an overview of the project.

### Data Description

The dataset contains the following variables:

- **Unnamed: 0:** An index column (dropped during analysis).
- **vote:** The party the voter voted for (target variable).
- **age:** The voter's age.
- **economic.cond.national:** The voter's perception of the national economic condition.
- **economic.cond.personal:** The voter's perception of their personal economic condition.
- **moral.values:** The voter's rating of moral values.
- **health.insurance:** Whether the voter has health insurance.
- **race:** The voter's race.
- **gender:** The voter's gender.
- **edu:** The voter's education level.

### Methodology

The project follows these steps:

1. **Data Loading and Cleaning:** The dataset is loaded, and the index column is dropped. Duplicate rows are removed.
2. **Exploratory Data Analysis (EDA):** Univariate and bivariate analyses are performed to understand the data distribution and relationships between variables. Outliers are identified and treated.
3. **Data Encoding and Scaling:** Categorical variables are encoded using label encoding. Data scaling is performed using z-score standardization for models that require it (e.g., KNN).
4. **Model Building:** Various classification models are applied, including:
    - Logistic Regression
    - Linear Discriminant Analysis (LDA)
    - K-Nearest Neighbors (KNN)
    - Naive Bayes
    - Random Forest
    - AdaBoost
    - Gradient Boosting
    - Support Vector Machine
    - XGBoost
    - Bagging Classifier
5. **Model Tuning:** Grid search is used to find the optimal hyperparameters for each model.
6. **Model Evaluation:** The performance of each model is evaluated using accuracy, confusion matrix, ROC curve, ROC-AUC score, and classification report.
7. **SMOTE:** Synthetic Minority Over-sampling Technique (SMOTE) is applied to address class imbalance in the dataset.
8. **Performance Metrics:** The models' performances are compared based on various metrics. The final model is selected based on its overall performance.

### Results

The project identifies the best-performing model for predicting voter preferences. The performance metrics are summarized in a tabular format, allowing for easy comparison. The final model is selected based on its accuracy, ROC-AUC score, and other relevant metrics.

### Conclusion

This project by provides valuable insights into election data and demonstrates the effectiveness of machine learning models for predicting voter behavior. The chosen model can be used to create an exit poll and assist in predicting election outcomes.

### Future Work

Future work could involve:

- Exploring more advanced feature engineering techniques.
- Implementing ensemble methods to combine multiple models.
- Deploying the model as a web application for real-time predictions.

### Author

- Aravindan Natarajan

---

## License

This repository and all its contents are licensed under the [MIT License](https://opensource.org/licenses/MIT).