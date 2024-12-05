# Student Performance Prediction Project

This project aims to predict student academic performance using advanced machine learning techniques and comprehensive socioeconomic feature analysis. Leveraging both regression and classification models, the project not only forecasts final grades but also classifies students as pass or fail. Additionally, an interactive web application has been developed to facilitate real-time predictions and insights for educators and administrators. 

The data used in this project comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance).

## Project Overview

Predicting student academic performance is crucial for educational institutions to identify at-risk students, allocate resources effectively, and tailor educational experiences to individual needs. This project enhances traditional predictive models by integrating socioeconomic and demographic variables with advanced machine learning algorithms to improve prediction accuracy and reliability.

### Key Objectives

- **Early Intervention Programs:** Identify students needing additional support.
- **Resource Allocation:** Optimize the distribution of educational resources.
- **Policy Development:** Inform policies addressing factors affecting student performance.
- **Personalized Learning:** Customize educational experiences based on predictive outcomes.

<br/>

# Usage Instructions

## 1. Setting Up the Environment

### Clone the Repository:
```bash
git clone https://github.com/KhoaDao03/CAP5771_DataMining.git
cd CAP5771_DataMining
```

### Create a Virtual Environment (Optional but Recommended):
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

---

## Note: To run everything at once:

```bash
python script.py
```

## 2. Running Data Preprocessing

Execute the data preprocessing script to prepare the data for modeling:
```bash
python src/data_preprocessing.py
```

---

## 3. Training and Evaluating Models

Train and evaluate each machine learning model by running the corresponding scripts:

- **Linear Regression:**
    ```bash
    python src/linear_regression.py
    ```

- **Logistic Regression:**
    ```bash
    python src/logistic_regression.py
    ```

- **Decision Tree:**
    ```bash
    python src/decision_tree.py
    ```

- **Random Forest:**
    ```bash
    python src/random_forest.py
    ```

- **Support Vector Machines (SVM):**
    ```bash
    python src/svm.py
    ```

- **Artificial Neural Networks (ANN):**
    ```bash
    python src/neural_network.py
    ```

- **XGBoost:**
    ```bash
    python src/xgboost_model.py
    ```

- **K-Means Clustering:**
    ```bash
    python src/k_means_clustering.py
    ```

Each script will train the respective model, evaluate its performance, generate visualizations, and save the trained model in the `outputs/models/` directory.

---

## 4. Launching the Web Application

Start the Streamlit web application to interactively predict student performance:
```bash
streamlit run app.py
```

Access the application by navigating to `http://localhost:8501` in your web browser.

---

## 5. Exploring the Outputs

- **Models:** Trained models are saved in the `outputs/models/` directory.
- **Visualizations:** All generated plots and figures are stored in the `outputs/figures/` directory.
- **Preprocessed Data:** Access the preprocessed datasets in the `outputs/` folder.
- **Encoders and Scalers:** Label encoders and scaler objects are located in the `outputs/encoders/` directory.


<br/>
<br/>

# Description of the project

## 1. Data Preprocessing

Data preprocessing is the foundational step in preparing raw datasets for modeling. This involves:
- Cleaning the data
- Handling missing values
- Encoding categorical variables
- Scaling numerical features
- Splitting the data into training and testing sets for regression and classification tasks.

### Script: `data_preprocessing.py`

#### Functionality:

1. **Loading Data**:
   - Reads the Math (`student-mat.csv`) and Portuguese (`student-por.csv`) datasets.
   
2. **Merging Datasets**:
   - Merges both datasets based on common student attributes.
   - For students appearing in both datasets, it averages their grades.

3. **Preprocessing**:
   - **Handling Missing Values**: Replaces missing values ('?') with NaN and drops rows with missing values.
   - **Encoding Categorical Variables**: Uses `LabelEncoder` to convert categorical variables into numerical formats.
   - **Feature Scaling**: Applies `StandardScaler` to normalize numerical features.

4. **Creating Classification Target**:
   - A binary target column is created where 1 indicates a passing grade (G3 ≥ 10) and 0 indicates failing.

5. **Splitting Data**:
   - Splits the data into training and testing sets for both **regression** and **classification** tasks.

6. **Saving Preprocessed Data**:
   - The preprocessed data and the training/testing splits are saved in the `outputs/` folder for future use.

#### Running the Script:

To run the preprocessing:

```bash
python src/data_preprocessing.py
```

**Outputs:**
- Preprocessed data (`outputs/preprocessed_data.csv`)
- Training and testing splits for regression and classification
- Encoders and scalers

---

## 2. Machine Learning Models

### Linear Regression Model
**Script:** `linear_regression.py`  
Predicts the final grade (`G3`) of students.

**Running the Script:**
```bash
python src/linear_regression.py
```

**Outputs:**
- Model: `outputs/models/linear_regression_model.pkl`
- Visualizations: `outputs/figures/linear_regression_actual_vs_predicted.png`

---

### Logistic Regression Model
**Script:** `logistic_regression.py`  
Classifies students into pass/fail categories.

**Running the Script:**
```bash
python src/logistic_regression.py
```

**Outputs:**
- Model: `outputs/models/logistic_regression_model.pkl`
- Visualizations: Confusion Matrix, ROC Curve

---

### Decision Tree Model
**Script:** `decision_tree.py`  
Trains both regression and classification Decision Tree models.

**Running the Script:**
```bash
python src/decision_tree.py
```

**Outputs:**
- Models for regression and classification
- Various visualizations including decision tree structures

---

### Random Forest Model
**Script:** `random_forest.py`  
Builds ensemble models for regression and classification.

**Running the Script:**
```bash
python src/random_forest.py
```

**Outputs:**
- Models for regression and classification
- Visualizations of trees and feature importance

---

### Support Vector Machines (SVM) Model
**Script:** `svm.py`  
Implements SVM models for regression and classification.

**Running the Script:**
```bash
python src/svm.py
```

**Outputs:**
- Models for regression and classification
- Decision boundary visualizations

---

### Artificial Neural Networks (ANN) Model
**Script:** `neural_network.py`  
Uses deep learning for nonlinear data relationships.

**Running the Script:**
```bash
python src/neural_network.py
```

**Outputs:**
- Models: Saved in `.h5` format
- Visualizations: Accuracy and ROC curves

---

### XGBoost Model
**Script:** `xgboost_model.py`  
Employs gradient boosting for high performance.

**Running the Script:**
```bash
python src/xgboost_model.py
```

**Outputs:**
- Models for regression and classification
- SHAP and feature importance visualizations

---

### K-Means Clustering
**Script:** `k_means_clustering.py`  
Clusters students into groups based on their attributes.

**Running the Script:**
```bash
python src/k_means_clustering.py
```

**Outputs:**
- Model: `outputs/models/kmeans_model.pkl`
- Cluster visualizations


## 3. Web Application Deployment

## 3. Web Application Deployment

The student performance prediction models are deployed in an interactive web application built with **Streamlit**, enabling real-time predictions based on user-inputted student data.

### 3.1 Features
- **Intuitive Interface:** Input student data easily using a sidebar.
- **Model Predictions:** Predicts academic performance using XGBoost, Random Forest, Decision Trees, Logistic Regression, SVM, and ANN.
- **Real-Time Feedback:** Instantly displays predicted grades and pass/fail status with confidence scores.
- **Transparency:** Shows preprocessed input data for clarity.
- **Future-Ready:** Framework for integrating SHAP-based model interpretability tools.

### 3.2 Implementation Details
- **Framework:** Built with Streamlit for a lightweight, efficient interface.
- **Model Loading:** Utilizes `joblib` for traditional models and TensorFlow/Keras for ANN.
- **Preprocessing:** Ensures input data undergoes the same preprocessing as training data.
- **Pipeline:** Processes input data to generate predictions for both regression (final grades) and classification (pass/fail).


---

### Running the Web Application
To run the web application locally, execute the following command:
```bash
streamlit run src/web_app.py
```

Ensure that all dependencies are installed, and the trained models are located in the `outputs/models/` directory as outlined above.


## Requirements

Make sure to install the required Python packages before running the scripts:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include the following dependencies:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
tensorflow
keras
joblib
streamlit
os-sys
shap
warnings
pydot
statsmodels
graphviz
```


## Project Structure

The project is structured into several components, each with its own functionality:

- **`data_preprocessing.py`**: Preprocesses the raw data to prepare it for modeling.
- **`linear_regression.py`**: Builds and evaluates a linear regression model to predict student grades (G3).
- **`logistic_regression.py`**: Builds and evaluates a logistic regression model to classify student pass/fail status.

### Directory Structure

```
│   app.py                          # Streamlit web application for student performance prediction
│   README.md                       # Project documentation and overview
│   requirements.txt                # List of Python dependencies required to run the project
│   script.py                       # script file to run the whole project   
|
├── .idea
│   │   .gitignore                   # Specifies intentionally untracked files to ignore
│   │   CAP5771_DataMining.iml        # IntelliJ IDEA module file containing project configuration
│   │   misc.xml                     # Miscellaneous IDE settings
│   │   modules.xml                  # Configuration for project modules in the IDE
│   │   vcs.xml                      # Version Control System settings for the IDE
│   │   
│   └── inspectionProfiles
│           profiles_settings.xml     # Custom inspection profiles settings for code analysis
│           Project_Default.xml       # Default inspection profiles provided by the IDE
│           
├── data
│       student-mat.csv              # Raw dataset containing Math-related student data
│       student-por.csv              # Raw dataset containing Portuguese-related student data
│       student-merge.R              # R script used to merge and preprocess the Math and Portuguese datasets
│       student.txt                  # Text file with additional information about the datasets or project specifics
│       
├── outputs
│   │   label_encoder_activities.pkl    # Label encoder for the 'activities' categorical feature
│   │   label_encoder_address.pkl        # Label encoder for the 'address' categorical feature
│   │   label_encoder_famsize.pkl        # Label encoder for the 'famsize' categorical feature
│   │   label_encoder_famsup.pkl         # Label encoder for the 'famsup' categorical feature
│   │   label_encoder_Fjob.pkl           # Label encoder for the 'Fjob' categorical feature
│   │   label_encoder_guardian.pkl       # Label encoder for the 'guardian' categorical feature
│   │   label_encoder_higher.pkl         # Label encoder for the 'higher' categorical feature
│   │   label_encoder_internet.pkl       # Label encoder for the 'internet' categorical feature
│   │   label_encoder_Mjob.pkl           # Label encoder for the 'Mjob' categorical feature
│   │   label_encoder_nursery.pkl        # Label encoder for the 'nursery' categorical feature
│   │   label_encoder_paid.pkl           # Label encoder for the 'paid' categorical feature
│   │   label_encoder_Pstatus.pkl        # Label encoder for the 'Pstatus' categorical feature
│   │   label_encoder_reason.pkl         # Label encoder for the 'reason' categorical feature
│   │   label_encoder_romantic.pkl       # Label encoder for the 'romantic' categorical feature
│   │   label_encoder_school.pkl         # Label encoder for the 'school' categorical feature
│   │   label_encoder_schoolsup.pkl      # Label encoder for the 'schoolsup' categorical feature
│   │   label_encoder_sex.pkl            # Label encoder for the 'sex' categorical feature
│   │   preprocessed_data.csv            # Complete preprocessed dataset ready for modeling
│   │   scaler.pkl                       # Scaler object used for normalizing numerical features
│   │   X_test_clf.csv                   # Testing feature set for classification models
│   │   X_test_reg.csv                   # Testing feature set for regression models
│   │   X_train_clf.csv                  # Training feature set for classification models
│   │   X_train_reg.csv                  # Training feature set for regression models
│   │   y_test_clf.csv                   # Testing target labels (pass/fail) for classification models
│   │   y_test_reg.csv                   # Testing target labels (G3 final grade) for regression models
│   │   y_train_clf.csv                  # Training target labels (pass/fail) for classification models
│   │   y_train_reg.csv                  # Training target labels (G3 final grade) for regression models
│   │   
│   ├── encoders
│   │       activities_encoder.pkl          # Saved encoder for 'activities' feature
│   │       address_encoder.pkl              # Saved encoder for 'address' feature
│   │       famsize_encoder.pkl              # Saved encoder for 'famsize' feature
│   │       famsup_encoder.pkl               # Saved encoder for 'famsup' feature
│   │       Fjob_encoder.pkl                 # Saved encoder for 'Fjob' feature
│   │       guardian_encoder.pkl             # Saved encoder for 'guardian' feature
│   │       higher_encoder.pkl               # Saved encoder for 'higher' feature
│   │       internet_encoder.pkl             # Saved encoder for 'internet' feature
│   │       label_encoder_activities.pkl      # Additional encoder for 'activities'
│   │       label_encoder_address.pkl          # Additional encoder for 'address'
│   │       label_encoder_famsize.pkl          # Additional encoder for 'famsize'
│   │       label_encoder_famsup.pkl           # Additional encoder for 'famsup'
│   │       label_encoder_Fjob.pkl             # Additional encoder for 'Fjob'
│   │       label_encoder_guardian.pkl         # Additional encoder for 'guardian'
│   │       label_encoder_higher.pkl           # Additional encoder for 'higher'
│   │       label_encoder_internet.pkl         # Additional encoder for 'internet'
│   │       label_encoder_Mjob.pkl             # Additional encoder for 'Mjob'
│   │       label_encoder_nursery.pkl          # Additional encoder for 'nursery'
│   │       label_encoder_paid.pkl             # Additional encoder for 'paid'
│   │       label_encoder_Pstatus.pkl          # Additional encoder for 'Pstatus'
│   │       label_encoder_reason.pkl           # Additional encoder for 'reason'
│   │       label_encoder_romantic.pkl         # Additional encoder for 'romantic'
│   │       label_encoder_school.pkl           # Additional encoder for 'school'
│   │       label_encoder_schoolsup.pkl        # Additional encoder for 'schoolsup'
│   │       label_encoder_sex.pkl              # Additional encoder for 'sex'
│   │       Mjob_encoder.pkl                   # Additional encoder for 'Mjob'
│   │       nursery_encoder.pkl                # Additional encoder for 'nursery'
│   │       paid_encoder.pkl                   # Additional encoder for 'paid'
│   │       Pstatus_encoder.pkl                # Additional encoder for 'Pstatus'
│   │       reason_encoder.pkl                 # Additional encoder for 'reason'
│   │       romantic_encoder.pkl               # Additional encoder for 'romantic'
│   │       scaler.pkl                         # Additional scaler
│   │       schoolsup_encoder.pkl              # Additional encoder for 'schoolsup'
│   │       school_encoder.pkl                 # Additional encoder for 'school'
│   │       sex_encoder.pkl                    # Additional encoder for 'sex'
│   │       
│   ├── figures
│   │       ann_classification_confusion_matrix.png       # Confusion matrix for ANN classification model
│   │       ann_classification_roc_curve.png              # ROC curve for ANN classification model
│   │       ann_regression_actual_vs_predicted.png         # Actual vs. predicted grades for ANN regression model
│   │       decision_tree_classification_confusion_matrix.png  # Confusion matrix for Decision Tree classification model
│   │       decision_tree_classification_roc_curve.png         # ROC curve for Decision Tree classification model
│   │       decision_tree_classification_tree_visualization.png  # Visual representation of Decision Tree classification model
│   │       decision_tree_regression_actual_vs_predicted.png      # Actual vs. predicted grades for Decision Tree regression model
│   │       decision_tree_regression_tree_visualization.png         # Visual representation of Decision Tree regression model
│   │       kmeans_clustering_3_clusters.png                       # Visualization of K-Means clustering with 3 clusters
│   │       linear_regression_actual_vs_predicted.png                # Actual vs. predicted grades for Linear Regression model
│   │       logistic_regression_confusion_matrix.png                 # Confusion matrix for Logistic Regression model
│   │       logistic_regression_roc_curve.png                        # ROC curve for Logistic Regression model
│   │       random_forest_classification_confusion_matrix.png        # Confusion matrix for Random Forest classification model
│   │       random_forest_classification_roc_curve.png               # ROC curve for Random Forest classification model
│   │       random_forest_multiple_trees.png                          # Visualization of multiple trees in the Random Forest model
│   │       random_forest_regression_actual_vs_predicted.png           # Actual vs. predicted grades for Random Forest regression model
│   │       random_forest_tree_0.png                                  # Visualization of the first tree in the Random Forest model
│   │       svm_classification_confusion_matrix.png                    # Confusion matrix for SVM classification model
│   │       svm_classification_roc_curve.png                           # ROC curve for SVM classification model
│   │       svm_decision_boundary.png                                  # Decision boundary visualization for SVM model
│   │       svm_regression_actual_vs_predicted.png                      # Actual vs. predicted grades for SVM regression model
│   │       svm_regression_residuals.png                                # Residuals plot for SVM regression model
│   │       svm_support_vectors.png                                     # Visualization of support vectors in SVM model
│   │       xgboost_classification_confusion_matrix.png                 # Confusion matrix for XGBoost classification model
│   │       xgboost_classification_learning_curve.png                   # Learning curve for XGBoost classification model
│   │       xgboost_classification_roc_curve.png                        # ROC curve for XGBoost classification model
│   │       xgboost_feature_importance.png                              # Feature importance plot for XGBoost model
│   │       xgboost_regression_actual_vs_predicted.png                   # Actual vs. predicted grades for XGBoost regression model
│   │       xgboost_regression_learning_curve.png                         # Learning curve for XGBoost regression model
│   │       xgboost_residuals.png                                         # Residuals plot for XGBoost regression model
│   │       xgboost_shap_summary.png                                      # SHAP summary plot for XGBoost model
│   │       
│   ├── models
│   │       ann_classification_model.h5                  # Trained Artificial Neural Network for classification
│   │       ann_classification_model.h5_scaler.pkl       # Scaler associated with ANN classification model
│   │       ann_regression_model.h5                      # Trained Artificial Neural Network for regression
│   │       ann_regression_model.h5_scaler.pkl           # Scaler associated with ANN regression model
│   │       decision_tree_classification_model.pkl       # Trained Decision Tree for classification
│   │       decision_tree_regression_model.pkl           # Trained Decision Tree for regression
│   │       kmeans_model.pkl                             # Trained K-Means clustering model
│   │       linear_regression_model.pkl                  # Trained Linear Regression model
│   │       logistic_regression_model.pkl                # Trained Logistic Regression model
│   │       random_forest_classification_model.pkl       # Trained Random Forest for classification
│   │       random_forest_regression_model.pkl           # Trained Random Forest for regression
│   │       svm_classification_model.pkl                  # Trained Support Vector Machine for classification
│   │       svm_regression_model.pkl                      # Trained Support Vector Machine for regression
│   │       xgboost_classification_model.pkl             # Trained XGBoost model for classification
│   │       xgboost_regression_model.pkl                 # Trained XGBoost model for regression
│   │       
│   └── results
│           xgboost_classification_results.csv           # Evaluation results for XGBoost classification model
│           xgboost_regression_results.csv               # Evaluation results for XGBoost regression model
│           
└── src
        data_preprocessing.py                      # Script to preprocess raw data, handle encoding and scaling
        decision_tree.py                           # Script to train and evaluate Decision Tree models
        k_means_clustering.py                      # Script to perform K-Means clustering analysis
        linear_regression.py                       # Script to train and evaluate Linear Regression model
        logistic_regression.py                     # Script to train and evaluate Logistic Regression model
        neural_network.py                          # Script to train and evaluate Artificial Neural Network models
        random_forest.py                           # Script to train and evaluate Random Forest models
        svm.py                                     # Script to train and evaluate Support Vector Machine models
        xgboost_model.py                           # Script to train and evaluate XGBoost models
```



## Conclusion

### Summary of Findings
- **Top-Performing Models:** Random Forest and XGBoost demonstrated superior accuracy for regression and classification tasks.
- **Key Influential Factors:** Parental education levels and study time were significant predictors of performance.
- **Clustering Insights:** K-Means clustering revealed distinct student profiles, suggesting potential for personalized education strategies.

### Practical Considerations
- **Model Selection:** Ensemble methods like Random Forest and XGBoost effectively handle complex educational data.
- **Feature Importance:** Identifying key factors enables targeted educational interventions.
- **Scalability:** Balancing model accuracy and computational efficiency ensures practical deployment.

### Future Research Directions
- **Additional Data:** Incorporate attendance, behavior, and extracurricular activities to refine predictions.
- **Model Interpretability:** Expand the use of SHAP values to improve model transparency.
- **Real-Time Deployment:** Build a real-time prediction system for continuous student support.
- **Longitudinal Studies:** Evaluate the long-term effectiveness of predictive models across multiple academic years.

## Final Remarks
This project highlights the importance of integrating academic, socioeconomic, and demographic data with advanced machine learning techniques. By providing an interactive web application, educational institutions can make informed, data-driven decisions to support student success and optimize resources.
