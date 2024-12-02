# To run the file run:
# python -m streamlit run app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import tensorflow as tf

def load_models(project_dir):
    """
    Load the trained models.
    """
    models_dir = os.path.join(project_dir, 'outputs', 'models')
    
    # Load XGBoost models
    regression_model_xgb = joblib.load(os.path.join(models_dir, 'xgboost_regression_model.pkl'))
    classification_model_xgb = joblib.load(os.path.join(models_dir, 'xgboost_classification_model.pkl'))
    
    # Load ANN models
    regression_model_ann = tf.keras.models.load_model(os.path.join(models_dir, 'ann_regression_model.h5'))
    classification_model_ann = tf.keras.models.load_model(os.path.join(models_dir, 'ann_classification_model.h5'))
    
    # Load Random Forest models
    regression_model_rf = joblib.load(os.path.join(models_dir, 'random_forest_regression_model.pkl'))
    classification_model_rf = joblib.load(os.path.join(models_dir, 'random_forest_classification_model.pkl'))
    
    # Load Decision Tree models
    regression_model_dt = joblib.load(os.path.join(models_dir, 'decision_tree_regression_model.pkl'))
    classification_model_dt = joblib.load(os.path.join(models_dir, 'decision_tree_classification_model.pkl'))
    
    # Load Linear Regression model
    regression_model_lr = joblib.load(os.path.join(models_dir, 'linear_regression_model.pkl'))
    
    # Load Logistic Regression model
    classification_model_logr = joblib.load(os.path.join(models_dir, 'logistic_regression_model.pkl'))
    
    # Load SVM models
    svm_reg = joblib.load(os.path.join(models_dir, 'svm_regression_model.pkl'))
    regression_model_svm = svm_reg['model']
    scaler_reg_svm = svm_reg['scaler']
    
    svm_clf = joblib.load(os.path.join(models_dir, 'svm_classification_model.pkl'))
    classification_model_svm = svm_clf['model']
    scaler_clf_svm = svm_clf['scaler']
    
    return (regression_model_xgb, classification_model_xgb, 
            regression_model_ann, classification_model_ann,
            regression_model_rf, classification_model_rf,
            regression_model_dt, classification_model_dt,
            regression_model_lr, classification_model_logr,
            regression_model_svm, classification_model_svm,
            scaler_reg_svm, scaler_clf_svm)

def load_preprocessors(project_dir):
    """
    Load the label encoders and scaler.
    """
    outputs_dir = os.path.join(project_dir, 'outputs')
    encoders = {}
    categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus',
                        'Mjob', 'Fjob', 'reason', 'guardian',
                        'schoolsup', 'famsup', 'paid',
                        'activities', 'nursery', 'higher',
                        'internet', 'romantic']
    for col in categorical_cols:
        encoder_path = os.path.join(outputs_dir, f'label_encoder_{col}.pkl')
        encoders[col] = joblib.load(encoder_path)
    scaler = joblib.load(os.path.join(outputs_dir, 'scaler.pkl'))
    return encoders, scaler

def main():
    st.title("Student Performance Prediction App")
    st.write("Input the student's features to get the predicted final grade and pass/fail status from all models.")
    
    # Determine the absolute path to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '.'))
    
    # Load models
    (regression_model_xgb, classification_model_xgb, 
     regression_model_ann, classification_model_ann,
     regression_model_rf, classification_model_rf,
     regression_model_dt, classification_model_dt,
     regression_model_lr, classification_model_logr,
     regression_model_svm, classification_model_svm,
     scaler_reg_svm, scaler_clf_svm) = load_models(project_dir)
    
    # Load encoders and scaler
    encoders, scaler = load_preprocessors(project_dir)
    
    st.sidebar.header('Input Features')

    # 1. school
    # school = st.sidebar.selectbox('School', options=['GP', 'MS'])
    school = st.sidebar.selectbox(
        'School',
        options=['GP', 'MS'],
        index=0,
        help='"GP" - Gabriel Pereira or "MS" - Mousinho da Silveira'
    )

     # Feature Inputs
    age = st.sidebar.slider(
        'Age',
        min_value=15,
        max_value=22,
        value=17,
        help='Select the student\'s age in years.'
    )
    
    sex = st.sidebar.selectbox(
        'Sex',
        options=['Female', 'Male'],
        index=0,
        help='Select the student\'s biological sex.'
    )
    
    address = st.sidebar.selectbox(
        'Home Address Type',
        options=['Urban', 'Rural'],
        index=0,
        help='Urban: student lives in the city; Rural: student lives in the countryside.'
    )
    
    famsize = st.sidebar.selectbox(
        'Family Size',
        options=['Less than 3', '3 or more'],
        index=1,
        help='Family size: Less than 3 or 3 or more members.'
    )
    
    Pstatus = st.sidebar.selectbox(
        'Parent\'s Cohabitation Status',
        options=['Living Together', 'Apart'],
        index=0,
        help='Parent\'s cohabitation status: Living together or apart.'
    )
    
    Medu = st.sidebar.slider(
        'Mother\'s Education Level',
        min_value=0,
        max_value=4,
        value=2,
        help='Mother\'s education level (0: none, 1: primary, 2: 5th-9th grade, 3: secondary, 4: higher).'
    )
    
    Fedu = st.sidebar.slider(
        'Father\'s Education Level',
        min_value=0,
        max_value=4,
        value=2,
        help='Father\'s education level (0: none, 1: primary, 2: 5th-9th grade, 3: secondary, 4: higher).'
    )
    
    Mjob = st.sidebar.selectbox(
        'Mother\'s Job',
        options=['At Home', 'Health Care', 'Civil Services', 'Other', 'Teacher'],
        index=0,
        help='Mother\'s occupation.'
    )
    
    Fjob = st.sidebar.selectbox(
        'Father\'s Job',
        options=['At Home', 'Health Care', 'Civil Services', 'Other', 'Teacher'],
        index=0,
        help='Father\'s occupation.'
    )
    
    reason = st.sidebar.selectbox(
        'Reason for Choosing School',
        options=['Close to Home', 'School Reputation', 'Course Preference', 'Other'],
        index=0,
        help='Primary reason for choosing the school.'
    )
    
    guardian = st.sidebar.selectbox(
        'Student\'s Guardian',
        options=['Mother', 'Father', 'Other'],
        index=0,
        help='Person responsible for the student.'
    )
    
    traveltime = st.sidebar.slider(
        'Home to School Travel Time',
        min_value=1,
        max_value=4,
        value=1,
        help='Travel time from home to school (1: <15 min, 2: 15-30 min, 3: 30 min-1 hour, 4: >1 hour).'
    )
    
    studytime = st.sidebar.slider(
        'Weekly Study Time',
        min_value=1,
        max_value=4,
        value=2,
        help='Weekly study time (1: <2 hours, 2: 2-5 hours, 3: 5-10 hours, 4: >10 hours).'
    )
    
    failures = st.sidebar.slider(
        'Number of Past Class Failures',
        min_value=0,
        max_value=3,
        value=0,
        help='Number of past class failures (0: none, 1: one, 2: two, 3: three or more).'
    )
    
    schoolsup = st.sidebar.selectbox(
        'Extra Educational Support',
        options=['Yes', 'No'],
        index=1,
        help='Whether the student receives extra educational support.'
    )
    
    famsup = st.sidebar.selectbox(
        'Family Educational Support',
        options=['Yes', 'No'],
        index=1,
        help='Whether the student receives family educational support.'
    )
    
    paid = st.sidebar.selectbox(
        'Extra Paid Classes',
        options=['Yes', 'No'],
        index=1,
        help='Whether the student attends extra paid classes.'
    )
    
    activities = st.sidebar.selectbox(
        'Extra-Curricular Activities',
        options=['Yes', 'No'],
        index=1,
        help='Whether the student participates in extra-curricular activities.'
    )
    
    nursery = st.sidebar.selectbox(
        'Attended Nursery School',
        options=['Yes', 'No'],
        index=0,
        help='Whether the student attended nursery school.'
    )
    
    higher = st.sidebar.selectbox(
        'Desire for Higher Education',
        options=['Yes', 'No'],
        index=0,
        help='Whether the student wants to pursue higher education.'
    )
    
    internet = st.sidebar.selectbox(
        'Internet Access at Home',
        options=['Yes', 'No'],
        index=0,
        help='Whether the student has internet access at home.'
    )
    
    romantic = st.sidebar.selectbox(
        'In a Romantic Relationship',
        options=['Yes', 'No'],
        index=1,
        help='Whether the student is in a romantic relationship.'
    )
    
    famrel = st.sidebar.slider(
        'Quality of Family Relationships',
        min_value=1,
        max_value=5,
        value=4,
        help='Quality of family relationships (1: very bad to 5: excellent).'
    )
    
    freetime = st.sidebar.slider(
        'Free Time after School',
        min_value=1,
        max_value=5,
        value=3,
        help='Amount of free time after school (1: very low to 5: very high).'
    )
    
    goout = st.sidebar.slider(
        'Going Out with Friends',
        min_value=1,
        max_value=5,
        value=3,
        help='Frequency of going out with friends (1: very low to 5: very high).'
    )
    
    Dalc = st.sidebar.slider(
        'Workday Alcohol Consumption',
        min_value=1,
        max_value=5,
        value=1,
        help='Workday alcohol consumption (1: very low to 5: very high).'
    )
    
    Walc = st.sidebar.slider(
        'Weekend Alcohol Consumption',
        min_value=1,
        max_value=5,
        value=1,
        help='Weekend alcohol consumption (1: very low to 5: very high).'
    )
    
    health = st.sidebar.slider(
        'Current Health Status',
        min_value=1,
        max_value=5,
        value=5,
        help='Current health status (1: very bad to 5: very good).'
    )
    
    absences = st.sidebar.slider(
        'Number of School Absences',
        min_value=0,
        max_value=93,
        value=4,
        help='Total number of school absences.'
    )
    
    G1 = st.sidebar.slider(
        'First Period Grade (G1)',
        min_value=0,
        max_value=20,
        value=10,
        help='Grade in the first period (0-20 scale).'
    )
    
    G2 = st.sidebar.slider(
        'Second Period Grade (G2)',
        min_value=0,
        max_value=20,
        value=10,
        help='Grade in the second period (0-20 scale).'
    )

    # Collect all inputs into a dictionary
    input_dict = {
        'school': school,
        'sex': sex,
        'age': age,
        'address': address,
        'famsize': famsize,
        'Pstatus': Pstatus,
        'Medu': Medu,
        'Fedu': Fedu,
        'Mjob': Mjob,
        'Fjob': Fjob,
        'reason': reason,
        'guardian': guardian,
        'traveltime': traveltime,
        'studytime': studytime,
        'failures': failures,
        'schoolsup': schoolsup,
        'famsup': famsup,
        'paid': paid,
        'activities': activities,
        'nursery': nursery,
        'higher': higher,
        'internet': internet,
        'romantic': romantic,
        'famrel': famrel,
        'freetime': freetime,
        'goout': goout,
        'Dalc': Dalc,
        'Walc': Walc,
        'health': health,
        'absences': absences,
        'G1': G1,
        'G2': G2
    }

    # Convert the dictionary to a DataFrame
    input_data = pd.DataFrame(input_dict, index=[0])
    
    # Map user-friendly options back to original codes
    binary_mapping = {'Yes': 'yes', 'No': 'no'}
    for col in ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']:
        input_data[col] = input_data[col].map(binary_mapping)
    
    famsize_mapping = {'Less than 3': 'LE3', '3 or more': 'GT3'}
    input_data['famsize'] = input_data['famsize'].map(famsize_mapping)
    
    Pstatus_mapping = {'Living Together': 'T', 'Apart': 'A'}
    input_data['Pstatus'] = input_data['Pstatus'].map(Pstatus_mapping)
    
    address_mapping = {'Urban': 'U', 'Rural': 'R'}
    input_data['address'] = input_data['address'].map(address_mapping)
    
    sex_mapping = {'Female': 'F', 'Male': 'M'}
    input_data['sex'] = input_data['sex'].map(sex_mapping)
    
    job_mapping = {
        'At Home': 'at_home',
        'Health Care': 'health',
        'Civil Services': 'services',
        'Other': 'other',
        'Teacher': 'teacher'
    }
    input_data['Mjob'] = input_data['Mjob'].map(job_mapping)
    input_data['Fjob'] = input_data['Fjob'].map(job_mapping)
    
    reason_mapping = {
        'Close to Home': 'home',
        'School Reputation': 'reputation',
        'Course Preference': 'course',
        'Other': 'other'
    }
    input_data['reason'] = input_data['reason'].map(reason_mapping)
    
    guardian_mapping = {'Mother': 'mother', 'Father': 'father', 'Other': 'other'}
    input_data['guardian'] = input_data['guardian'].map(guardian_mapping)
    
    # Encode categorical variables using the loaded LabelEncoders
    categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus',
                        'Mjob', 'Fjob', 'reason', 'guardian',
                        'schoolsup', 'famsup', 'paid',
                        'activities', 'nursery', 'higher',
                        'internet', 'romantic']

    for col in categorical_cols:
        le = encoders[col]
        input_data[col] = le.transform([input_data[col].iloc[0]])
    
    # Numeric columns to scale
    numeric_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime',
                    'failures', 'famrel', 'freetime', 'goout',
                    'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']

    # Apply scaling to numeric columns
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
    
    # Define expected columns
    expected_columns = [
        'school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
        'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
        'traveltime', 'studytime', 'failures', 'schoolsup',
        'famsup', 'paid', 'activities', 'nursery', 'higher',
        'internet', 'romantic', 'famrel', 'freetime', 'goout',
        'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2'
    ]

    # Ensure all expected columns are present and in the correct order
    input_data = input_data[expected_columns]
      # Display input data for debugging
    st.write("Preprocessed Input Data:")
    st.write(input_data)
    # Make copies for SVM models
    input_data_svm_reg = input_data.copy()
    input_data_svm_clf = input_data.copy()
    
    # Scale all features for SVM regression model
    input_data_svm_reg = scaler_reg_svm.transform(input_data_svm_reg)
    
    # Scale all features for SVM classification model
    input_data_svm_clf = scaler_clf_svm.transform(input_data_svm_clf)
    
    # Prediction
    if st.button('Predict'):
        # XGBoost Regression Prediction
        reg_prediction_xgb = regression_model_xgb.predict(input_data)[0]
        
        # Neural Network Regression Prediction
        reg_prediction_ann = regression_model_ann.predict(input_data)[0][0]
        
        # Random Forest Regression Prediction
        reg_prediction_rf = regression_model_rf.predict(input_data)[0]
        
        # Decision Tree Regression Prediction
        reg_prediction_dt = regression_model_dt.predict(input_data)[0]
        
        # Linear Regression Prediction
        reg_prediction_lr = regression_model_lr.predict(input_data)[0]
        
        # SVM Regression Prediction
        reg_prediction_svm = regression_model_svm.predict(input_data_svm_reg)[0]
        
        st.subheader("Predicted Final Grade (G3)")
        st.write(f"**XGBoost Prediction:** {reg_prediction_xgb:.2f} out of 20.")
        st.write(f"**Neural Network Prediction:** {reg_prediction_ann:.2f} out of 20.")
        st.write(f"**Random Forest Prediction:** {reg_prediction_rf:.2f} out of 20.")
        st.write(f"**Decision Tree Prediction:** {reg_prediction_dt:.2f} out of 20.")
        st.write(f"**Linear Regression Predictions:** {', '.join([f'{x:.2f}' for x in reg_prediction_lr])} out of 20.")
        st.write(f"**SVM Prediction:** {reg_prediction_svm:.2f} out of 20.")
        
        # XGBoost Classification Prediction
        clf_prediction_xgb = classification_model_xgb.predict(input_data)[0]
        clf_probability_xgb = classification_model_xgb.predict_proba(input_data)[0][int(clf_prediction_xgb)]
        pass_fail_xgb = 'Pass' if clf_prediction_xgb == 1 else 'Fail'
        
        # Neural Network Classification Prediction
        probability_ann = classification_model_ann.predict(input_data)[0][0]
        clf_prediction_ann = int(probability_ann >= 0.5)
        clf_probability_ann = probability_ann
        pass_fail_ann = 'Pass' if clf_prediction_ann == 1 else 'Fail'
        
        # Random Forest Classification Prediction
        clf_prediction_rf = classification_model_rf.predict(input_data)[0]
        clf_probability_rf = classification_model_rf.predict_proba(input_data)[0][int(clf_prediction_rf)]
        pass_fail_rf = 'Pass' if clf_prediction_rf == 1 else 'Fail'
        
        # Decision Tree Classification Prediction
        clf_prediction_dt = classification_model_dt.predict(input_data)[0]
        clf_probability_dt = classification_model_dt.predict_proba(input_data)[0][int(clf_prediction_dt)]
        pass_fail_dt = 'Pass' if clf_prediction_dt == 1 else 'Fail'
        
        # Logistic Regression Prediction
        clf_prediction_logr = classification_model_logr.predict(input_data)[0]
        clf_probability_logr = classification_model_logr.predict_proba(input_data)[0][int(clf_prediction_logr)]
        pass_fail_logr = 'Pass' if clf_prediction_logr == 1 else 'Fail'
        
        # SVM Classification Prediction
        clf_prediction_svm = classification_model_svm.predict(input_data_svm_clf)[0]
        clf_probability_svm = classification_model_svm.predict_proba(input_data_svm_clf)[0][int(clf_prediction_svm)]
        pass_fail_svm = 'Pass' if clf_prediction_svm == 1 else 'Fail'
        
        st.subheader("Predicted Pass/Fail Status")
        st.write(f"**XGBoost Prediction:** {pass_fail_xgb} (Confidence: {clf_probability_xgb*100:.2f}%)")
        st.write(f"**Neural Network Prediction:** {pass_fail_ann} (Confidence: {clf_probability_ann*100:.2f}%)")
        st.write(f"**Random Forest Prediction:** {pass_fail_rf} (Confidence: {clf_probability_rf*100:.2f}%)")
        st.write(f"**Decision Tree Prediction:** {pass_fail_dt} (Confidence: {clf_probability_dt*100:.2f}%)")
        st.write(f"**Logistic Regression Prediction:** {pass_fail_logr} (Confidence: {clf_probability_logr*100:.2f}%)")
        st.write(f"**SVM Prediction:** {pass_fail_svm} (Confidence: {clf_probability_svm*100:.2f}%)")

if __name__ == '__main__':
    main()