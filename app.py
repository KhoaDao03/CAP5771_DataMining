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
    school = st.sidebar.selectbox('School', options=['GP', 'MS'])

    # 2. sex
    sex = st.sidebar.selectbox('Sex', options=['F', 'M'])

    # 3. age
    age = st.sidebar.slider('Age', 15, 22, 17)

    # 4. address
    address = st.sidebar.selectbox('Home Address', options=['U', 'R'])

    # 5. famsize
    famsize = st.sidebar.selectbox('Family Size', options=['LE3', 'GT3'])

    # 6. Pstatus
    Pstatus = st.sidebar.selectbox("Parent's Cohabitation Status", options=['T', 'A'])

    # 7. Medu
    Medu = st.sidebar.selectbox("Mother's Education", options=[0, 1, 2, 3, 4])

    # 8. Fedu
    Fedu = st.sidebar.selectbox("Father's Education", options=[0, 1, 2, 3, 4])

    # 9. Mjob
    Mjob = st.sidebar.selectbox("Mother's Job", options=['at_home', 'health', 'other', 'services', 'teacher'])

    # 10. Fjob
    Fjob = st.sidebar.selectbox("Father's Job", options=['at_home', 'health', 'other', 'services', 'teacher'])

    # 11. reason
    reason = st.sidebar.selectbox("Reason to Choose School", options=['home', 'reputation', 'course', 'other'])

    # 12. guardian
    guardian = st.sidebar.selectbox("Guardian", options=['mother', 'father', 'other'])

    # 13. traveltime
    traveltime = st.sidebar.selectbox("Travel Time to School", options=[1, 2, 3, 4])

    # 14. studytime
    studytime = st.sidebar.selectbox("Weekly Study Time", options=[1, 2, 3, 4])

    # 15. failures
    failures = st.sidebar.slider('Number of Past Class Failures', 0, 4, 0)

    # 16. schoolsup
    schoolsup = st.sidebar.selectbox('Extra Educational Support', options=['yes', 'no'])

    # 17. famsup
    famsup = st.sidebar.selectbox('Family Educational Support', options=['yes', 'no'])

    # 18. paid
    paid = st.sidebar.selectbox('Extra Paid Classes', options=['yes', 'no'])

    # 19. activities
    activities = st.sidebar.selectbox('Extra-curricular Activities', options=['yes', 'no'])

    # 20. nursery
    nursery = st.sidebar.selectbox('Attended Nursery School', options=['yes', 'no'])

    # 21. higher
    higher = st.sidebar.selectbox('Wants Higher Education', options=['yes', 'no'])

    # 22. internet
    internet = st.sidebar.selectbox('Internet Access at Home', options=['yes', 'no'])

    # 23. romantic
    romantic = st.sidebar.selectbox('In a Romantic Relationship', options=['yes', 'no'])

    # 24. famrel
    famrel = st.sidebar.slider('Family Relationship Quality', 1, 5, 4)

    # 25. freetime
    freetime = st.sidebar.slider('Free Time After School', 1, 5, 3)

    # 26. goout
    goout = st.sidebar.slider('Going Out with Friends', 1, 5, 3)

    # 27. Dalc
    Dalc = st.sidebar.slider('Workday Alcohol Consumption', 1, 5, 1)

    # 28. Walc
    Walc = st.sidebar.slider('Weekend Alcohol Consumption', 1, 5, 1)

    # 29. health
    health = st.sidebar.slider('Current Health Status', 1, 5, 5)

    # 30. absences
    absences = st.sidebar.number_input('Number of School Absences', min_value=0, max_value=93, value=0)

    # 31. G1 - first period grade
    G1 = st.sidebar.slider('First Period Grade (G1)', 0, 20, 10)

    # 32. G2 - second period grade
    G2 = st.sidebar.slider('Second Period Grade (G2)', 0, 20, 10)

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