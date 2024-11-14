import streamlit as st
import joblib
import pandas as pd
import os

def load_models(project_dir):
    """
    Load the trained models.
    """
    models_dir = os.path.join(project_dir, 'outputs', 'models')
    regression_model = joblib.load(os.path.join(models_dir, 'xgboost_regression_model.pkl'))
    classification_model = joblib.load(os.path.join(models_dir, 'xgboost_classification_model.pkl'))
    return regression_model, classification_model

def main():
    st.title("Student Performance Prediction App")
    st.write("Input the student's features to get the predicted final grade and pass/fail status.")

    # Determine the absolute path to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '.'))

    # Load models
    regression_model, classification_model = load_models(project_dir)

    st.sidebar.header('Input Features')

    # Binary Variables Mapping
    binary_options = {'yes': 1, 'no': 0, 'F': 0, 'M': 1, 'U': 0, 'R': 1, 'LE3': 0, 'GT3': 1, 'T': 1, 'A': 0}

    # Feature Inputs

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

    # Encode binary variables
    binary_vars = ['sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup',
                   'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    for var in binary_vars:
        input_data[var] = input_data[var].map(binary_options)

    # Map categorical variables using label encoding
    school_mapping = {'GP': 0, 'MS': 1}
    Mjob_mapping = {'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4}
    Fjob_mapping = {'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4}
    reason_mapping = {'home': 0, 'reputation': 1, 'course': 2, 'other': 3}
    guardian_mapping = {'mother': 0, 'father': 1, 'other': 2}

    input_data['school'] = input_data['school'].map(school_mapping)
    input_data['Mjob'] = input_data['Mjob'].map(Mjob_mapping)
    input_data['Fjob'] = input_data['Fjob'].map(Fjob_mapping)
    input_data['reason'] = input_data['reason'].map(reason_mapping)
    input_data['guardian'] = input_data['guardian'].map(guardian_mapping)

    # Define expected columns
    expected_columns = [
        'school',
        'sex',
        'age',
        'address',
        'famsize',
        'Pstatus',
        'Medu',
        'Fedu',
        'Mjob',
        'Fjob',
        'reason',
        'guardian',
        'traveltime',
        'studytime',
        'failures',
        'schoolsup',
        'famsup',
        'paid',
        'activities',
        'nursery',
        'higher',
        'internet',
        'romantic',
        'famrel',
        'freetime',
        'goout',
        'Dalc',
        'Walc',
        'health',
        'absences',
        'G1',
        'G2'
    ]

    # Ensure all expected columns are present
    missing_cols = set(expected_columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0

    # Reorder columns to match expected_columns
    input_data = input_data[expected_columns]

    # Prediction
    if st.button('Predict'):
        # Regression Prediction
        reg_prediction = regression_model.predict(input_data)[0]
        st.subheader("Predicted Final Grade (G3)")
        st.write(f"The predicted final grade is: **{reg_prediction:.2f}** out of 20.")

        # Classification Prediction
        clf_prediction = classification_model.predict(input_data)[0]
        clf_probability = classification_model.predict_proba(input_data)[0][int(clf_prediction)]
        pass_fail = 'Pass' if clf_prediction == 1 else 'Fail'
        st.subheader("Predicted Pass/Fail Status")
        st.write(f"The student is predicted to: **{pass_fail}**")
        st.write(f"Prediction Confidence: **{clf_probability*100:.2f}%**")

if __name__ == '__main__':
    main()
