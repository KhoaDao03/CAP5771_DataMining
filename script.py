import subprocess

def run_scripts():
    try:
        # Run the scripts for the project

        # Data Preprocessing
        print("Running Data Preprocessing")
        subprocess.run(["python3", "src/data_preprocessing.py"], check=True)
        
        # Train the models
        print("Linear Regression")
        subprocess.run(["python3", "src/linear_regression.py"], check=True)
        
        print("Logistic Regression")
        subprocess.run(["python3", "src/logistic_regression.py"], check=True)

        print("Decision Tree")
        subprocess.run(["python3", "src/decision_tree.py"], check=True)

        print("Random Forest")
        subprocess.run(["python3", "src/random_forest.py"], check=True)

        print("Support Vector Machines (SVM):")
        subprocess.run(["python3", "src/svm.py"], check=True)

        print("Artificial Neural Networks (ANN)")
        subprocess.run(["python3", "src/neural_network.py"], check=True)

        print("XGBoost")
        subprocess.run(["python3", "src/xgboost_model.py"], check=True)

        print("K-Means Clustering")
        subprocess.run(["python3", "src/k_means_clustering.py"], check=True)

        

        print("All scripts executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error while running a script: {e}")
    except FileNotFoundError as e:
        print(f"Script not found or Python3 is not properly configured: {e}")

    try:    
        # Launching Web Application 
        print("Launching the Web Application")
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("Process interrupted by user.")

if __name__ == "__main__":
    run_scripts()
