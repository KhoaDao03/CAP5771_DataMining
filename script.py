import subprocess

def run_scripts():
    
    """
    Runs a command using subprocess. Attempts to run with 'python3' first,
    and falls back to 'python' if 'python3' is not available.
    """
    command = "python3"
    try:
        # Pass the command as a list
        subprocess.run([command, "--version"], check=True)
    except subprocess.CalledProcessError:
        print(f"{command} not found. Retrying with python.")
        command = "python"
        try:
            subprocess.run([command, "--version"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"{command} also not found.")
            raise e
    except subprocess.CalledProcessError as e:
        print(f"Error while running script: {e}")
        raise e    
    """
    Executes all project scripts sequentially.
    """
    try:
        # Data Preprocessing
        print("Running Data Preprocessing")
        subprocess.run([command, "src/data_preprocessing.py"])

        # Train the models
        print("Linear Regression")
        subprocess.run([command, "src/linear_regression.py"])

        print("Logistic Regression")
        subprocess.run([command, "src/logistic_regression.py"])

        print("Decision Tree")
        subprocess.run([command, "src/decision_tree.py"])

        print("Random Forest")
        subprocess.run([command, "src/random_forest.py"])

        print("Support Vector Machines (SVM)")
        subprocess.run([command, "src/svm.py"])

        print("Artificial Neural Networks (ANN)")
        subprocess.run([command, "src/neural_network.py"])

        print("XGBoost")
        subprocess.run([command, "src/xgboost_model.py"])

        print("K-Means Clustering")
        subprocess.run([command, "src/k_means_clustering.py"])

        print("All scripts executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Script execution halted due to an error: {e}")
    except FileNotFoundError as e:
        print(f"Script not found: {e}")

    try:
        # Launching Web Application
        print("Launching the Web Application")
        subprocess.run(["streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("Process interrupted by user.")

if __name__ == "__main__":
    run_scripts()
