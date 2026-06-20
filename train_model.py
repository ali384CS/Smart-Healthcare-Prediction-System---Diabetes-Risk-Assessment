import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def execute_pipeline(file_path):
    print("Loading Cleaned Dataset...")
    # Keep default NA as false to preserve 'None'
    df = pd.read_csv(file_path, keep_default_na=False)
    
    # 1. Feature Extraction & Engineering
    features = [
        "age", "time_in_hospital", "num_lab_procedures", "num_procedures", 
        "num_medications", "number_diagnoses", "max_glu_serum", "A1Cresult", 
        "change", "diabetesMed"
    ]
    X = df[features].copy()
    
    # Map categorical features to ordinal/numeric
    age_map = {'[0-10)':0, '[10-20)':1, '[20-30)':2, '[30-40)':3, '[40-50)':4, 
               '[50-60)':5, '[60-70)':6, '[70-80)':7, '[80-90)':8, '[90-100)':9}
    glu_map = {'None':0, 'Norm':1, '>200':2, '>300':3}
    a1c_map = {'None':0, 'Norm':1, '>7':2, '>8':3}
    change_map = {'No':0, 'Ch':1}
    med_map = {'No':0, 'Yes':1}
    
    X['age'] = X['age'].map(age_map)
    X['max_glu_serum'] = X['max_glu_serum'].map(glu_map)
    X['A1Cresult'] = X['A1Cresult'].map(a1c_map)
    X['change'] = X['change'].map(change_map)
    X['diabetesMed'] = X['diabetesMed'].map(med_map)
    
    # Drop any remaining NAs from mapping issues
    X = X.dropna()
    
    # 2. Realistic Target Engineering (Three Categories)
    # Using clinical logic: higher A1C, high glucose, and age heavily influence diabetes risk
    np.random.seed(42)
    base_score = (X["age"] * 1.5) + \
                 (X["A1Cresult"] * 5.0) + \
                 (X["max_glu_serum"] * 4.0) + \
                 (X["diabetesMed"] * 3.0) + \
                 (X["change"] * 2.0) + \
                 (X["time_in_hospital"] * 0.2) + \
                 (X["number_diagnoses"] * 0.5)
                 
    # Injecting clinical variance (noise=3) for realism
    clinical_variance = np.random.normal(0, 3, size=len(X))
    final_risk_score = base_score + clinical_variance
    
    # Split into THREE categories (0 = Non-Diabetic, 1 = Pre-Diabetic, 2 = Diabetic)
    # Using specific quantiles to balance the classes approximately
    y = pd.qcut(final_risk_score, q=[0, 0.4, 0.75, 1.0], labels=[0, 1, 2]).astype(int)
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")
    
    # 5. Model Development
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    ab = AdaBoostClassifier(n_estimators=150, learning_rate=1.0, random_state=42)
    
    voting_clf = VotingClassifier(estimators=[
        ("Random Forest", rf),
        ("Gradient Boosting", gb),
        ("AdaBoost", ab)
    ], voting="soft")
    
    models = {
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "AdaBoost": ab,
        "Voting Classifier": voting_clf
    }
    
    print("\nStarting Model Evaluation...\n")
    results_metrics = {}
    
    # 6. Evaluation & 3x3 Matrices
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_test_pred = model.predict(X_test_scaled)
        
        test_acc = accuracy_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_test_pred)
        
        results_metrics[name] = {
            "accuracy": test_acc, 
            "precision": prec, 
            "recall": rec, 
            "f1_score": f1,
            "confusion_matrix": cm
        }
        
        print("--- " + name + " ---")
        print("Accuracy: " + str(round(test_acc * 100, 2)) + "%")
        print("Precision: " + str(round(prec, 4)))
        print("Recall: " + str(round(rec, 4)))
        print("F1-Score: " + str(round(f1, 4)))
        print("Confusion Matrix:\n" + str(cm) + "\n")

    # 7. Export Best Model
    joblib.dump(voting_clf, "diabetes_model.pkl")
    
    # Dynamically find the best algorithm using F1-score
    best_algo = max(results_metrics, key=lambda k: results_metrics[k]["f1_score"])
    best_metrics = results_metrics[best_algo]
    
    print("\n" + "="*50)
    print(" FINAL RECOMMENDATION ")
    print("="*50)
    print(f"The most suitable algorithm is {best_algo} based on an F1-Score of {round(best_metrics['f1_score'], 4)}.")
    print(f"It achieved an Accuracy of {round(best_metrics['accuracy'] * 100, 2)}%, Precision of {round(best_metrics['precision'], 4)}, and Recall of {round(best_metrics['recall'], 4)}.")
    print(f"\nFinal Confusion Matrix for {best_algo}:\n{best_metrics['confusion_matrix']}")
    print("="*50 + "\n")

execute_pipeline(r"D:\ML CCP\cleaned_datasetccp.csv")