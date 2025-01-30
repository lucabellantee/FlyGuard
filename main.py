import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from model_inference import ModelInference

if __name__ == "__main__":
    path = 'Dataset/Train/X_train_with_trq_margin.csv'
    regressor_path = 'Model/RandomForestRegressor.pkl'
    classifier_path = 'Model/random_forest_modelClassifier.pkl'
    
    # Carica il dataset
    dataset = pd.read_csv(path)

    # Divido in training e test set
    X_train, X_test = train_test_split(dataset, test_size=0.2, random_state=42)

    # Label per regressione e classificazione
    y_true_regression = X_test['trq_target']
    y_true_classification = X_test['faulty']

    # Seleziona le colonne di input
    X_rg = X_test[['oat', 'mgt', 'pa', 'ias', 'np', 'ng']]
    X_cl = X_test[['trq_measured', 'oat', 'mgt', 'pa', 'ias', 'np', 'ng', 'trq_target', 'trq_margin']]
    
    # Inizializza la classe
    inference_model = ModelInference(regressor_path, classifier_path)

    # Effettua previsioni
    y_pred_regression = inference_model.predict_regression(X_rg)
    X_cl['trq_target'] = y_pred_regression
    y_pred_classification = inference_model.predict_classification(X_cl)
    X_cl['faulty'] = y_pred_classification

    # Salva il dataset con le previsioni
    inference_model.save_results(X_cl, 'Dataset/Results/X_cl_results.csv')

    # Stampa metriche
    inference_model.evaluate_regression(y_true_regression, y_pred_regression)
    inference_model.evaluate_classification(y_true_classification, y_pred_classification)

    # Calcola la distribuzione logistica
    best_distribution_result = inference_model.fit_best_distribution(X_cl['trq_target'])
    
    print("\n📊 **Distribuzione Logistica**")
    print(f"Nome Distribuzione: {best_distribution_result['dist_name']}")
    print(f"Media (mu - loc): {best_distribution_result['loc']:.4f}")
    print(f"Scala (scale): {best_distribution_result['scale']:.4f}")
    print(f"KS Statistica: {best_distribution_result['ks_stat']:.4f}")
    print(f"p-value: {best_distribution_result['ks_pvalue']:.4f}")
