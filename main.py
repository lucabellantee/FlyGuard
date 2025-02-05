import pandas as pd
import numpy as np
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
    X_cl['trq_margin'] = ((X_cl['trq_measured'] - X_cl['trq_target']) / X_cl['trq_target'])*100
    y_pred_classification, y_pred_confidence = inference_model.predict_classification(X_cl)
    X_cl['faulty'] = y_pred_classification
    X_cl['confidence'] = y_pred_confidence

    trq_margin = np.array(X_cl['trq_margin'])

    # Stampa metriche
    inference_model.evaluate_regression(y_true_regression, y_pred_regression)
    inference_model.evaluate_classification(y_true_classification, y_pred_classification)

    # Calcola la miglior distribuzione per l'intera colonna 'trq_target'
    best_distribution_results = inference_model.fit_best_distribution_per_sample(trq_margin)

    if 'best_pdf' not in X_cl.columns:
        X_cl['best_pdf'] = None
    if 'loc' not in X_cl.columns:
        X_cl['loc'] = None
    if 'scale' not in X_cl.columns:
        X_cl['scale'] = None

    # Verifica che la lunghezza di best_distribution_results non sia inferiore a 100
    n_rows = min(len(X_cl), len(best_distribution_results))

    # Aggiorna solo i primi n_rows di X_cl
    X_cl.iloc[:n_rows, X_cl.columns.get_loc('best_pdf')] = [res["pdf_type"] if res["pdf_type"] else "None" for res in best_distribution_results[:n_rows]]
    X_cl.iloc[:n_rows, X_cl.columns.get_loc('loc')] = [res["loc"] if res["pdf_type"] else None for res in best_distribution_results[:n_rows]]
    X_cl.iloc[:n_rows, X_cl.columns.get_loc('scale')] = [res["scale"] if res["pdf_type"] else None for res in best_distribution_results[:n_rows]]
    
    # Salva il dataset con le previsioni
    inference_model.save_results(X_cl, 'Results/X_cl_results.csv')

    # Crea il file json di risposta
    inference_model.build_json(X_cl)
