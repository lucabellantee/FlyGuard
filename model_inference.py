import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import logistic, kstest

class ModelInference:
    def __init__(self, regressor_path: str, classifier_path: str):
        """
        Inizializza la classe caricando i modelli.
        """
        self.rf_regressor = joblib.load(regressor_path)
        self.rf_classifier = joblib.load(classifier_path)
    
    def predict_regression(self, X):
        """
        Effettua una previsione con il modello di regressione.
        """
        return self.rf_regressor.predict(X)
    
    def predict_classification(self, X):
        """
        Effettua una previsione con il modello di classificazione.
        """
        return self.rf_classifier.predict(X)

    def evaluate_regression(self, y_true, y_pred):
        """
        Calcola e stampa le metriche della regressione.
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)

        print("\n📊 **Metriche Regressione**")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")

    def evaluate_classification(self, y_true, y_pred):
        """
        Calcola e stampa le metriche della classificazione.
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        conf_matrix = confusion_matrix(y_true, y_pred)

        print("\n📊 **Metriche Classificazione**")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("\nMatrice di Confusione:")
        print(conf_matrix)

    def save_results(self, X_cl, filename):
        """
        Salva il dataset con le previsioni.
        """
        X_cl.to_csv(filename, index=False)
        print(f"\n✅ Dataset salvato in: {filename}")

    def fit_logistic_distribution(self, data):
        """
        Calcola la distribuzione logistica e ritorna i parametri stimati
        e il test KS.
        """
        try:
            # Stima i parametri della distribuzione logistica
            params = logistic.fit(data)

            # Estraggo i singoli parametri
            mu = params[0]  # Media (location)
            s = params[1]   # Scala (scale)
            
            # Calcola il test KS per verificare l'adattamento
            ks_stat, ks_pvalue = kstest(data, logistic.cdf, args=params)
            
            return {
                "dist_name": "Logistic (logistic)",
                "loc": mu,
                "scale": s,
                "ks_stat": ks_stat,
                "ks_pvalue": ks_pvalue
            }
        except Exception as e:
            print(f"Errore durante l'adattamento della distribuzione logistica: {e}")
            return None
