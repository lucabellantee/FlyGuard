import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import (norm, expon, uniform, gamma, beta, lognorm, chi2, weibull_min,
                         t, f, cauchy, laplace, rayleigh, pareto, gumbel_r, logistic,
                         erlang, powerlaw, nakagami, betaprime, kstest)

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

    def fit_best_distribution(self, data):
        """
        Trova la distribuzione migliore per i dati in input basandosi sulla statistica KS.
        Restituisce la distribuzione con i parametri stimati, la KS statistic e il p-value.
        """
        distributions = {
            "Normal (norm)": norm,
            "Exponential (expon)": expon,
            "Uniform (uniform)": uniform,
            "Gamma (gamma)": gamma,
            "Beta (beta)": beta,
            "Log-Normal (lognorm)": lognorm,
            "Chi-Squared (chi2)": chi2,
            "Weibull (weibull_min)": weibull_min,
            "Student’s t (t)": t,
            "F (f)": f,
            "Cauchy (cauchy)": cauchy,
            "Laplace (laplace)": laplace,
            "Rayleigh (rayleigh)": rayleigh,
            "Pareto (pareto)": pareto,
            "Gumbel (gumbel_r)": gumbel_r,
            "Logistic (logistic)": logistic,
            "Erlang (erlang)": erlang,
            "Power Law (powerlaw)": powerlaw,
            "Nakagami (nakagami)": nakagami,
            "Beta Prime (betaprime)": betaprime,
        }
        
        results = []

        # Calcola la migliore distribuzione con il test KS
        for dist_name, dist in distributions.items():
            try:
                # Stima dei parametri della distribuzione
                params = dist.fit(data)
                # Calcolo KS test
                ks_stat, ks_pvalue = kstest(data, dist.cdf, args=params)
                results.append((dist_name, ks_stat, ks_pvalue, params))
            except Exception:
                results.append((dist_name, np.inf, 0, None))

        # Ordina i risultati per KS statistic (più basso è migliore)
        results = sorted(results, key=lambda x: x[1])

        # Restituisce la miglior distribuzione con la migliore KS-statistica
        best_fit = results[0]

        # Estraggo i singoli parametri
        mu = best_fit[3][0]  # Media (location)
        s = best_fit[3][1]   # Scala (scale)
        
        return {
            "dist_name": best_fit[0],
            "ks_stat": best_fit[1],
            "ks_pvalue": best_fit[2],
            "loc": mu,
            "scale": s,
        }
