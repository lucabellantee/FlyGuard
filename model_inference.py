import json
import numpy as np
import scipy.stats as stats
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
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
        prediction = self.rf_classifier.predict(X)
        confidence_score = self.rf_classifier.predict_proba(X)

        # Creiamo una lista per i punteggi di confidenza corrispondenti
        confidence_scores = []

        # Iteriamo su ciascuna previsione
        for i, pred in enumerate(prediction):
            if pred == 0:
                # Se la previsione è 0, prendiamo la probabilità associata alla classe 0
                confidence_scores.append(confidence_score[i, 0])
            else:
                # Se la previsione è 1, prendiamo la probabilità associata alla classe 1
                confidence_scores.append(confidence_score[i, 1])
        
        return prediction, confidence_scores

    def evaluate_regression(self, y_true, y_pred):
        """
        Calcola e stampa le metriche della regressione.
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print("\n📊 **Metriche Regressione**")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
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
        Restituisce la distribuzione con i parametri stimati, la KS statistic, il p-value ed i
        parametri loc e scale.
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
        
    def fit_best_distribution_per_sample(self, y_pred_regression):
        """
        Per ogni valore previsto di trq_target, trova la miglior distribuzione tra quelle candidate.
        """
        results = []
        distributions = {
            "norm": stats.norm,
            "expon": stats.expon,
            "uniform": stats.uniform,
            "lognorm": stats.lognorm,
            "chi2": stats.chi2,
            "cauchy": stats.cauchy,
            "laplace": stats.laplace,
            "logistic": stats.logistic
        }

        for i in tqdm(range(len(y_pred_regression)), desc="Calcolo miglior distribuzione per ogni campione"):
            single_pred_samples = y_pred_regression[i]
            best_fit = None
            best_ll = -np.inf
            best_params = None
            best_pdf_type = None

            for dist_name, dist in distributions.items():
                try:
                    params = dist.fit([single_pred_samples])
                    ll = np.sum(dist.logpdf([single_pred_samples], *params))
                    if ll > best_ll:
                        best_ll = ll
                        best_fit = dist
                        best_pdf_type = dist_name
                        best_params = params
                except Exception:
                    continue

            if best_params:
                results.append({
                    "trq_target_pred": single_pred_samples,
                    "pdf_type": best_pdf_type,
                    "loc": best_params[1] if len(best_params) > 1 else 0,
                    "scale": best_params[2] if len(best_params) > 2 else 1
                })
            else:
                results.append({
                    "trq_target_pred": single_pred_samples,
                    "pdf_type": best_pdf_type,
                    "loc": 0,
                    "scale": 0
                })

        return results
    
    
    def build_json(self, data):
        json_data = {}
        
        for i, row in data.iterrows():
            # Estrae i valori delle colonne rilevanti
            class_value = row['faulty']
            class_conf = row['confidence']
            pdf_type = row['best_pdf']
            loc = row['loc']
            scale = row['scale']
            
            # Crea la struttura del JSON per questa riga
            json_data[str(i)] = {
                "class": int(class_value),
                "class_conf": float(class_conf),
                "pdf_type": pdf_type,
                "pdf_args": {
                    "loc": float(loc),
                    "scale": float(scale)
                }
            }
        
        # Salva tutto in un unico file JSON
        with open('Results/output.json', 'w') as outfile:
            json.dump(json_data, outfile, indent=4)
        
        print("JSON salvato correttamente")
