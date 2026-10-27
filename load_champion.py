
import mlflow
import mlflow.sklearn
import pandas as pd

# ===== 1. Connexion à la base MLflow =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ===== 2. Charger le modèle champion =====
model_uri = "models:/churn-model@champion"

try:
    model = mlflow.sklearn.load_model(model_uri)
    print("Modèle chargé :", type(model).__name__)
except Exception as e:
    print("Erreur lors du chargement du modèle :", e)
    raise

# ===== 3. Afficher les variables attendues =====
if not hasattr(model, "feature_names_in_"):
    raise ValueError(
        "Le modèle ne possède pas feature_names_in_. "
        "Vérifie les variables utilisées pendant l'entraînement."
    )

features = list(model.feature_names_in_)
print("Variables attendues :", features)

# ===== 4. Créer un nouveau client =====
client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": 0,
    "abonnement_Standard": 1,
}])

# ===== 5. Vérifier les colonnes nécessaires =====
missing = [col for col in features if col not in client.columns]

if missing:
    raise ValueError(f"Variables manquantes : {missing}")

# Respecter l'ordre des variables attendu par le modèle
client = client[features]

# ===== 6. Effectuer la prédiction =====
prediction = model.predict(client)[0]

print(
    "Le client va-t-il partir ? ->",
    "OUI" if prediction == 1 else "NON"
)
