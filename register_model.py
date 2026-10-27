
# ===== 1. Importer les bibliothèques =====
import mlflow
from mlflow import MlflowClient

# ===== 2. Se connecter à la base MLflow =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# ===== 3. Retrouver l'expérience =====
experiment = client.get_experiment_by_name("churn-prediction")

if experiment is None:
    raise ValueError(
        "L'expérience 'churn-prediction' n'existe pas dans mlflow.db."
    )

# ===== 4. Rechercher les runs avec une métrique F1 =====
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.f1 DESC"],
    max_results=100
)

if runs.empty:
    raise ValueError(
        "Aucun run terminé trouvé pour cette expérience."
    )

if "metrics.f1" not in runs.columns or runs["metrics.f1"].isna().all():
    raise ValueError(
        "Aucun run ne contient la métrique 'f1'. Vérifie son nom."
    )

# Garder uniquement les runs ayant une valeur F1
runs = runs.dropna(subset=["metrics.f1"])

if runs.empty:
    raise ValueError("Aucun run ne possède une valeur F1 valide.")

# ===== 5. Sélectionner le meilleur run =====
best = runs.iloc[0]
run_id = best["run_id"]
f1_score = best["metrics.f1"]

run_name = best.get("tags.mlflow.runName", "Run sans nom")

print(f"Meilleur run : {run_name}")
print(f"ID : {run_id}")
print(f"Score F1 : {f1_score:.4f}")

# ===== 6. Enregistrer le modèle dans le Registry =====
model_uri = f"runs:/{run_id}/model"
model_name = "churn-model"

result = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print(f"Modèle enregistré : {model_name}")
print(f"Version : {result.version}")

# ===== 7. Définir l'alias champion =====
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=str(result.version)
)

print(
    f"Alias 'champion' -> version {result.version}"
)
