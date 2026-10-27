
import mlflow
from mlflow import MlflowClient

# ===== 1. Connexion à MLflow =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# ===== 2. Retrouver le run "essai-2" =====
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="tags.mlflow.runName = 'essai-2'"
)

if runs.empty:
    raise ValueError(
        "Aucun run nommé 'essai-2' trouvé dans l'expérience."
    )

run_id = runs.iloc[0]["run_id"]

print("Run trouvé :", run_id)

# ===== 3. Enregistrer une nouvelle version =====
model_uri = f"runs:/{run_id}/model"
model_name = "churn-model"

v = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print("Nouvelle version créée :", v.version)

# ===== 4. Mémoriser la version actuelle de champion =====
actuel = client.get_model_version_by_alias(
    model_name,
    "champion"
)

ancienne_version = actuel.version

print("Champion actuel : version", ancienne_version)

# ===== 5. Déplacer temporairement l'alias =====
client.set_registered_model_alias(
    model_name,
    "champion",
    str(v.version)
)

nouveau_champion = client.get_model_version_by_alias(
    model_name,
    "champion"
)

print(
    "Après déplacement : champion -> version",
    nouveau_champion.version
)

# ===== 6. Restaurer l'alias initial =====
client.set_registered_model_alias(
    model_name,
    "champion",
    str(ancienne_version)
)

champion_final = client.get_model_version_by_alias(
    model_name,
    "champion"
)

print(
    "Après retour : champion -> version",
    champion_final.version
)
