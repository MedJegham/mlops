
# ===== 1. Importer les bibliothèques =====
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ===== 2. Créer l'application =====
app = FastAPI(title="API Churn")

# ===== 3. Charger le modèle champion une seule fois =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)

# Récupérer les colonnes attendues par le modèle
features = list(model.feature_names_in_)

# ===== 4. Décrire les données reçues =====
class Client(BaseModel):
    age: int = Field(ge=0)
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float = Field(ge=0)
    consommation: float = Field(ge=0)


# ===== 5. Route de vérification =====
@app.get("/health")
def health():
    return {"status": "ok"}


# ===== 6. Route de prédiction =====
@app.post("/predict")
def predict(client: Client):
    # a) Convertir les données en DataFrame
    ligne = pd.DataFrame([client.model_dump()])

    # b) Transformer abonnement en colonnes indicatrices
    ligne = pd.get_dummies(
        ligne,
        columns=["abonnement"],
        dtype=int
    )

    # c) Aligner les colonnes sur celles du modèle
    ligne = ligne.reindex(
        columns=features,
        fill_value=0
    )

    # d) Effectuer la prédiction
    try:
        resultat = int(model.predict(ligne)[0])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prédiction : {str(e)}"
        )

    # e) Retourner le résultat au format JSON
    return {
        "churn": resultat,
        "message": (
            "Le client va partir"
            if resultat == 1
            else "Le client reste"
        )
    }
