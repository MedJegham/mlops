# ===== 1. Importer les bibliothèques =====
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ===== 2. Charger les données =====
df = pd.read_csv("churn_data.csv")
y = df["churn"]  # 1 = le client part, 0 = le client reste


# ===== 3. Fonction : préparer les données =====
def preparer(colonnes):
    # Transformer les colonnes catégorielles en colonnes numériques
    X = pd.get_dummies(df[colonnes], drop_first=True)

    # Diviser les données : 80 % apprentissage, 20 % test
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


# ===== 4. Fonction : lancer un essai et l'enregistrer dans MLflow =====
def lancer_essai(nom, n_estimators, max_depth, colonnes):
    X_train, X_test, y_train, y_test = preparer(colonnes)

    with mlflow.start_run(run_name=nom):

        # Créer le modèle
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        # Entraîner le modèle
        model.fit(X_train, y_train)

        # Effectuer les prédictions
        pred = model.predict(X_test)

        # --- Paramètres ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", ",".join(colonnes))

        # --- Métriques ---
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(
            y_test, pred, zero_division=0
        )
        recall = recall_score(
            y_test, pred, zero_division=0
        )
        f1 = f1_score(
            y_test, pred, zero_division=0
        )

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # --- Enregistrer le modèle ---
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            # Type interne des arbres sklearn : sûr car le modèle est créé ici
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        # Afficher les résultats
        print(
            nom,
            "-> accuracy :", round(accuracy, 3),
            "| f1 :", round(f1, 3)
        )


# ===== 5. Programme principal =====
mlflow.set_experiment("churn-prediction")

TOUT = [
    "age",
    "salaire",
    "abonnement",
    "anciennete",
    "consommation"
]

# Essai 1 : 100 arbres, profondeur 5, toutes les colonnes
lancer_essai("essai-1", 100, 5, TOUT)

# Essai 2 : 50 arbres, profondeur 3
lancer_essai("essai-2", 50, 3, TOUT)

# Essai 3 : sans la colonne salaire
lancer_essai(
    "essai-3",
    100,
    5,
    ["age", "abonnement", "anciennete", "consommation"]
)