# TP1 MLOps – Prédiction du churn client

Modèle de classification qui prédit si un client va résilier son abonnement (churn), entraîné sur un jeu de données synthétique.

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `generate_data.py` | Génère 500 clients fictifs et les enregistre dans `churn_data.csv` |
| `churn_data.csv` | Jeu de données généré |
| `churn_model.ipynb` | Exploration des données, entraînement et évaluation du modèle |
| `model.pkl` | Modèle entraîné, sauvegardé avec `joblib` |

## Données

Chaque client est décrit par :

- `age` : 18 à 69 ans
- `salaire` : 1200 à 7999
- `abonnement` : Basique, Standard ou Premium
- `anciennete` : 0 à 14 ans
- `consommation` : 0 à 99
- `churn` (cible) : 1 si `consommation < 30` et `anciennete < 2`, sinon 0

La graine aléatoire est fixée (`np.random.seed(42)`), donc les données sont reproductibles.

## Modèle

`RandomForestClassifier` de scikit-learn, avec un découpage train/test 80/20 (`random_state=42`).

| Essai | Paramètres | Features | Accuracy |
|---|---|---|---|
| 1 | 100 arbres, `max_depth=5` | 5 | 0.99 |
| 2 | 50 arbres, `max_depth=3` | 5 | 0.99 |
| 3 | 100 arbres, `max_depth=5` | 4 (sans `salaire`) | 1.00 |

Chaque essai écrase `model.pkl` : le fichier actuel correspond à l'**essai 3** et attend 4 features (`age`, `abonnement`, `anciennete`, `consommation`).

> Le jeu de test ne contient que 5 clients en churn sur 100 : l'accuracy est donc peu représentative.

## Installation

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn joblib jupyter
```

## Utilisation

```powershell
python generate_data.py    # génère churn_data.csv
jupyter notebook churn_model.ipynb
```

Charger le modèle :

```python
import joblib
import pandas as pd

model = joblib.load("model.pkl")
client = pd.DataFrame([{"age": 30, "abonnement": 0, "anciennete": 1, "consommation": 10}])
print(model.predict(client))  # abonnement : 0 = Basique, 1 = Standard, 2 = Premium
```

## Branches

- `master` : version stable
- `develop` : intégration
- `feature/*` : développement des nouvelles fonctionnalités
