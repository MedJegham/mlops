
import requests

URL = "http://127.0.0.1:8000"

# ===== 1. Vérifier que l'API fonctionne =====
try:
    r = requests.get(URL + "/health", timeout=5)
    print("GET /health ->", r.status_code, r.json())
    r.raise_for_status()
except requests.RequestException as e:
    print("Erreur : impossible de contacter l'API.")
    print("Vérifie que le serveur FastAPI est démarré.", e)
    raise SystemExit(1)


# ===== 2. Tester le client 1 =====
client = {
    "age": 35,
    "salaire": 2500,
    "abonnement": "Standard",
    "anciennete": 1,
    "consommation": 10
}

r = requests.post(URL + "/predict", json=client, timeout=10)
print("POST /predict (client 1) ->", r.status_code, r.json())


# ===== 3. Tester le client 2 =====
client2 = {
    "age": 50,
    "salaire": 4000,
    "abonnement": "Premium",
    "anciennete": 10,
    "consommation": 80
}

r = requests.post(URL + "/predict", json=client2, timeout=10)
print("POST /predict (client 2) ->", r.status_code, r.json())


# ===== 4. Tester une erreur de validation =====
mauvais = dict(client, age="abc")

r = requests.post(URL + "/predict", json=mauvais, timeout=10)

print("POST /predict (age='abc') ->", r.status_code)

try:
    erreur = r.json()

    if r.status_code == 422:
        print("Erreur de validation :", erreur["detail"][0]["msg"])
    else:
        print("Réponse :", erreur)

except (ValueError, KeyError, IndexError):
    print("Réponse inattendue :", r.text)
