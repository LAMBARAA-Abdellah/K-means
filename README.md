# 🧠 Atelier 1 – K-Means (Implémentation From Scratch)

## 🎯 Objectif du TP

L’objectif de ce travail pratique est d’implémenter l’algorithme de **K-Means** depuis zéro, en utilisant uniquement **NumPy** (sans scikit-learn), puis d’ajouter des fonctions pour évaluer la qualité du clustering à chaque itération à l’aide de deux mesures :
- **Inertie intra-classes** (Within-Cluster Sum of Squares – WCSS)
- **Inertie inter-classes** (Between-Cluster Sum of Squares – BCSS)

---

## 🧩 Travail à réaliser

### 1️⃣ Génération des données
- Générer des données aléatoires en 2 dimensions à l’aide de `make_blobs`.
- Visualiser les points à l’aide de `matplotlib` ou `seaborn`.

```python
from sklearn.datasets import make_blobs
X_train, true_labels = make_blobs(n_samples=100, centers=4, random_state=42)
```

---

### 2️⃣ Implémentation de la distance euclidienne

```python
def euclid(centre, data):
    return np.sqrt(np.sum((centre - data)**2, axis=1))
```

---

### 3️⃣ Implémentation de K-Means

Créer une classe `Kmeans` avec :
- `__init__` pour initialiser le nombre de clusters et le nombre d’itérations.
- `fit(X_train)` pour :
  - Initialiser les centres aléatoirement.
  - Répéter l’affectation et la mise à jour des centres.
  - Calculer les inerties intra et inter à chaque itération.

---

### 4️⃣ Fonctions de calcul des inerties

```python
def inertia_intra(X, labels, centres):
    X = np.asarray(X)
    centres = np.asarray(centres)
    s = 0.0
    for k in range(len(centres)):
        members = X[labels == k]
        if members.size == 0:
            continue
        diffs = members - centres[k]
        s += float(np.sum(diffs * diffs))
    return s

def inertia_inter(X, labels, centres):
    X = np.asarray(X)
    centres = np.asarray(centres)
    mu = np.mean(X, axis=0)
    total = 0.0
    for k, c in enumerate(centres):
        n_k = np.sum(labels == k)
        if n_k == 0:
            continue
        diff = c - mu
        total += n_k * float(np.dot(diff, diff))
    return total
```

---

### 5️⃣ Intégration dans la classe K-Means

Dans la méthode `fit`, afficher à **chaque itération** :

```python
intra = inertia_intra(X, labels, new_centres)
inter = inertia_inter(X, labels, new_centres)
print(f"Iteration {t:02d} — Intra: {intra:.4f} | Inter: {inter:.4f}")
```

---

### 6️⃣ Visualisation du résultat final

```python
plt.scatter(X_train[:, 0], X_train[:, 1], c=labels, cmap='viridis')
plt.scatter(centres[:, 0], centres[:, 1], marker='+', s=200, c='red')
plt.title("Résultat final du K-Means")
plt.show()
```

---

## 🧮 Résultat attendu

- À chaque itération, la console doit afficher :
  ```
  Iteration 01 — Intra: 4563.12 | Inter: 37921.45
  Iteration 02 — Intra: 2389.54 | Inter: 38452.90
  ...
  ```
- Une convergence visible des centres vers les clusters.
- Un graphique avec les points colorés et les centres marqués d’un `+`.

---

## ⚙️ Technologies utilisées
- **Python 3.x**
- **NumPy**
- **Matplotlib / Seaborn**
- **Scikit-learn** (uniquement pour la génération des données `make_blobs`)

---

## 📁 Structure recommandée

```
Atelier1_Kmeans/
│
├── Atelier1_Kmeans_donnees_aléatoires.ipynb   # Notebook principal
├── README.md                                  # Ce fichier
└── data/                                      # (optionnel)
```
