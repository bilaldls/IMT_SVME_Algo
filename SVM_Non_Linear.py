import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 0) Charger les données
# =========================
# Fichier à adapter si besoin
# On définit juste le chemin du CSV ici :

#csv_path = "Echec4_T.csv"  
#csv_path = "Echec4_L.csv"  
csv_path = "HeartAche_1.csv" 


def load_csv_auto(path):
    """
    Charge un CSV en essayant d'inférer le séparateur.
    - Essaye d'abord le séparateur par défaut de pandas.
    - Si on obtient une seule colonne avec des ';' dedans, on réessaie avec sep=';'.
    """
    df_tmp = pd.read_csv(path)
    if df_tmp.shape[1] == 1:
        first_val = df_tmp.iloc[0, 0]
        if isinstance(first_val, str) and ';' in first_val:
            df_tmp = pd.read_csv(path, sep=';')
    return df_tmp

df = load_csv_auto(csv_path)
print(df.head())

# Détection automatique de la colonne d'étiquette :
# 1) Si 'HeartDisease' existe, on la prend (cas HeartAche_1.csv)
# 2) Sinon, si 'u' existe, on la prend (cas données TP linéaires)
# 3) Sinon, on suppose que la dernière colonne est l'étiquette
if "HeartDisease" in df.columns:
    label_col = "HeartDisease"
elif "u" in df.columns:
    label_col = "u"
else:
    label_col = df.columns[-1]

# Séparer X (toutes les features) et y (étiquette)
feature_cols = [c for c in df.columns if c != label_col]
df_X = df[feature_cols]
u_raw = df[label_col].to_numpy()


#dummies 
from sklearn.model_selection import train_test_split

# Encodage des variables catégorielles en variables numériques (one-hot)
# Cela gère automatiquement Sex, ChestPainType, RestingECG, etc.
df_X_encoded = pd.get_dummies(df_X, drop_first=False)

# Matrice numérique de toutes les features
X_all = df_X_encoded.to_numpy(dtype=float)

# Split train / test sur les données encodées
X_train, X_test, y_train, y_test = train_test_split(
    X_all, u_raw, test_size=0.2, random_state=0, shuffle=True, stratify=u_raw
)


'''#Sans dummies en supprimant les colonnes catégorielles 

from sklearn.model_selection import train_test_split

# Suppression des variables catégorielles : on ne garde que les colonnes numériques
# (pour HeartAche_1.csv, ça garde typiquement :
#  Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak)
df_X_numeric = df_X.select_dtypes(include=[np.number])

# Matrice numérique de toutes les features
X_all = df_X_numeric.to_numpy(dtype=float)

# Split train / test sur les données numériques
X_train, X_test, y_train, y_test = train_test_split(
    X_all, u_raw, test_size=0.2, random_state=0, shuffle=True, stratify=u_raw
)'''

from sklearn.preprocessing import StandardScaler

# Standardisation des features (moyenne 0, variance 1) sur l'ensemble d'entraînement ===> Z Score !!!!!!

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Normalisation des étiquettes en {-1, 1} sur l'ensemble d'entraînement
labels = np.unique(y_train)
if set(labels) == {0, 1}:
    y = np.where(y_train == 0, -1.0, 1.0)
else:
    y = y_train.astype(float)

# On travaille ensuite avec X = X_train pour apprendre le SVM
X = X_train
n = X.shape[0]

'''# Visualisation des données (projection PCA en 2D pour inspection rapide)
from sklearn.decomposition import PCA
pca_vis = PCA(n_components=2)
X_vis = pca_vis.fit_transform(X)

plt.figure()
plt.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="y=1")
plt.scatter(X_vis[y == -1, 0], X_vis[y == -1, 1], label="y=-1")
plt.legend()
plt.title("Données projetées en PCA (2D) — non linéairement séparables")
plt.show()'''

# ==========================
# 1) Définition du noyau RBF
# ==========================
C = 0.05
gamma = 1

def rbf_kernel(X1, X2, gamma=gamma):
    """
    Noyau RBF (gaussien) entre deux ensembles de points.
    X1 : (n1, d)
    X2 : (n2, d)
    Retour : (n1, n2)
    """
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    sq1 = np.sum(X1 ** 2, axis=1)[:, np.newaxis]   # (n1, 1)
    sq2 = np.sum(X2 ** 2, axis=1)[np.newaxis, :]   # (1, n2)
    sq_dist = sq1 + sq2 - 2 * X1 @ X2.T
    return np.exp(-gamma * sq_dist)


K = rbf_kernel(X, X, gamma=gamma)  # matrice de Gram

# =========================
# 2) Construire la matrice Q pour le dual
# =========================
# Dual standard :
#   min  0.5 α^T Q α - 1^T α
#   s.c. 0 <= α_k <= C
# où Q_ij = y_i y_j K(x_i, x_j)

Q = (y[:, np.newaxis] * y[np.newaxis, :]) * K  # (n, n)

# =========================
# 3) Résolution du dual avec CVXOPT
# =========================
from cvxopt import matrix, solvers

# On résout :
#   min 1/2 α^T Q α - 1^T α
#   s.c. 0 ≤ α ≤ C,  y^T α = 0


'''
Résolution manuelle du gradient 
C = 1.0          # pénalisation des erreurs
alpha = np.zeros(n)
eta = 0.001      # pas d'apprentissage
n_iter = 10000

ones = np.ones(n)

dual_values = []

for _ in range(n_iter):
    # On minimise φ(α) = 0.5 α^T Q α - 1^T α
    # Dual du SVM : max g(α) = 1^T α - 0.5 α^T Q α
    # sous 0 <= α_k <= C et y^T α = 0 (comme dans le cours)
    grad = Q @ alpha - ones
    alpha = alpha - eta * grad

    # 1) Projection sur le segment [0, C]
    alpha = np.clip(alpha, 0.0, C)

    # 2) Projection sur l'hyperplan y^T α = 0
    #    α ← α - ( (y^T α) / (y^T y) ) y
    y_dot_alpha = np.dot(y, alpha)
    y_norm_sq = np.dot(y, y)
    if y_norm_sq > 0:
        alpha = alpha - (y_dot_alpha / y_norm_sq) * y

    dual_val = np.dot(alpha, ones) - 0.5 * alpha @ Q @ alpha
    if _ % 500 == 0:
        print("It=", _, " dual=", dual_val)
        dual_values.append(dual_val)

'''



solvers.options['show_progress'] = False

P = matrix(Q)
q = matrix(-np.ones(n))

# Inégalités 0 ≤ α ≤ C
G_top = np.diag(-np.ones(n))
h_top = np.zeros(n)
G_bottom = np.diag(np.ones(n))
h_bottom = C * np.ones(n)
G = matrix(np.vstack([G_top, G_bottom]))
h = matrix(np.hstack([h_top, h_bottom]))

# Égalité y^T α = 0
a = matrix(y.reshape(1, -1))
b_eq = matrix(np.zeros(1))

solution = solvers.qp(P, q, G, h, a, b_eq)
alpha = np.array(solution['x']).flatten()

'''# ================
# (Optionnel) Simplified SMO step (not used in main solver)
# ================
def smo_step(i, j, alpha, y, K, C):
    if i == j:
        return alpha
    eta = K[i,i] + K[j,j] - 2*K[i,j]
    if eta <= 0:
        return alpha
    alpha_j = alpha[j] + y[j]*(1 - y[i]*alpha[i] - y[j]*alpha[j]) / eta
    alpha_j = np.clip(alpha_j, 0, C)
    alpha_i = alpha[i] + y[i]*y[j]*(alpha[j] - alpha_j)
    alpha_i = np.clip(alpha_i, 0, C)
    alpha[i] = alpha_i
    alpha[j] = alpha_j
    return alpha
'''
# =========================
# 4) Calcul du biais b
# =========================
# On utilise les vecteurs de support avec 0 < α_k < C

support_idx = np.where((alpha > 1e-5) & (alpha < C - 1e-5))[0]
if support_idx.size == 0:
    # fallback : prendre ceux pour lesquels alpha > 1e-5
    support_idx = np.where(alpha > 1e-5)[0]

def decision_function_raw(X_new):
    """
    f(x) = sum_k α_k y_k K(x_k, x) + b
    """
    K_new = rbf_kernel(X_new, X, gamma=gamma)   # (m, n)
    return K_new @ (alpha * y) + b

# Pour calculer b, on impose y_i (sum_j α_j y_j K(x_j, x_i) + b) = 1 pour i support
b_vals = []
for i in support_idx:
    # somme_j α_j y_j K(x_j, x_i)
    s = np.sum(alpha * y * K[:, i])
    b_vals.append(y[i] - s)
b = np.mean(b_vals)

print("Nombre de vecteurs de support :", support_idx.size)
print("b =", b)

# =========================
# 5) Prédictions du modèle
# =========================

def decision_function(X_new):
    X_new = np.asarray(X_new)
    K_new = rbf_kernel(X_new, X, gamma=gamma)
    return K_new @ (alpha * y) + b

def predict(X_new):
    scores = decision_function(X_new)
    y_pred = np.sign(scores)
    y_pred[y_pred == 0] = 1   # au cas où
    return y_pred

y_hat = predict(X)

print("Précision sur l'ensemble d'entraînement :",
      np.mean(y_hat == y))

y_pred_test = predict(X_test)
y_test_norm = np.where(y_test == 0, -1, 1)
print("Précision TEST :", np.mean(y_pred_test == y_test_norm))

# =========================
# 2) Boucle sur plusieurs valeurs de (C, gamma)
# =========================




from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

def train_svm_rbf_single(C, gamma, X, y, X_test, y_test):
    """
    Entraîne TON SVM (exactement comme ton code original)
    pour un couple (C, gamma), et renvoie :
    - nombre de vecteurs de support
    - valeur de b
    - précision train
    - précision test
    """
    n = X.shape[0]

    # Matrice de Gram
    K = rbf_kernel(X, X, gamma=gamma)

    # Matrice Q du dual : y_i y_j K(x_i, x_j)
    Q = (y[:, None] * y[None, :]) * K

    # QP : min 1/2 a^T Q a − 1^T a
    P = matrix(Q)
    q = matrix(-np.ones(n))

    # Contraintes 0 ≤ α ≤ C
    G = matrix(np.vstack([-np.eye(n), np.eye(n)]))
    h = matrix(np.hstack([np.zeros(n), C * np.ones(n)]))

    # Egalité : y^T α = 0
    A = matrix(y.reshape(1, -1))
    b_eq = matrix(np.zeros(1))

    sol = solvers.qp(P, q, G, h, A, b_eq)
    alpha = np.array(sol['x']).flatten()

    # Vecteurs de support marginaux
    support_idx = np.where((alpha > 1e-5) & (alpha < C - 1e-5))[0]
    if support_idx.size == 0:
        support_idx = np.where(alpha > 1e-5)[0]

    # Calcul de b
    b_vals = []
    for i in support_idx:
        s = np.sum(alpha * y * K[:, i])
        b_vals.append(y[i] - s)
    b = np.mean(b_vals)

    # Définition des prédicteurs
    def decision_function(X_new):
        K_new = rbf_kernel(X_new, X, gamma=gamma)
        return K_new @ (alpha * y) + b

    def predict(X_new):
        s = decision_function(X_new)
        y_pred = np.sign(s)
        y_pred[y_pred == 0] = 1
        return y_pred

    # Accuracy train
    y_hat_train = predict(X)
    train_acc = np.mean(y_hat_train == y)

    # Accuracy test
    y_test_norm = np.where(y_test == 0, -1, 1)
    y_hat_test = predict(X_test)
    test_acc = np.mean(y_hat_test == y_test_norm)

    return support_idx.size, b, train_acc, test_acc

Cs = [0.01, 0.05, 0.1, 1.0, 10.0]
gammas = [0.01, 0.1, 1.0, 10.0]

for C in Cs:
    for gamma in gammas:
        sv_count, b_val, train_acc, test_acc = train_svm_rbf_single(C, gamma, X, y, X_test, y_test)
        print(f"=== C={C}, gamma={gamma} ===")
        print(f"  Nombre de vecteurs de support : {sv_count}")
        print(f"  b = {b_val}")
        print(f"  Précision TRAIN : {train_acc:.3f}")
        print(f"  Précision TEST  : {test_acc:.3f}")
        print()

'''
# =========================
# 6) Visualisation PCA (2D)
# =========================
# Lorsque le nombre de paramètres > 2, on projette les données en 2D
# pour pouvoir tracer la frontière dans le plan principal

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

# On construit une grille dans l'espace PCA
xmin, xmax = X2[:, 0].min() - 1, X2[:, 0].max() + 1
ymin, ymax = X2[:, 1].min() - 1, X2[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(xmin, xmax, 300),
    np.linspace(ymin, ymax, 300)
)

# On repasse la grille par l'inverse de la PCA pour obtenir des points dans l'espace original
grid_2d = np.c_[xx.ravel(), yy.ravel()]
grid_original = pca.inverse_transform(grid_2d)

zz = decision_function(grid_original).reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, zz, levels=np.linspace(zz.min(), zz.max(), 50), alpha=0.3)
plt.contour(xx, yy, zz, levels=[0], colors='k', linewidths=2)

plt.scatter(X2[y == 1, 0], X2[y == 1, 1], label="y=1")
plt.scatter(X2[y == -1, 0], X2[y == -1, 1], label="y=-1")
plt.legend()
plt.title("SVM noyau RBF — visualisation PCA (2D)")
plt.show()'''