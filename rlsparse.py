import numpy as np                                   
from sklearn.linear_model import RidgeClassifier     
from sklearn.metrics import accuracy_score           

# === Load data ===
X_train = np.loadtxt("xtrain.txt")                   
X_test  = np.loadtxt("xtest.txt")              

y_train = np.loadtxt("ytrain.txt").astype(int)     
y_test_raw = np.loadtxt("ytest.txt").astype(int)   

print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test_raw:", y_test_raw.shape)  

# y_test est 2x50 → on l’aplatit en 100
y_test = y_test_raw.reshape(-1)                     

print("y_test after reshape:", y_test.shape)       

# === Ridge classification ===
alpha = 1.0                                          # Paramètre de régularisation L2 (lambda du cours)

ridge_clf = RidgeClassifier(alpha=alpha)             # Crée un classifieur Ridge avec pénalisation alpha
ridge_clf.fit(X_train, y_train)                      # Apprend les poids du classifieur sur les données d'entraînement

y_pred_ridge = ridge_clf.predict(X_test)             # Prédit les labels sur les données de test
acc_ridge = accuracy_score(y_test, y_pred_ridge)     # Calcule le taux de bonnes réponses (accuracy)
err_ridge = 1 - acc_ridge                            # Taux d'erreur = 1 - accuracy

print(f"[Ridge sklearn] alpha={alpha}  accuracy={acc_ridge:.3f}  error={err_ridge:.3f}")  # Affiche les résultats de Ridge

from sklearn.linear_model import Lasso               # Lasso (régression L1) pour sparsity learning
from sklearn.preprocessing import StandardScaler     # Pour standardiser/normaliser les features


# (Optionnel mais souvent important pour Lasso)
scaler = StandardScaler(with_mean=False)             # StandardScaler sans centrage (with_mean=False) car X est sparse C'EST LE Z SCOOOOORE !!!!
X_train_scaled = scaler.fit_transform(X_train)       
X_test_scaled  = scaler.transform(X_test)      

def make_one_vs_rest_targets(y, k):
    # labels {1..K} -> y^(k) ∈ {+1, -1}
    return np.where(y == k, 1.0, -1.0)               # Construit les cibles OVR : +1 si classe k, -1 sinon

def lasso_one_vs_rest_sklearn(X_train, y_train, alpha_l1, n_classes, max_iter=1000):
    n, d = X_train.shape                             # n = nb d'exemples, d = dimension
    W = np.zeros((n_classes, d))                     # Matrice des poids par classe (K x d)
    b = np.zeros(n_classes)                          # Vecteur des intercepts par classe (taille K)

    for k in range(1, n_classes+1):                 # Boucle sur chaque classe k = 1..K
        yk = make_one_vs_rest_targets(y_train, k)    # Construit les cibles OVR pour la classe k

        # Lasso regression pour y^(k)
        model = Lasso(alpha=alpha_l1, max_iter=max_iter)  # Crée un modèle Lasso avec régularisation L1 = alpha_l1
        model.fit(X_train, yk)                             # Apprend les poids pour séparer classe k vs le reste

        W[k-1, :] = model.coef_                           # Sauvegarde les coefficients de la classe k dans W
        b[k-1] = model.intercept_                         # Sauvegarde l'intercept de la classe k dans b

    return W, b                                           # Renvoie la matrice de poids et les intercepts

def predict_one_vs_rest(X, W, b):
    # scores: (n, K)
    scores = X @ W.T + b                                  # Calcule les scores linéaires pour chaque classe (broadcasting sur b)
    # classes 1..K
    y_pred = np.argmax(scores, axis=1) + 1                # Prend la classe avec le score maximal pour chaque exemple
    return y_pred                                         # Renvoie les prédictions de classe

def sparsity(W, tol=1e-4):
    total = W.size                                        # Nombre total de coefficients (K*d)
    nonzero = np.sum(np.abs(W) > tol)                     # Nombre de coefficients dont la valeur absolue dépasse tol
    return nonzero, nonzero / total                       # Renvoie (nb de non-zéros, ratio de non-zéros)

n_classes = int(y_train.max())                            # Nombre de classes = valeur max des labels (supposés 1..K)

lambdas = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0]        # Liste de valeurs de lambda (régularisation L1) à tester

for lam in lambdas:                                       # Boucle sur chaque valeur de lambda
    W_lasso, b_lasso = lasso_one_vs_rest_sklearn(
        X_train_scaled, y_train,
        alpha_l1=lam,                                     # Passe lambda comme pénalisation L1
        n_classes=n_classes,
        max_iter=10000
    )

    y_pred_lasso = predict_one_vs_rest(X_test_scaled, W_lasso, b_lasso)  # Prédictions Lasso OVR sur le test
    acc = accuracy_score(y_test, y_pred_lasso)           # Calcul de l'accuracy sur le test
    err = 1 - acc                                        # Taux d'erreur
    nnz, ratio = sparsity(W_lasso)                       # Mesure de la sparsité des poids (nb et ratio de non-zéros)

    print(f"[Lasso sklearn] lambda={lam:.4g}  error={err:.3f}  nnz={nnz}  ratio={ratio:.3f}")  # Affiche perf + sparsité



    '''	1.	“Use the solver lasso”
→ Tu as implémenté un Lasso multi-classe en one-vs-rest, avec normalisation et pénalisation L1 sur un problème très grande dimension.
	2.	“Do it work with the whole matrix?”
→ Oui, le solveur Lasso tourne sur la matrice complète de taille (125, 93754). Tu obtiens des solutions pour plusieurs λ, sans devoir découper ou réduire X.
	3.	“Tune the regularization factor and find a trade-off…”
→ En testant plusieurs λ, tu observes que :
	•	λ trop petit : peu de sparsité, erreur autour de 0.55–0.57,
	•	λ intermédiaire (≈ 0.03) : meilleur compromis ≈ 0.53 d’erreur avec ~710 coefficients non nuls,
	•	λ trop grand : modèle ultra sparse (parfois tous les poids à 0) mais erreur 0.72 → 0.96.
→ Tu peux retenir λ ≈ 0.03 comme bon trade-off entre erreur de classification et nombre de features importantes.

Le Lasso, grâce à sa pénalisation L1, réalise automatiquement une sélection de variables en mettant de nombreux coefficients exactement à zéro. Les features conservées correspondent aux variables les plus importantes pour la prédiction. Cela en fait un outil essentiel en high-dimensional learning lorsque d ≫ n. n étant le nombre d'exemple et d le nombre de features.  '''