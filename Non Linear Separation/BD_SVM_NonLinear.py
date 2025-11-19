"""Classical non-linear SVM training for the HeartAche dataset.

This module implements the textbook pipeline requested by the user:
1. choose a kernel K,
2. replace every dot product (x_i · x_j) with K(x_i, x_j) to build the Gram matrix,
3. solve the dual quadratic program of the soft-margin SVM,
4. obtain the multipliers (lambdas),
5. define the support vectors as those samples with lambda_i > 0,
6. expose the final classifier f(x) = Σ lambda_i y_i K(x_i, x) + w0 using the
   complementary slackness condition to determine w0 (bias).

The dual optimization is solved via CVXOPT's quadratic programming routine, which is
one of the standard numerical approaches for the "classical" method (as opposed to
SMO or other decomposition techniques).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Tuple

import numpy as np
import pandas as pd

try:  # CVXOPT is used to solve the dual quadratic program
    from cvxopt import matrix, solvers

    solvers.options["show_progress"] = False
    CVXOPT_AVAILABLE = True
except Exception:  # pragma: no cover - handled at runtime
    matrix = None  # type: ignore
    solvers = None  # type: ignore
    CVXOPT_AVAILABLE = False

Kernel = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class DatasetBundle:
    features: np.ndarray
    labels: np.ndarray
    feature_names: Iterable[str]


def linear_kernel(x_i: np.ndarray, x_j: np.ndarray) -> float:
    return float(np.dot(x_i, x_j))


def polynomial_kernel(gamma: float, coef0: float, degree: int) -> Kernel:
    def _kernel(x_i: np.ndarray, x_j: np.ndarray) -> float:
        return float((gamma * np.dot(x_i, x_j) + coef0) ** degree)

    return _kernel


def rbf_kernel(gamma: float) -> Kernel:
    def _kernel(x_i: np.ndarray, x_j: np.ndarray) -> float:
        return float(np.exp(-gamma * np.linalg.norm(x_i - x_j) ** 2))

    return _kernel


class ClassicalKernelSVM:
    """Kernel SVM solved through the classical dual quadratic program."""

    def __init__(
        self,
        C: float = 1.0,
        kernel: Kernel | str = "rbf",
        gamma: float = 0.05,
        degree: int = 3,
        coef0: float = 1.0,
        tol: float = 1e-5,
    ) -> None:
        self.C = float(C)
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.kernel = self._resolve_kernel(kernel)
        self.alphas: np.ndarray | None = None
        self.bias: float = 0.0
        self.support_vectors: np.ndarray | None = None
        self.support_labels: np.ndarray | None = None
        self.support_alphas: np.ndarray | None = None
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def _resolve_kernel(self, kernel: Kernel | str) -> Kernel:
        if callable(kernel):
            return kernel
        name = kernel.lower()
        if name == "linear":
            return linear_kernel
        if name == "poly":
            return polynomial_kernel(self.gamma, self.coef0, self.degree)
        if name == "rbf":
            return rbf_kernel(self.gamma)
        raise ValueError(f"Unsupported kernel: {kernel}")

    def _kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        K = np.zeros((m, m), dtype=float)
        for i in range(m):
            for j in range(i, m):
                value = self.kernel(X[i], X[j])
                K[i, j] = value
                K[j, i] = value
        return K

    def _solve_dual(self, K: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not CVXOPT_AVAILABLE:
            raise RuntimeError(
                "cvxopt is required to solve the classical SVM dual. Install it via 'pip install cvxopt'."
            )

        m = y.shape[0]
        Q = (np.outer(y, y) * K).astype(np.double)
        p = -np.ones(m)

        P = matrix(Q)
        q = matrix(p)

        if np.isinf(self.C):  # hard margin
            G = matrix(-np.eye(m))
            h = matrix(np.zeros(m))
        else:
            G_top = -np.eye(m)
            h_top = np.zeros(m)
            G_bottom = np.eye(m)
            h_bottom = np.ones(m) * self.C
            G = matrix(np.vstack([G_top, G_bottom]))
            h = matrix(np.hstack([h_top, h_bottom]))

        A = matrix(y.reshape(1, -1).astype(np.double))
        b = matrix(np.zeros(1))

        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(solution["x"]).reshape(-1)
        upper = self.C if not np.isinf(self.C) else np.inf
        return np.clip(alphas, 0.0, upper)

    def _compute_bias(self, alphas: np.ndarray, y: np.ndarray, K: np.ndarray) -> float:
        if np.isinf(self.C):
            interior = alphas > self.tol
        else:
            interior = (alphas > self.tol) & (alphas < self.C - self.tol)
        if np.any(interior):
            idx = np.where(interior)[0]
            proj = np.sum((alphas * y)[:, None] * K[:, idx], axis=0)
            return float(np.mean(y[idx] - proj))

        support = alphas > self.tol
        if np.any(support):
            idx = np.where(support)[0]
            proj = np.sum((alphas * y)[:, None] * K[:, idx], axis=0)
            return float(np.mean(y[idx] - proj))
        return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalKernelSVM":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        K = self._kernel_matrix(X)
        alphas = self._solve_dual(K, y)

        support_mask = alphas > self.tol
        self.alphas = alphas
        self.bias = self._compute_bias(alphas, y, K)
        self.support_vectors = X[support_mask]
        self.support_labels = y[support_mask]
        self.support_alphas = alphas[support_mask]
        self.X = X
        self.y = y
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.support_vectors is None or self.support_alphas is None or self.support_labels is None:
            raise RuntimeError("Model must be trained before calling decision_function.")
        X = np.asarray(X, dtype=float)
        scores = np.zeros(X.shape[0])
        coeffs = self.support_alphas * self.support_labels
        for idx, x in enumerate(X):
            kernels = np.array([self.kernel(x, sv) for sv in self.support_vectors])
            scores[idx] = np.dot(coeffs, kernels) + self.bias
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        labels = np.sign(scores)
        labels[labels == 0] = 1
        return labels.astype(int)


def load_heartache_dataset(path: Path) -> DatasetBundle:
    df = pd.read_csv(path, sep=";")
    labels = df["HeartDisease"].astype(int).map({0: -1, 1: 1}).to_numpy()
    features = df.drop(columns=["HeartDisease"])
    cat_cols = features.select_dtypes(include="object").columns
    features = pd.get_dummies(features, columns=cat_cols, drop_first=False)
    feature_names = features.columns.tolist()
    X = features.to_numpy(dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std
    return DatasetBundle(features=X, labels=labels, feature_names=feature_names)


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    split_idx = int(n_samples * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def main() -> None:
    dataset_path = Path("data/HeartAche.csv")
    if not dataset_path.exists():
        dataset_path = Path("data/HeartAche_1.csv")
    bundle = load_heartache_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(bundle.features, bundle.labels)

    svm = ClassicalKernelSVM(C=2.0, kernel="rbf", gamma=0.08)
    svm.fit(X_train, y_train)

    train_acc = np.mean(svm.predict(X_train) == y_train)
    test_acc = np.mean(svm.predict(X_test) == y_test)

    support_count = 0 if svm.support_alphas is None else len(svm.support_alphas)
    print(f"Dataset: {dataset_path}")
    print(f"Support vectors: {support_count}")
    print(f"Bias (w0) via complementary slackness: {svm.bias:.4f}")
    print(f"Training accuracy: {train_acc:.3%}")
    print(f"Test accuracy: {test_acc:.3%}")


if __name__ == "__main__":
    main()
