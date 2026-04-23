"""
QuantumSepsis Shield - Quantum Kernel Module
============================================

Phase 2 implementation for running a QSVM-style classifier on real LSTM
embeddings without attempting an impossible full 4M x 4M kernel matrix.

Workflow:
  1. Load train/val/test embeddings from lstm_embeddings.npz
  2. Balanced subsample the training embeddings to a tractable size
  3. Fit PCA from 16 -> 8 dimensions on the sampled training embeddings
  4. Train an SVM with a precomputed kernel on the sampled set
  5. Score the full test set in batches using support vectors only

If Qiskit is installed, the kernel can use FidelityQuantumKernel. Otherwise,
the module falls back to an RBF kernel so the pipeline remains runnable.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import QuantumConfig, get_default_config
from src.evaluation.metrics import compute_all_metrics, format_metrics

logger = logging.getLogger(__name__)


class QuantumKernelSepsis:
    """Kernel classifier over LSTM embeddings with tractable subsampling."""

    def __init__(
        self,
        config: Optional[QuantumConfig] = None,
        random_state: int = 42,
    ):
        if config is None:
            config = get_default_config().quantum

        self.config = config
        self.random_state = random_state
        self.n_qubits = config.n_qubits
        self.pca_components = config.pca_components

        self.pca: Optional[PCA] = None
        self.svm: Optional[SVC] = None
        self._qiskit_kernel = None
        self._kernel_name = "rbf"
        self._fitted = False
        self._uses_precomputed_kernel = False
        self.rbf_gamma_: Optional[float] = None
        self.best_params_: Dict[str, object] = {}

        self.train_features_: Optional[np.ndarray] = None
        self.support_features_: Optional[np.ndarray] = None
        self.support_indices_: Optional[np.ndarray] = None

    @staticmethod
    def load_embeddings(path: str) -> Dict[str, np.ndarray]:
        """Load train/val/test embeddings with backward-compatible key names."""
        loaded = np.load(path)

        def pick(*names: str) -> np.ndarray:
            for name in names:
                if name in loaded:
                    return loaded[name]
            raise KeyError(f"Missing embedding key. Expected one of: {names}")

        data = {
            "train_embeddings": pick("train_embeddings", "X_train"),
            "train_labels": pick("train_labels", "y_train"),
            "val_embeddings": pick("val_embeddings", "X_val"),
            "val_labels": pick("val_labels", "y_val"),
            "test_embeddings": pick("test_embeddings", "X_test"),
            "test_labels": pick("test_labels", "y_test"),
        }

        for key, value in data.items():
            if "labels" in key:
                data[key] = value.astype(np.int8, copy=False)
            else:
                data[key] = value.astype(np.float32, copy=False)

        return data

    def balanced_subsample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_samples: int = 5000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a balanced train subset capped by max_samples total points."""
        if max_samples < 2:
            raise ValueError("max_samples must be at least 2 for a balanced subset")

        rng = np.random.default_rng(self.random_state)
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            raise ValueError("Both positive and negative samples are required")

        per_class = min(max_samples // 2, len(pos_idx), len(neg_idx))
        if per_class == 0:
            raise ValueError("Subsample size too small to include both classes")

        sampled_pos = rng.choice(pos_idx, size=per_class, replace=False)
        sampled_neg = rng.choice(neg_idx, size=per_class, replace=False)
        sampled_idx = np.concatenate([sampled_pos, sampled_neg])
        rng.shuffle(sampled_idx)

        X_sub = X[sampled_idx]
        y_sub = y[sampled_idx]

        logger.info(
            "Balanced subsample: %d total (%d positive, %d negative)",
            len(sampled_idx),
            int((y_sub == 1).sum()),
            int((y_sub == 0).sum()),
        )

        return X_sub, y_sub, sampled_idx

    def fit_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit PCA to reduce embeddings from 16-dim to 8-dim."""
        self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
        reduced = self.pca.fit_transform(embeddings)

        explained = float(self.pca.explained_variance_ratio_.sum())
        logger.info(
            "PCA: %d-dim -> %d-dim (explained variance: %.2f%%)",
            embeddings.shape[1],
            self.pca_components,
            explained * 100.0,
        )

        return reduced.astype(np.float32, copy=False)

    def transform_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply fitted PCA transformation."""
        if self.pca is None:
            raise RuntimeError("Must call fit_pca() before transform_pca()")
        reduced = self.pca.transform(embeddings)
        return reduced.astype(np.float32, copy=False)

    def setup_qiskit_kernel(self) -> bool:
        """Initialize Qiskit kernel if available."""
        try:
            from qiskit.circuit.library import ZZFeatureMap
            from qiskit_machine_learning.kernels import FidelityQuantumKernel

            feature_map = ZZFeatureMap(
                feature_dimension=self.n_qubits,
                reps=self.config.reps,
                entanglement=self.config.entanglement,
            )

            self._qiskit_kernel = FidelityQuantumKernel(feature_map=feature_map)
            self._kernel_name = "qiskit_fidelity"
            logger.info(
                "Qiskit kernel enabled: %d qubits, reps=%d, entanglement=%s",
                self.n_qubits,
                self.config.reps,
                self.config.entanglement,
            )
            return True
        except ImportError:
            logger.warning(
                "Qiskit not available. Falling back to classical RBF kernel."
            )
            self._qiskit_kernel = None
            self._kernel_name = "rbf"
            return False

    def compute_kernel_matrix(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute either the quantum kernel or the fallback RBF kernel."""
        if self._qiskit_kernel is not None:
            kernel = self._qiskit_kernel.evaluate(X, Y)
            return np.asarray(kernel, dtype=np.float32)

        from sklearn.metrics.pairwise import rbf_kernel

        if self.rbf_gamma_ is None:
            reference = X if Y is None else Y
            variance = float(reference.var())
            gamma = 1.0 / max(reference.shape[1] * variance, 1e-8)
        else:
            gamma = float(self.rbf_gamma_)
        kernel = rbf_kernel(X, Y, gamma=gamma)
        return kernel.astype(np.float32, copy=False)

    def tune_rbf_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 3,
    ) -> Dict[str, object]:
        """Grid-search a standard RBF SVC on the sampled training set."""
        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", "auto", 0.01, 0.1],
        }
        grid = GridSearchCV(
            estimator=SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=self.random_state,
            ),
            param_grid=param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
            refit=True,
        )
        grid.fit(X_train, y_train)

        self.best_params_ = {
            "C": grid.best_params_["C"],
            "gamma": grid.best_params_["gamma"],
            "cv_auroc": float(grid.best_score_),
        }
        logger.info(
            "Best RBF params: C=%s gamma=%s cv_auroc=%.4f",
            grid.best_params_["C"],
            grid.best_params_["gamma"],
            grid.best_score_,
        )
        return self.best_params_

    def fit_rbf(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune_hyperparameters: bool = True,
    ) -> Dict[str, float]:
        """Fit a standard RBF SVC on sampled features and keep support vectors."""
        params: Dict[str, object]
        if tune_hyperparameters:
            params = self.tune_rbf_hyperparameters(X_train, y_train)
        else:
            params = {"C": 1.0, "gamma": "scale", "cv_auroc": np.nan}
            self.best_params_ = dict(params)

        self.svm = SVC(
            kernel="rbf",
            C=float(params["C"]),
            gamma=params["gamma"],
            probability=True,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.svm.fit(X_train, y_train)

        self._kernel_name = "rbf"
        self._uses_precomputed_kernel = False
        self._fitted = True
        self.train_features_ = X_train
        self.support_indices_ = self.svm.support_.copy()
        self.support_features_ = self.svm.support_vectors_.astype(np.float32, copy=False)
        self.rbf_gamma_ = float(self.svm._gamma)

        train_scores = self.svm.predict_proba(X_train)[:, 1]
        metrics = compute_all_metrics(y_train, train_scores, prefix="train_")
        metrics["train_support_vectors"] = int(len(self.support_indices_))
        metrics["train_support_vector_ratio"] = float(len(self.support_indices_) / len(X_train))
        metrics["train_best_C"] = float(params["C"])
        metrics["train_best_gamma"] = float(self.rbf_gamma_)
        metrics["train_cv_auroc"] = float(params["cv_auroc"])

        logger.info(
            "RBF SVM support vectors: %d / %d (%.2f%%)",
            len(self.support_indices_),
            len(X_train),
            100.0 * len(self.support_indices_) / len(X_train),
        )
        return metrics

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        C: float = 1.0,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> Dict[str, float]:
        """Train the kernel SVM on the subsampled training set."""
        if self._qiskit_kernel is None:
            return self.fit_rbf(X_train, y_train, tune_hyperparameters=True)

        if class_weight is None:
            class_weight = {0: 1.0, 1: 10.0}

        logger.info("Training QSVM on %d sampled training points", len(X_train))
        K_train = self.compute_kernel_matrix(X_train)

        self.svm = SVC(
            kernel="precomputed",
            C=C,
            class_weight=class_weight,
            probability=True,
            random_state=self.random_state,
        )
        self.svm.fit(K_train, y_train)

        self._uses_precomputed_kernel = True
        self.train_features_ = X_train
        self.support_indices_ = self.svm.support_.copy()
        self.support_features_ = X_train[self.support_indices_]
        self._fitted = True

        train_scores = self.svm.predict_proba(K_train)[:, 1]
        metrics = compute_all_metrics(y_train, train_scores, prefix="train_")
        metrics["train_support_vectors"] = int(len(self.support_indices_))
        metrics["train_support_vector_ratio"] = float(len(self.support_indices_) / len(X_train))

        logger.info(
            "Support vectors: %d / %d (%.2f%%)",
            len(self.support_indices_),
            len(X_train),
            100.0 * len(self.support_indices_) / len(X_train),
        )

        return metrics

    def _decision_to_probability(self, decision: np.ndarray) -> np.ndarray:
        """Apply Platt scaling learned by SVC if available."""
        if self.svm is None:
            raise RuntimeError("Model not trained")

        prob_a = getattr(self.svm, "probA_", None)
        prob_b = getattr(self.svm, "probB_", None)
        if prob_a is None or prob_b is None or len(prob_a) == 0:
            return 1.0 / (1.0 + np.exp(-decision))

        logits = decision * prob_a[0] + prob_b[0]
        logits = np.clip(logits, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(logits))

    def decision_function_support_only(
        self,
        X: np.ndarray,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """Score samples using only support-vector kernel columns."""
        if not self._fitted or self.svm is None or self.support_features_ is None:
            raise RuntimeError("Must call fit() before decision_function_support_only()")

        if not self._uses_precomputed_kernel:
            K_batch = self.compute_kernel_matrix(X, self.support_features_)
            dual_coef = self.svm.dual_coef_.reshape(-1).astype(np.float32, copy=False)
            intercept = float(self.svm.intercept_[0])
            return (K_batch @ dual_coef + intercept).astype(np.float32, copy=False)

        dual_coef = self.svm.dual_coef_.reshape(-1).astype(np.float32, copy=False)
        intercept = float(self.svm.intercept_[0])
        outputs = np.empty(len(X), dtype=np.float32)

        for start in range(0, len(X), batch_size):
            stop = min(start + batch_size, len(X))
            K_batch = self.compute_kernel_matrix(X[start:stop], self.support_features_)
            outputs[start:stop] = K_batch @ dual_coef + intercept

        return outputs

    def predict_scores(
        self,
        X: np.ndarray,
        batch_size: int = 1024,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict risk scores and labels using support vectors only."""
        decision = self.decision_function_support_only(X, batch_size=batch_size)
        scores = self._decision_to_probability(decision)
        preds = (decision >= 0.0).astype(np.int8)
        return scores.astype(np.float32, copy=False), preds

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        prefix: str = "test_",
        batch_size: int = 1024,
    ) -> Dict[str, float]:
        """Evaluate on any split using support-vector-only inference."""
        scores, _ = self.predict_scores(X, batch_size=batch_size)
        return compute_all_metrics(y, scores, prefix=prefix)

    def get_centroids(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_centroids: int = 5,
    ) -> np.ndarray:
        """Compute centroids of positive training embeddings."""
        from sklearn.cluster import KMeans

        positives = X_train[y_train == 1]
        if len(positives) == 0:
            return np.empty((0, X_train.shape[1]), dtype=np.float32)
        if len(positives) <= n_centroids:
            return positives.astype(np.float32, copy=False)

        kmeans = KMeans(n_clusters=n_centroids, random_state=self.random_state, n_init=10)
        kmeans.fit(positives)
        centroids = kmeans.cluster_centers_.astype(np.float32, copy=False)

        logger.info("Computed %d sepsis centroids", len(centroids))
        return centroids


def run_quantum_pipeline(
    embeddings_path: str = "data/processed/lstm_embeddings.npz",
    output_path: str = "data/processed/quantum_results.json",
    max_train_samples: int = 5000,
    batch_size: int = 1024,
    use_qiskit: bool = True,
) -> Dict[str, float]:
    """Run the tractable Phase 2 quantum-kernel pipeline."""
    config = get_default_config()
    quantum = QuantumKernelSepsis(config.quantum, random_state=config.training.seed)
    data = quantum.load_embeddings(embeddings_path)

    X_train_sub, y_train_sub, train_idx = quantum.balanced_subsample(
        data["train_embeddings"],
        data["train_labels"],
        max_samples=max_train_samples,
    )

    if use_qiskit:
        quantum.setup_qiskit_kernel()

    X_train_pca = quantum.fit_pca(X_train_sub)
    X_val_pca = quantum.transform_pca(data["val_embeddings"])
    X_test_pca = quantum.transform_pca(data["test_embeddings"])

    logger.info("Running kernel training on sampled set")
    train_metrics = quantum.fit(X_train_pca, y_train_sub)

    logger.info("Evaluating validation split with support-vector-only inference")
    val_metrics = quantum.evaluate(
        X_val_pca,
        data["val_labels"],
        prefix="val_",
        batch_size=batch_size,
    )

    logger.info("Evaluating full test split with support-vector-only inference")
    test_metrics = quantum.evaluate(
        X_test_pca,
        data["test_labels"],
        prefix="test_",
        batch_size=batch_size,
    )

    results = {
        "embeddings_path": embeddings_path,
        "kernel_backend": quantum._kernel_name,
        "max_train_samples": int(max_train_samples),
        "label_stats": {
            "train_full_positive_rate": float(data["train_labels"].mean()),
            "val_positive_rate": float(data["val_labels"].mean()),
            "test_positive_rate": float(data["test_labels"].mean()),
        },
        "sampled_train_size": int(len(y_train_sub)),
        "sampled_train_positive_rate": float(y_train_sub.mean()),
        "sampled_train_indices": {
            "count": int(len(train_idx)),
            "min_index": int(train_idx.min()),
            "max_index": int(train_idx.max()),
        },
        "pca_components": int(quantum.pca_components),
        "pca_explained_variance": float(quantum.pca.explained_variance_ratio_.sum()),
        "support_vector_count": int(len(quantum.support_indices_)),
        "support_vector_ratio": float(len(quantum.support_indices_) / len(y_train_sub)),
        "best_params": quantum.best_params_,
        **train_metrics,
        **val_metrics,
        **test_metrics,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    logger.info("Quantum results saved to %s", output)

    print(format_metrics(test_metrics, "Quantum Kernel - Test Results"))

    return results


def _synthetic_smoke_test() -> None:
    """Small synthetic test for local verification."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    rng = np.random.default_rng(42)
    train_embeddings = rng.normal(size=(400, 16)).astype(np.float32)
    train_labels = np.zeros(400, dtype=np.int8)
    train_labels[:100] = 1
    rng.shuffle(train_labels)

    val_embeddings = rng.normal(size=(80, 16)).astype(np.float32)
    val_labels = rng.integers(0, 2, size=80, dtype=np.int8)
    test_embeddings = rng.normal(size=(120, 16)).astype(np.float32)
    test_labels = rng.integers(0, 2, size=120, dtype=np.int8)

    tmp_path = Path("data/processed/_synthetic_quantum_embeddings.npz")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        tmp_path,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
    )

    results = run_quantum_pipeline(
        embeddings_path=str(tmp_path),
        output_path="data/processed/_synthetic_quantum_results.json",
        max_train_samples=200,
        batch_size=32,
        use_qiskit=False,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run tractable quantum-kernel evaluation")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="data/processed/lstm_embeddings.npz",
        help="Path to npz file containing train/val/test embeddings",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/quantum_results.json",
        help="Path to save quantum results JSON",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=5000,
        help="Maximum total training samples after balanced subsampling",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for support-vector-only inference",
    )
    parser.add_argument(
        "--no-qiskit",
        action="store_true",
        help="Disable Qiskit and force the classical RBF fallback kernel",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run a synthetic smoke test instead of the real embedding pipeline",
    )
    args = parser.parse_args()

    if args.synthetic:
        _synthetic_smoke_test()
    else:
        results = run_quantum_pipeline(
            embeddings_path=args.embeddings,
            output_path=args.output,
            max_train_samples=args.max_train_samples,
            batch_size=args.batch_size,
            use_qiskit=not args.no_qiskit,
        )
        print(json.dumps(results, indent=2))
