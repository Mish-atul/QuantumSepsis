"""
FOU Quantum Kernel
==================
Multi-class quantum kernel for FOU detection using One-vs-Rest QSVM.

Architecture:
- 8 qubits (PCA: 16-dim → 8-dim)
- ZZFeatureMap with linear entanglement
- 2 repetitions
- One-vs-Rest strategy for 4-class classification

Training:
- Balanced subsample: 500 per class × 4 = 2000 samples
- Quantum kernel matrix: 2000×2000
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging
from pathlib import Path

# Qiskit imports
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_aer import AerSimulator
    from qiskit.primitives import Sampler
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logging.warning("Qiskit not available, quantum kernel will use classical fallback")

# Scikit-learn imports
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FouQuantumKernel:
    """Multi-class quantum kernel for FOU detection."""

    def __init__(
        self,
        n_qubits: int = 8,
        feature_map: str = "ZZFeatureMap",
        entanglement: str = "linear",
        reps: int = 2,
        backend: str = "aer_simulator"
    ):
        """
        Args:
            n_qubits: Number of qubits
            feature_map: Feature map type
            entanglement: Entanglement pattern
            reps: Number of repetitions
            backend: Qiskit backend
        """
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map
        self.entanglement = entanglement
        self.reps = reps
        self.backend_name = backend

        self.pca = None
        self.qsvm = None
        self.kernel_matrix_train = None

        if not QISKIT_AVAILABLE:
            logger.warning("Qiskit not available, using classical RBF kernel")
            self.use_classical_fallback = True
        else:
            self.use_classical_fallback = False
            self._initialize_quantum_kernel()

    def _initialize_quantum_kernel(self):
        """Initialize quantum kernel."""
        # Create feature map
        if self.feature_map_type == "ZZFeatureMap":
            self.feature_map = ZZFeatureMap(
                feature_dimension=self.n_qubits,
                reps=self.reps,
                entanglement=self.entanglement
            )
        else:
            raise ValueError(f"Unknown feature map: {self.feature_map_type}")

        # Create backend
        self.backend = AerSimulator()

        # Create quantum kernel
        self.quantum_kernel = FidelityQuantumKernel(feature_map=self.feature_map)

        logger.info(f"Initialized quantum kernel: {self.n_qubits} qubits, {self.reps} reps")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        max_samples_per_class: int = 500
    ) -> None:
        """
        Train multi-class QSVM using One-vs-Rest strategy.

        Args:
            X_train: Training embeddings, shape (N, embedding_dim)
            y_train: Training labels, shape (N,) with values in [0, 3]
            max_samples_per_class: Maximum samples per class for balanced training
        """
        logger.info("Training FOU quantum kernel (multi-class)...")

        # Balance dataset (500 samples per class)
        X_balanced, y_balanced = self._balance_dataset(X_train, y_train, max_samples_per_class)
        logger.info(f"Balanced dataset: {len(X_balanced)} samples")
        logger.info(f"Class distribution: {np.bincount(y_balanced)}")

        # PCA reduction to n_qubits dimensions
        logger.info(f"Applying PCA: {X_balanced.shape[1]} → {self.n_qubits} dimensions")
        self.pca = PCA(n_components=self.n_qubits)
        X_reduced = self.pca.fit_transform(X_balanced)

        if self.use_classical_fallback:
            # Classical RBF kernel fallback
            logger.info("Using classical RBF kernel (Qiskit not available)")
            self.qsvm = OneVsRestClassifier(
                SVC(kernel='rbf', C=0.1, class_weight='balanced', probability=True)
            )
            self.qsvm.fit(X_reduced, y_balanced)
        else:
            # Compute quantum kernel matrix
            logger.info("Computing quantum kernel matrix...")
            self.kernel_matrix_train = self.quantum_kernel.evaluate(x_vec=X_reduced)
            logger.info(f"Kernel matrix shape: {self.kernel_matrix_train.shape}")

            # Train One-vs-Rest QSVM
            logger.info("Training One-vs-Rest QSVM...")
            self.qsvm = OneVsRestClassifier(
                SVC(kernel='precomputed', C=0.1, class_weight='balanced', probability=True)
            )
            self.qsvm.fit(self.kernel_matrix_train, y_balanced)

        # Store training data for prediction
        self.X_train_reduced = X_reduced
        self.y_train = y_balanced

        logger.info("FOU quantum kernel training complete")

    def _balance_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_samples_per_class: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Balance dataset by sampling equal number of samples per class."""
        # Get unique classes that actually exist in the data
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        logger.info(f"Found {n_classes} classes in data: {unique_classes}")
        
        X_balanced = []
        y_balanced = []

        for c in unique_classes:
            class_mask = y == c
            X_class = X[class_mask]
            y_class = y[class_mask]

            if len(X_class) == 0:
                logger.warning(f"Class {c}: 0 samples, SKIPPING")
                continue

            # Sample up to max_samples_per_class
            n_samples = min(len(X_class), max_samples_per_class)
            
            if n_samples < max_samples_per_class:
                logger.warning(f"Class {c}: only {n_samples} samples (target: {max_samples_per_class})")
                
                # If very few samples, use oversampling
                if n_samples < 50:
                    logger.warning(f"Class {c}: too few samples ({n_samples}), oversampling to 50")
                    indices = np.random.choice(len(X_class), size=50, replace=True)
                else:
                    indices = np.arange(len(X_class))
            else:
                indices = np.random.choice(len(X_class), size=n_samples, replace=False)

            X_balanced.append(X_class[indices])
            y_balanced.append(y_class[indices])

        if len(X_balanced) == 0:
            raise ValueError("No classes have samples! Cannot train quantum kernel.")

        X_balanced = np.vstack(X_balanced)
        y_balanced = np.hstack(y_balanced)
        
        logger.info(f"Balanced dataset: {len(X_balanced)} total samples")
        logger.info(f"  Class distribution: {np.bincount(y_balanced.astype(int))}")

        return X_balanced, y_balanced

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X_test: Test embeddings, shape (N, embedding_dim)

        Returns:
            predictions: Predicted labels, shape (N,)
        """
        if self.qsvm is None:
            raise ValueError("Must train before prediction!")

        # PCA reduction
        X_reduced = self.pca.transform(X_test)

        if self.use_classical_fallback:
            # Classical prediction
            predictions = self.qsvm.predict(X_reduced)
        else:
            # Compute kernel matrix between test and train
            kernel_matrix_test = self.quantum_kernel.evaluate(
                x_vec=X_reduced,
                y_vec=self.X_train_reduced
            )

            # Predict using precomputed kernel
            predictions = self.qsvm.predict(kernel_matrix_test)

        return predictions

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X_test: Test embeddings, shape (N, embedding_dim)

        Returns:
            probabilities: Predicted probabilities, shape (N, n_classes)
        """
        if self.qsvm is None:
            raise ValueError("Must train before prediction!")

        # PCA reduction
        X_reduced = self.pca.transform(X_test)

        if self.use_classical_fallback:
            # Classical prediction
            probabilities = self.qsvm.predict_proba(X_reduced)
        else:
            # Compute kernel matrix between test and train
            kernel_matrix_test = self.quantum_kernel.evaluate(
                x_vec=X_reduced,
                y_vec=self.X_train_reduced
            )

            # Predict probabilities using precomputed kernel
            probabilities = self.qsvm.predict_proba(kernel_matrix_test)

        return probabilities

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate quantum kernel on test set.

        Args:
            X_test: Test embeddings
            y_test: Test labels

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        logger.info("Evaluating FOU quantum kernel...")

        # Predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')

        # Per-class metrics
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        metrics = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "class_0_f1": report['0']['f1-score'] if '0' in report else 0.0,
            "class_1_f1": report['1']['f1-score'] if '1' in report else 0.0,
            "class_2_f1": report['2']['f1-score'] if '2' in report else 0.0,
            "class_3_f1": report['3']['f1-score'] if '3' in report else 0.0,
        }

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Macro F1: {macro_f1:.4f}")
        logger.info(f"Weighted F1: {weighted_f1:.4f}")

        return metrics

    def get_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Get quantum kernel matrix for given data."""
        if self.pca is None:
            raise ValueError("Must train before computing kernel matrix!")

        X_reduced = self.pca.transform(X)

        if self.use_classical_fallback:
            # Classical RBF kernel
            from sklearn.metrics.pairwise import rbf_kernel
            return rbf_kernel(X_reduced)
        else:
            return self.quantum_kernel.evaluate(x_vec=X_reduced)


if __name__ == "__main__":
    # Test FOU quantum kernel
    print("=== FOU Quantum Kernel Test ===")

    np.random.seed(42)

    # Synthetic data: 2000 samples, 16-dim embeddings, 4 classes
    n_samples = 2000
    embedding_dim = 16
    n_classes = 4

    # Generate synthetic embeddings (class-separated)
    X_train = []
    y_train = []
    for c in range(n_classes):
        X_class = np.random.randn(n_samples // n_classes, embedding_dim) + c * 2
        y_class = np.full(n_samples // n_classes, c)
        X_train.append(X_class)
        y_train.append(y_class)

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    # Test set
    X_test = []
    y_test = []
    for c in range(n_classes):
        X_class = np.random.randn(50, embedding_dim) + c * 2
        y_class = np.full(50, c)
        X_test.append(X_class)
        y_test.append(y_class)

    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)

    # Train quantum kernel
    qkernel = FouQuantumKernel(n_qubits=8, reps=2)
    qkernel.fit(X_train, y_train, max_samples_per_class=500)

    # Evaluate
    metrics = qkernel.evaluate(X_test, y_test)

    print(f"\nTest metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")

    # Test predictions
    y_pred = qkernel.predict(X_test[:10])
    y_proba = qkernel.predict_proba(X_test[:10])

    print(f"\nSample predictions:")
    print(f"  True labels: {y_test[:10]}")
    print(f"  Predictions: {y_pred}")
    print(f"  Probabilities shape: {y_proba.shape}")

    print("\n✓ FOU quantum kernel test passed")
