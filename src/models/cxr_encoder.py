"""
QuantumSepsis Shield — Chest X-Ray Encoder (Phase 2)
=====================================================
Uses pretrained TorchXRayVision DenseNet121 (CheXpert weights).
NO TRAINING REQUIRED. Frozen weights. Runs on CPU in ~80ms.

Outputs 18 pathology scores including:
  - Effusion (pleural effusion → organ dysfunction)
  - Pneumonia (pulmonary infection → sepsis source)
  - Lung Opacity (ARDS indicator)

Medical Context:
  - Pleural effusion appears in 40-60% of sepsis patients (capillary leak)
  - Pneumonia is the #1 source of sepsis (40% of all cases)
  - Bilateral infiltrates + PaO2/FiO2 < 300 = ARDS (Berlin definition)

Model: DenseNet121 trained on 224,316 CheXpert X-rays (Stanford)
  - Effusion AUROC: 0.93
  - Pneumonia AUROC: 0.76
  - Lung Opacity AUROC: 0.87
"""

import logging
import io
import base64
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded singletons — avoid heavy import at startup
_model = None
_pathology_names = None


def _ensure_model():
    """Load the pretrained CXR model on first use (lazy singleton)."""
    global _model, _pathology_names
    if _model is not None:
        return

    try:
        import torch
        import torchxrayvision as xrv

        logger.info("Loading TorchXRayVision DenseNet121 (CheXpert weights)...")
        _model = xrv.models.get_model("densenet121-res224-chex", from_hf_hub=True)
        _model.eval()
        _pathology_names = list(xrv.datasets.default_pathologies)
        logger.info(
            "CXR model loaded. %d pathologies: %s",
            len(_pathology_names),
            ", ".join(_pathology_names[:5]) + "...",
        )
    except ImportError:
        logger.error(
            "torchxrayvision not installed. Run: pip install torchxrayvision"
        )
        raise
    except Exception as exc:
        logger.error("Failed to load CXR model: %s", exc)
        raise


def analyze_cxr_base64(image_base64: str) -> Dict[str, float]:
    """Analyze a base64-encoded chest X-ray image.

    Args:
        image_base64: Base64-encoded JPEG/PNG image string
                      (without the data:image/... prefix)

    Returns:
        Dict mapping pathology names to confidence scores (0.0–1.0).
        Example: {"Effusion": 0.87, "Pneumonia": 0.62, ...}
    """
    _ensure_model()

    import torch
    import torchvision
    import torchxrayvision as xrv
    from PIL import Image

    # --- Decode base64 → PIL Image (grayscale) ---
    image_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(image_bytes)).convert("L")

    # --- Convert to numpy and normalise ---
    img_np = np.array(img, dtype=np.float32)
    img_np = xrv.datasets.normalize(img_np, 255)

    # Ensure single-channel: (H, W) → (1, H, W)
    if img_np.ndim == 2:
        img_np = img_np[None, ...]

    # --- Resize to 224×224 (model input size) ---
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    img_np = transform(img_np)

    # --- Run inference ---
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # (1, 1, 224, 224)
    with torch.no_grad():
        preds = _model(img_tensor).cpu().numpy()[0]

    # --- Map to pathology names ---
    results: Dict[str, float] = {}
    for name, score in zip(_pathology_names, preds):
        results[name] = round(float(max(0.0, min(1.0, score))), 4)

    # Log key findings
    logger.info(
        "CXR analysis complete — Effusion=%.2f, Pneumonia=%.2f, Lung Opacity=%.2f",
        results.get("Effusion", 0.0),
        results.get("Pneumonia", 0.0),
        results.get("Lung Opacity", 0.0),
    )

    return results


def get_sepsis_relevant_findings(cxr_results: Dict[str, float]) -> Dict:
    """Extract sepsis-relevant findings and compute a risk modifier.

    Args:
        cxr_results: Full pathology scores from ``analyze_cxr_base64``.

    Returns:
        Dict with keys:
          - findings:       list of concerning findings [{name, score, severity}]
          - risk_modifier:  float to ADD to the ensemble risk score (0.0–0.15)
          - summary:        human-readable one-line summary
          - all_scores:     dict of pathologies with score > 0.1
    """
    effusion = cxr_results.get("Effusion", 0.0)
    pneumonia = cxr_results.get("Pneumonia", 0.0)
    opacity = cxr_results.get("Lung Opacity", 0.0)
    consolidation = cxr_results.get("Consolidation", 0.0)
    edema = cxr_results.get("Edema", 0.0)

    findings: List[Dict] = []
    risk_modifier = 0.0

    # Pleural effusion — capillary leak / organ dysfunction
    if effusion > 0.5:
        findings.append({
            "name": "Pleural Effusion",
            "score": effusion,
            "severity": "HIGH" if effusion > 0.7 else "MODERATE",
        })
        risk_modifier += 0.05 * effusion

    # Pneumonia — #1 source of sepsis
    if pneumonia > 0.4:
        findings.append({
            "name": "Pneumonia",
            "score": pneumonia,
            "severity": "HIGH" if pneumonia > 0.6 else "MODERATE",
        })
        risk_modifier += 0.08 * pneumonia

    # Lung opacity — possible ARDS
    if opacity > 0.5:
        findings.append({
            "name": "Lung Opacity (possible ARDS)",
            "score": opacity,
            "severity": "HIGH" if opacity > 0.7 else "MODERATE",
        })
        risk_modifier += 0.03 * opacity

    # Consolidation
    if consolidation > 0.5:
        findings.append({
            "name": "Consolidation",
            "score": consolidation,
            "severity": "MODERATE",
        })
        risk_modifier += 0.02 * consolidation

    # Pulmonary edema
    if edema > 0.5:
        findings.append({
            "name": "Pulmonary Edema",
            "score": edema,
            "severity": "HIGH" if edema > 0.7 else "MODERATE",
        })
        risk_modifier += 0.03 * edema

    # Cap risk modifier at +15%
    risk_modifier = min(risk_modifier, 0.15)

    # Summary
    if not findings:
        summary = "CXR: No significant pulmonary findings."
    else:
        parts = [f"{f['name']} ({f['score']:.0%})" for f in findings[:3]]
        summary = f"CXR findings: {', '.join(parts)}"

    return {
        "findings": findings,
        "risk_modifier": round(risk_modifier, 4),
        "summary": summary,
        "all_scores": {k: v for k, v in cxr_results.items() if v > 0.1},
    }
