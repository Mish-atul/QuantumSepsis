"""
Improvement Pipeline for QuantumSepsis
=======================================

Orchestrates the improvement workflow:
1. Try ensemble (LSTM + XGBoost)
2. If AUROC < 0.85: Run hyperparameter tuning
3. If AUROC >= 0.85: Proceed to quantum kernel training

Author: QuantumSepsis Team
Date: May 12, 2026
"""

import sys
import logging
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status"""
    logger.info(f"\n{'='*70}")
    logger.info(f"RUNNING: {description}")
    logger.info(f"{'='*70}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
        )
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with exit code {e.returncode}")
        return False


def load_results(filepath: Path) -> dict:
    """Load results from JSON file"""
    if not filepath.exists():
        return None
    with open(filepath, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run improvement pipeline")
    parser.add_argument('--data', type=str, default='data/processed/features.h5')
    parser.add_argument('--output-dir', type=str, default='data/processed/sepsis')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip-ensemble', action='store_true',
                        help='Skip ensemble training')
    parser.add_argument('--skip-tuning', action='store_true',
                        help='Skip hyperparameter tuning')
    parser.add_argument('--tuning-strategy', type=str, default='quick',
                        choices=['quick', 'thorough'])
    parser.add_argument('--quantum-samples', type=int, default=2000,
                        help='Samples for quantum kernel training')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_auroc = 0.85
    
    logger.info("="*70)
    logger.info("QUANTUMSEPSIS IMPROVEMENT PIPELINE")
    logger.info("="*70)
    logger.info(f"Target AUROC: {target_auroc}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info("="*70)
    
    # Track progress
    pipeline_results = {
        'start_time': datetime.now().isoformat(),
        'target_auroc': target_auroc,
        'phases': {}
    }
    
    # Phase 1: Ensemble Training
    if not args.skip_ensemble:
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: ENSEMBLE TRAINING")
        logger.info("="*70)
        
        cmd = [
            'python3', 'scripts/train_ensemble.py',
            '--data', args.data,
            '--output-dir', args.output_dir,
            '--device', args.device,
        ]
        
        success = run_command(cmd, "Ensemble Training")
        
        if success:
            # Load ensemble results
            ensemble_file = output_dir / "ensemble_results.json"
            ensemble_results = load_results(ensemble_file)
            
            if ensemble_results:
                best_auroc = ensemble_results['summary']['best_auroc']
                best_strategy = ensemble_results['summary']['best_strategy']
                
                pipeline_results['phases']['ensemble'] = {
                    'success': True,
                    'best_auroc': best_auroc,
                    'best_strategy': best_strategy,
                    'target_met': best_auroc >= target_auroc,
                }
                
                logger.info(f"\n✓ Ensemble Best AUROC: {best_auroc:.4f} ({best_strategy})")
                
                if best_auroc >= target_auroc:
                    logger.info(f"✅ TARGET MET! Proceeding to Quantum Kernel training...")
                    pipeline_results['recommendation'] = 'quantum_kernel'
                    
                    # Save and exit
                    summary_file = output_dir / "pipeline_summary.json"
                    with open(summary_file, 'w') as f:
                        json.dump(pipeline_results, f, indent=2)
                    
                    logger.info(f"\n✓ Pipeline summary saved to {summary_file}")
                    logger.info("\nNext step: Run quantum kernel training")
                    logger.info("  python3 scripts/run_quantum_kernel.py --samples 2000")
                    return
                else:
                    gap = target_auroc - best_auroc
                    logger.info(f"⚠️  Gap to target: {gap:.4f} AUROC points")
                    logger.info("   Proceeding to hyperparameter tuning...")
        else:
            pipeline_results['phases']['ensemble'] = {
                'success': False,
                'error': 'Ensemble training failed'
            }
    else:
        logger.info("\n⏭️  Skipping ensemble training")
    
    # Phase 2: Hyperparameter Tuning
    if not args.skip_tuning:
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: HYPERPARAMETER TUNING")
        logger.info("="*70)
        
        cmd = [
            'python3', 'scripts/tune_hierarchical_lstm.py',
            '--data', args.data,
            '--output-dir', args.output_dir,
            '--strategy', args.tuning_strategy,
            '--device', args.device,
        ]
        
        success = run_command(cmd, "Hyperparameter Tuning")
        
        if success:
            # Load tuning results
            tuning_file = output_dir / f"tuning_summary_{args.tuning_strategy}.json"
            tuning_results = load_results(tuning_file)
            
            if tuning_results:
                best_auroc = tuning_results['best_test_auroc']
                best_exp = tuning_results['best_experiment']
                
                pipeline_results['phases']['tuning'] = {
                    'success': True,
                    'strategy': args.tuning_strategy,
                    'best_auroc': best_auroc,
                    'best_experiment': best_exp,
                    'target_met': best_auroc >= target_auroc,
                }
                
                logger.info(f"\n✓ Tuning Best AUROC: {best_auroc:.4f} ({best_exp})")
                
                if best_auroc >= target_auroc:
                    logger.info(f"✅ TARGET MET! Proceeding to Quantum Kernel training...")
                    pipeline_results['recommendation'] = 'quantum_kernel'
                else:
                    gap = target_auroc - best_auroc
                    logger.info(f"⚠️  Still {gap:.4f} points below target")
                    logger.info("   Recommendation: Try ensemble with tuned model or thorough tuning")
                    pipeline_results['recommendation'] = 'ensemble_with_tuned_model'
        else:
            pipeline_results['phases']['tuning'] = {
                'success': False,
                'error': 'Hyperparameter tuning failed'
            }
    else:
        logger.info("\n⏭️  Skipping hyperparameter tuning")
    
    # Phase 3: Quantum Kernel (if target met)
    if pipeline_results.get('recommendation') == 'quantum_kernel':
        logger.info("\n" + "="*70)
        logger.info("PHASE 3: QUANTUM KERNEL TRAINING")
        logger.info("="*70)
        
        # Check if embeddings exist
        embeddings_file = output_dir / "hierarchical_lstm_embeddings.npz"
        if not embeddings_file.exists():
            logger.warning("⚠️  Embeddings not found. Run embedding extraction first.")
            logger.info("   python3 extract_fou_embeddings.py")
        else:
            cmd = [
                'python3', 'scripts/run_quantum_kernel.py',
                '--embeddings', str(embeddings_file),
                '--samples', str(args.quantum_samples),
                '--output-dir', args.output_dir,
            ]
            
            success = run_command(cmd, "Quantum Kernel Training")
            
            if success:
                pipeline_results['phases']['quantum_kernel'] = {
                    'success': True,
                    'samples': args.quantum_samples,
                }
                logger.info("✓ Quantum kernel training completed")
            else:
                pipeline_results['phases']['quantum_kernel'] = {
                    'success': False,
                    'error': 'Quantum kernel training failed'
                }
    
    # Final summary
    pipeline_results['end_time'] = datetime.now().isoformat()
    
    summary_file = output_dir / "pipeline_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(pipeline_results, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*70)
    
    for phase, results in pipeline_results['phases'].items():
        status = "✓" if results.get('success') else "✗"
        logger.info(f"{status} {phase.upper()}: {results}")
    
    if 'recommendation' in pipeline_results:
        logger.info(f"\n📋 RECOMMENDATION: {pipeline_results['recommendation']}")
    
    logger.info(f"\n✓ Pipeline summary saved to {summary_file}")
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
