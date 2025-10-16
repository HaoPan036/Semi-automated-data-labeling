"""
Semi-automated Data Labeling System - Main Entry Point
A comprehensive system combining LLM predictions, rule-based validation, 
ML model checks, and human-in-the-loop correction for optimal labeling efficiency.

Author: Semi-automated Labeling Team
Version: 1.0.0
"""

import os
import sys
import logging
from typing import Dict, Any, List

# Add src directory to Python path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import all modules from our package
from src.llm_generator import generate_labels
from src.rule_checker import check_rules
from src.model_validator import validate_labels
from src.human_review import review_labels
from src.utils import (
    setup_logging, load_data, save_results, generate_synthetic_dataset,
    calculate_metrics, generate_visualizations, save_metrics_summary,
    print_system_summary, load_config
)

# Load configuration and define standard paths
CONFIG = load_config()
BASE_DIR = CONFIG.get('base_output_dir', 'semi_auto_label_demo')
DATA_DIR = os.path.join(BASE_DIR, CONFIG.get('paths', {}).get('data', 'data'))
OUTPUTS_DIR = os.path.join(BASE_DIR, CONFIG.get('paths', {}).get('outputs', 'outputs'))
LOGS_DIR = os.path.join(BASE_DIR, CONFIG.get('paths', {}).get('logs', 'logs'))

def ensure_output_directories():
    """Create necessary output directories"""
    directories = [
        BASE_DIR,
        DATA_DIR,
        OUTPUTS_DIR,
        LOGS_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def stage1_data_generation() -> List[Dict[str, Any]]:
    """Stage 1: Generate synthetic dataset for demonstration"""
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("STAGE 1: SYNTHETIC DATA GENERATION")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print("STAGE 1: SYNTHETIC DATA GENERATION")
    print("="*80)
    
    # Generate synthetic dataset
    dataset = generate_synthetic_dataset(num_samples=250)
    
    # Save raw dataset
    save_results(dataset, os.path.join(DATA_DIR, 'raw_dataset.csv'))
    
    print(f"✅ Generated {len(dataset)} synthetic text samples")
    print(f"📁 Saved to: {os.path.join(DATA_DIR, 'raw_dataset.csv')}")
    
    # Print sample entries
    print(f"\nSample entries:")
    print("-" * 60)
    for i, sample in enumerate(dataset[:5]):
        print(f"{i+1}. [{sample['gold_label']:>13}] {sample['text'][:50]}...")
    
    logger.info(f"Stage 1 completed: {len(dataset)} samples generated")
    return dataset

def stage2_llm_labeling(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stage 2: LLM-based initial labeling simulation"""
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("STAGE 2: LLM-BASED INITIAL LABELING")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print("STAGE 2: LLM-BASED INITIAL LABELING (Mock Gemini API)")
    print("="*80)
    
    # Extract texts and gold labels
    texts = [item['text'] for item in dataset]
    gold_labels = [item['gold_label'] for item in dataset]
    
    # Generate LLM predictions with 15% noise
    llm_results = generate_labels(texts, gold_labels, noise_rate=CONFIG.get('labeling', {}).get('llm_noise_rate', 0.15))
    
    # Save LLM annotations
    save_results(llm_results, os.path.join(DATA_DIR, 'llm_annotations.csv'))
    
    # Calculate LLM accuracy
    llm_accuracy = sum(1 for item in llm_results if item['llm_label'] == item['gold_label']) / len(llm_results)
    
    print(f"🤖 LLM labeling completed with {llm_accuracy*100:.1f}% accuracy")
    print(f"📁 Saved to: {os.path.join(DATA_DIR, 'llm_annotations.csv')}")
    
    # Show some examples
    errors = [item for item in llm_results if item['llm_label'] != item['gold_label']]
    print(f"\nLLM Errors found: {len(errors)} out of {len(llm_results)}")
    
    if errors:
        print("Sample LLM errors:")
        print("-" * 60)
        for i, error in enumerate(errors[:3]):
            print(f"{i+1}. [{error['gold_label']} -> {error['llm_label']}] {error['text'][:50]}...")
    
    logger.info(f"Stage 2 completed: LLM accuracy = {llm_accuracy:.3f}")
    return llm_results

def stage3_rule_validation(llm_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stage 3: Rule-based validation and correction"""
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("STAGE 3: RULE-BASED VALIDATION")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print("STAGE 3: RULE-BASED VALIDATION")
    print("="*80)
    
    # Apply rule-based checks
    rule_results = check_rules(llm_results)
    
    # Save rule-checked results
    save_results(rule_results, os.path.join(DATA_DIR, 'rule_checked.csv'))
    
    # Calculate statistics
    flagged_items = [item for item in rule_results if item.get('needs_review', False)]
    rule_corrections = sum(1 for item in rule_results if item.get('rule_label') != item.get('llm_label'))
    
    print(f"📋 Rule validation completed")
    print(f"🚩 Items flagged for review: {len(flagged_items)} ({len(flagged_items)/len(rule_results)*100:.1f}%)")
    print(f"🔧 Rule corrections made: {rule_corrections}")
    print(f"📁 Saved to: {os.path.join(DATA_DIR, 'rule_checked.csv')}")
    
    if flagged_items:
        print(f"\nSample flagged items:")
        print("-" * 60)
        for i, item in enumerate(flagged_items[:3]):
            print(f"{i+1}. [{item['llm_label']} vs {item['rule_label']}] {item['text'][:50]}...")
    
    logger.info(f"Stage 3 completed: {len(flagged_items)} items flagged")
    return rule_results

def stage4_ml_validation(rule_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stage 4: ML model validation"""
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("STAGE 4: ML MODEL VALIDATION")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print("STAGE 4: ML MODEL VALIDATION")
    print("="*80)
    
    # Apply ML validation
    ml_results = validate_labels(rule_results)
    
    # Save ML-validated results
    save_results(ml_results, os.path.join(DATA_DIR, 'ml_validated.csv'))
    
    # Calculate statistics
    ml_flagged = sum(1 for item in ml_results if item.get('ml_flag'))
    total_flagged = sum(1 for item in ml_results if item.get('needs_review', False))
    
    print(f"🤖 ML validation completed")
    print(f"🎯 ML model trained and applied")
    print(f"🚩 Additional ML flags: {ml_flagged}")
    print(f"📋 Total items for review: {total_flagged} ({total_flagged/len(ml_results)*100:.1f}%)")
    print(f"📁 Saved to: {os.path.join(DATA_DIR, 'ml_validated.csv')}")
    
    logger.info(f"Stage 4 completed: {total_flagged} total items flagged")
    return ml_results

def stage5_human_review(ml_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stage 5: Human-in-the-loop correction (automated for demo)"""
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("STAGE 5: HUMAN-IN-THE-LOOP CORRECTION")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print("STAGE 5: HUMAN-IN-THE-LOOP CORRECTION")
    print("="*80)
    
    # Apply human review (automated using gold labels for demo)
    final_results = review_labels(ml_results, auto_mode=True)
    
    # Save final results
    save_results(final_results, os.path.join(OUTPUTS_DIR, 'final_labeled_dataset.csv'))
    
    # Calculate final statistics
    items_reviewed = sum(1 for item in final_results if item.get('human_reviewed', False))
    final_accuracy = sum(1 for item in final_results 
                        if item['final_label'] == item['gold_label']) / len(final_results)
    
    print(f"👥 Human review completed (automated for demo)")
    print(f"📝 Items reviewed: {items_reviewed}")
    print(f"🎯 Final accuracy: {final_accuracy*100:.1f}%")
    print(f"📁 Saved to: {os.path.join(OUTPUTS_DIR, 'final_labeled_dataset.csv')}")
    
    logger.info(f"Stage 5 completed: Final accuracy = {final_accuracy:.3f}")
    return final_results

def stage6_evaluation_and_visualization(final_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stage 6: Comprehensive evaluation and visualization"""
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("STAGE 6: EVALUATION AND VISUALIZATION")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print("STAGE 6: EVALUATION AND VISUALIZATION")
    print("="*80)
    
    # Calculate comprehensive metrics
    metrics = calculate_metrics(final_results, 'gold_label', 'final_label')
    
    # Calculate workload statistics
    items_reviewed = sum(1 for item in final_results if item.get('human_reviewed', False))
    workload_stats = {
        'total_items': len(final_results),
        'items_needing_review': sum(1 for item in final_results if item.get('needs_review', False)),
        'items_actually_reviewed': items_reviewed,
        'human_effort_percentage': items_reviewed / len(final_results),
        'automation_rate': (len(final_results) - items_reviewed) / len(final_results)
    }
    
    # Generate visualizations
    chart_path = generate_visualizations(final_results, OUTPUTS_DIR)
    
    # Save metrics summary
    save_metrics_summary(metrics, workload_stats, os.path.join(OUTPUTS_DIR, 'metrics_summary.txt'))
    
    print(f"📊 Evaluation metrics calculated")
    print(f"📈 Visualizations generated: {chart_path}")
    print(f"📄 Metrics summary: {os.path.join(OUTPUTS_DIR, 'metrics_summary.txt')}")
    
    # Print key metrics
    print(f"\n🏆 KEY PERFORMANCE INDICATORS:")
    print(f"  🎯 Final Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"  ⚡ Automation Rate: {workload_stats['automation_rate']*100:.1f}%")
    print(f"  📊 Precision: {metrics['precision']:.3f}")
    print(f"  📈 Recall: {metrics['recall']:.3f}")
    print(f"  ⭐ F1-Score: {metrics['f1_score']:.3f}")
    
    logger.info("Stage 6 completed: Evaluation and visualization finished")
    return {'metrics': metrics, 'workload_stats': workload_stats}

def main():
    """
    Main execution function - runs all stages of the semi-automated labeling system
    """
    # Setup logging
    ensure_output_directories()
    logger = setup_logging("INFO", os.path.join(LOGS_DIR, "system.log"))
    
    print("🚀 SEMI-AUTOMATED DATA LABELING SYSTEM")
    print("="*80)
    print("Starting comprehensive labeling pipeline...")
    print(f"All outputs will be saved to: {BASE_DIR}/")
    
    try:
        # Execute all stages sequentially
        dataset = stage1_data_generation()
        llm_results = stage2_llm_labeling(dataset)
        rule_results = stage3_rule_validation(llm_results)
        ml_results = stage4_ml_validation(rule_results)
        final_results = stage5_human_review(ml_results)
        evaluation_results = stage6_evaluation_and_visualization(final_results)
        
        # Print final system summary
        print_system_summary(
            evaluation_results['metrics'], 
            evaluation_results['workload_stats']
        )
        
        print(f"\n📁 All outputs saved to: {BASE_DIR}/")
        print("📋 Files generated:")
        print(f"  📊 {os.path.join(DATA_DIR, 'raw_dataset.csv')} - Original synthetic dataset")
        print(f"  🤖 {os.path.join(DATA_DIR, 'llm_annotations.csv')} - LLM predictions")
        print(f"  📋 {os.path.join(DATA_DIR, 'rule_checked.csv')} - Rule-validated results")
        print(f"  🎯 {os.path.join(DATA_DIR, 'ml_validated.csv')} - ML-validated results")
        print(f"  ✅ {os.path.join(OUTPUTS_DIR, 'final_labeled_dataset.csv')} - Final corrected dataset")
        print(f"  📈 {os.path.join(OUTPUTS_DIR, 'label_comparison.png')} - Visualization charts")
        print(f"  📄 {os.path.join(OUTPUTS_DIR, 'metrics_summary.txt')} - Comprehensive metrics")
        print(f"  📝 {os.path.join(LOGS_DIR, 'system.log')} - System execution log")
        
        logger.info("🎉 Semi-automated labeling system completed successfully!")
        print("\n🎉 All stages completed successfully! 🎉")
        
        return True
        
    except Exception as e:
        logger.error(f"System execution failed: {e}")
        print(f"\n❌ System execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)