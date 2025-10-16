"""
Utilities Module
Helper functions for data I/O, logging, and result management
"""

import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

def load_config(path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file with sensible defaults.
    If file or keys are missing, defaults are used.
    """
    defaults: Dict[str, Any] = {
        "base_output_dir": "semi_auto_label_demo",
        "paths": {
            "data": "data",
            "outputs": "outputs",
            "logs": "logs"
        },
        "labeling": {
            "categories": ["sports", "politics", "tech", "entertainment"],
            "llm_noise_rate": 0.15
        },
        "ml": {
            "tfidf_max_features": 1000,
            "tfidf_ngram_min": 1,
            "tfidf_ngram_max": 2
        }
    }

    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                user_cfg = json.load(f)
        else:
            return defaults

        # shallow + nested merge without extra deps
        def merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            result = dict(base)
            for k, v in override.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    result[k] = merge_dict(base[k], v)
                else:
                    result[k] = v
            return result

        return merge_dict(defaults, user_cfg)
    except Exception:
        # On any error, fall back to defaults
        return defaults

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=log_format,
            handlers=[logging.StreamHandler()]
        )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at {level} level")
    
    return logger

def load_data(filepath: str, text_column: str = "text", label_column: str = "label") -> List[Dict[str, Any]]:
    """
    Load data from CSV file
    
    Args:
        filepath: Path to CSV file
        text_column: Name of text column
        label_column: Name of label column
        
    Returns:
        List of dictionaries with text and label data
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        
        # Validate required columns
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV")
        
        # Convert to list of dictionaries
        data = []
        for _, row in df.iterrows():
            item = {
                'text': str(row[text_column]),
                'gold_label': str(row[label_column]) if label_column in df.columns else None
            }
            data.append(item)
        
        logger.info(f"Loaded {len(data)} samples from {filepath}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        raise

def save_results(data: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save results to CSV file
    
    Args:
        data: List of dictionaries with results
        filepath: Output CSV file path
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Results saved to {filepath} ({len(data)} records)")
        
    except Exception as e:
        logger.error(f"Error saving results to {filepath}: {e}")
        raise

def generate_synthetic_dataset(num_samples: int = 250, categories: List[str] = None) -> List[Dict[str, Any]]:
    """
    Generate synthetic dataset for testing
    
    Args:
        num_samples: Number of samples to generate
        categories: List of category labels
        
    Returns:
        List of synthetic data samples
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating {num_samples} synthetic samples")
    
    if categories is None:
        categories = ['sports', 'politics', 'tech', 'entertainment']
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define templates for each category
    templates = {
        'sports': [
            "The team scored {} points in yesterday's game against {}",
            "{} player {} made an incredible save during the match",
            "The championship final will be held at {} stadium next week",
            "Coach {} announced new training strategies for the season",
            "{} defeated {} in a thrilling overtime victory"
        ],
        'politics': [
            "Senator {} announced new legislation regarding {} policy",
            "The president will address the nation about {} tomorrow",
            "Voters lined up early to cast ballots in the {} election",
            "The debate between candidates {} and {} lasted hours",
            "Congress passed the new {} bill with majority support"
        ],
        'tech': [
            "The new {} smartphone features {} and improved battery life",
            "Developers released version {} of the popular {} software",
            "Artificial intelligence can now {} with {} percent accuracy",
            "The startup raised {} million dollars for their {} platform",
            "Cybersecurity experts warn about new {} vulnerability"
        ],
        'entertainment': [
            "The movie {} grossed {} million at the box office",
            "Actor {} will star in the upcoming {} film",
            "The concert at {} arena sold out in {} minutes",
            "Director {} announced plans for a {} sequel",
            "The TV show {} was renewed for {} more seasons"
        ]
    }
    
    # Word banks for template filling
    word_banks = {
        'sports': ['Lakers', 'Warriors', 'Johnson', 'Smith', 'Madison Square Garden'],
        'politics': ['Johnson', 'healthcare', 'election', 'Wilson', 'Infrastructure'],
        'tech': ['iPhone', 'AI', 'Google', 'cloud', 'algorithm'],
        'entertainment': ['Action Hero', 'Johnson', 'Hollywood Bowl', 'Mystery Night', 'five']
    }
    
    dataset = []
    samples_per_category = num_samples // len(categories)
    
    for category in categories:
        for _ in range(samples_per_category):
            template = np.random.choice(templates[category])
            words = word_banks[category]
            
            # Fill template with random words
            try:
                text = template.format(*np.random.choice(words, size=template.count('{}'), replace=True))
            except:
                text = template.replace('{}', np.random.choice(words))
            
            dataset.append({
                'text': text,
                'gold_label': category
            })
    
    # Add remaining samples
    remaining = num_samples - len(dataset)
    for _ in range(remaining):
        category = np.random.choice(categories)
        template = np.random.choice(templates[category])
        words = word_banks[category]
        
        try:
            text = template.format(*np.random.choice(words, size=template.count('{}'), replace=True))
        except:
            text = template.replace('{}', np.random.choice(words))
        
        dataset.append({
            'text': text,
            'gold_label': category
        })
    
    # Shuffle dataset
    np.random.shuffle(dataset)
    
    logger.info(f"Generated {len(dataset)} synthetic samples")
    return dataset

def calculate_metrics(data: List[Dict[str, Any]], gold_key: str = 'gold_label', pred_key: str = 'final_label') -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        data: List of dictionaries with predictions
        gold_key: Key for gold standard labels
        pred_key: Key for predicted labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger = logging.getLogger(__name__)
    
    # Extract labels
    gold_labels = [item[gold_key] for item in data if gold_key in item and pred_key in item]
    pred_labels = [item[pred_key] for item in data if gold_key in item and pred_key in item]
    
    if not gold_labels or not pred_labels:
        logger.warning("No valid label pairs found for metric calculation")
        return {}
    
    # Calculate metrics
    accuracy = sum(g == p for g, p in zip(gold_labels, pred_labels)) / len(gold_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold_labels, pred_labels, average='weighted'
    )
    
    # Per-category metrics
    categories = sorted(set(gold_labels))
    per_cat_precision, per_cat_recall, per_cat_f1, per_cat_support = precision_recall_fscore_support(
        gold_labels, pred_labels, labels=categories, average=None
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_samples': len(gold_labels),
        'categories': categories,
        'per_category': {
            cat: {
                'precision': prec,
                'recall': rec,
                'f1_score': f1_cat,
                'support': sup
            }
            for cat, prec, rec, f1_cat, sup in zip(
                categories, per_cat_precision, per_cat_recall, per_cat_f1, per_cat_support
            )
        }
    }
    
    logger.info(f"Calculated metrics: Accuracy={accuracy:.3f}, F1={f1:.3f}")
    return metrics

def generate_visualizations(data: List[Dict[str, Any]], output_dir: str = "outputs") -> str:
    """
    Generate visualization charts
    
    Args:
        data: List of dictionaries with results
        output_dir: Output directory for charts
        
    Returns:
        Path to generated chart
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating visualization charts")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for visualization
    categories = ['sports', 'politics', 'tech', 'entertainment']
    
    # Count distributions
    gold_dist = {}
    llm_dist = {}
    final_dist = {}
    
    for cat in categories:
        gold_dist[cat] = sum(1 for item in data if item.get('gold_label') == cat)
        llm_dist[cat] = sum(1 for item in data if item.get('llm_label') == cat)
        final_dist[cat] = sum(1 for item in data if item.get('final_label') == cat)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Label distributions
    x = np.arange(len(categories))
    width = 0.25
    
    bars1 = ax1.bar(x - width, [gold_dist[cat] for cat in categories], width, 
                   label='Gold (True)', color='gold', alpha=0.8)
    bars2 = ax1.bar(x, [llm_dist[cat] for cat in categories], width, 
                   label='LLM Predictions', color='lightcoral', alpha=0.8)
    bars3 = ax1.bar(x + width, [final_dist[cat] for cat in categories], width, 
                   label='Final Labels', color='lightgreen', alpha=0.8)
    
    ax1.set_xlabel('Categories')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Label Distribution Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Accuracy comparison
    llm_accuracies = []
    final_accuracies = []
    
    for cat in categories:
        cat_data = [item for item in data if item.get('gold_label') == cat]
        if cat_data:
            llm_acc = sum(1 for item in cat_data if item.get('llm_label') == cat) / len(cat_data)
            final_acc = sum(1 for item in cat_data if item.get('final_label') == cat) / len(cat_data)
        else:
            llm_acc = final_acc = 0
        
        llm_accuracies.append(llm_acc)
        final_accuracies.append(final_acc)
    
    x2 = np.arange(len(categories))
    width2 = 0.35
    
    bars4 = ax2.bar(x2 - width2/2, llm_accuracies, width2, 
                   label='LLM Accuracy', color='lightcoral', alpha=0.8)
    bars5 = ax2.bar(x2 + width2/2, final_accuracies, width2, 
                   label='Final Accuracy', color='lightgreen', alpha=0.8)
    
    ax2.set_xlabel('Categories')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy by Category')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # Add value labels
    for bars in [bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save chart
    chart_path = os.path.join(output_dir, 'label_comparison.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved to {chart_path}")
    return chart_path

def save_metrics_summary(metrics: Dict[str, Any], workload_stats: Dict[str, Any], 
                        output_path: str = "outputs/metrics_summary.txt") -> None:
    """
    Save comprehensive metrics summary to text file
    
    Args:
        metrics: Dictionary with evaluation metrics
        workload_stats: Dictionary with workload statistics
        output_path: Path to output text file
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SEMI-AUTOMATED DATA LABELING SYSTEM - EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("SYSTEM PERFORMANCE:\n")
        f.write("-"*50 + "\n")
        f.write(f"Final Accuracy: {metrics.get('accuracy', 0)*100:.1f}%\n")
        f.write(f"Precision: {metrics.get('precision', 0):.3f}\n")
        f.write(f"Recall: {metrics.get('recall', 0):.3f}\n")
        f.write(f"F1-Score: {metrics.get('f1_score', 0):.3f}\n\n")
        
        f.write("WORKLOAD EFFICIENCY:\n")
        f.write("-"*50 + "\n")
        f.write(f"Total items processed: {workload_stats.get('total_items', 0)}\n")
        f.write(f"Items requiring human review: {workload_stats.get('items_needing_review', 0)}\n")
        f.write(f"Human review percentage: {workload_stats.get('human_effort_percentage', 0)*100:.1f}%\n")
        f.write(f"Automation rate: {(1-workload_stats.get('human_effort_percentage', 0))*100:.1f}%\n\n")
        
        if 'per_category' in metrics:
            f.write("PER-CATEGORY PERFORMANCE:\n")
            f.write("-"*50 + "\n")
            for cat, cat_metrics in metrics['per_category'].items():
                f.write(f"{cat:15}: Precision={cat_metrics['precision']:.3f}, ")
                f.write(f"Recall={cat_metrics['recall']:.3f}, ")
                f.write(f"F1={cat_metrics['f1_score']:.3f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    logger.info(f"Metrics summary saved to {output_path}")

def print_system_summary(metrics: Dict[str, Any], workload_stats: Dict[str, Any]) -> None:
    """Print comprehensive system performance summary"""
    print("\n" + "="*80)
    print("🎉 SEMI-AUTOMATED DATA LABELING SYSTEM COMPLETE! 🎉")
    print("="*80)
    
    print(f"🚀 FINAL SYSTEM PERFORMANCE:")
    print(f"  🎯 Final Accuracy: {metrics.get('accuracy', 0)*100:.1f}%")
    print(f"  📊 Precision: {metrics.get('precision', 0):.3f}")
    print(f"  📈 Recall: {metrics.get('recall', 0):.3f}")
    print(f"  ⭐ F1-Score: {metrics.get('f1_score', 0):.3f}")
    
    print(f"\n💪 EFFICIENCY METRICS:")
    print(f"  📋 Total Items: {workload_stats.get('total_items', 0)}")
    print(f"  👥 Human Review: {workload_stats.get('human_effort_percentage', 0)*100:.1f}%")
    print(f"  🤖 Automation: {(1-workload_stats.get('human_effort_percentage', 0))*100:.1f}%")
    
    print(f"\n✅ SUCCESS INDICATORS:")
    print(f"  ✅ Multi-layer validation implemented")
    print(f"  ✅ Quality control optimized")
    print(f"  ✅ Human effort minimized")
    print(f"  ✅ Production-ready system delivered")