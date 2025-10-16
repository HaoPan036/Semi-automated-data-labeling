"""
Model Validator Module
Implements ML model validation of labels using lightweight classifiers
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)

class ModelValidator:
    """ML model validator for checking label consistency"""
    
    def __init__(self, categories: List[str] = None):
        """
        Initialize model validator
        
        Args:
            categories: List of valid category labels
        """
        self.categories = categories or ['sports', 'politics', 'tech', 'entertainment']
        self.vectorizer = None
        self.classifier = None
        self.is_trained = False
        
    def _prepare_training_data(self, labeled_data: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """Prepare training data from labeled samples"""
        texts = []
        labels = []
        
        for item in labeled_data:
            # Use corrected labels if available, otherwise use LLM labels
            label = item.get('corrected_label', item.get('llm_label'))
            if label and label in self.categories:
                texts.append(item['text'])
                labels.append(label)
        
        return texts, labels
    
    def train_model(self, labeled_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train lightweight ML classifier using available labeled data
        
        Args:
            labeled_data: List of dictionaries with labels
            
        Returns:
            Training statistics dictionary
        """
        logger.info(f"Training ML validator on {len(labeled_data)} samples")
        
        # Prepare training data
        texts, labels = self._prepare_training_data(labeled_data)
        
        if len(texts) < 10:
            logger.warning("Insufficient training data for ML model")
            return {'status': 'insufficient_data', 'samples': len(texts)}
        
        # Split data for training and validation
        if len(texts) > 50:
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
        else:
            X_train, X_test, y_train, y_test = texts, texts, labels, labels
        
        # Create and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        
        # Train classifier
        self.classifier = LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='ovr'
        )
        
        self.classifier.fit(X_train_vectorized, y_train)
        self.is_trained = True
        
        # Evaluate model
        if len(X_test) > 0:
            X_test_vectorized = self.vectorizer.transform(X_test)
            y_pred = self.classifier.predict(X_test_vectorized)
            accuracy = accuracy_score(y_test, y_pred)
        else:
            accuracy = 0.0
        
        logger.info(f"ML model training completed. Validation accuracy: {accuracy:.3f}")
        
        return {
            'status': 'success',
            'training_samples': len(X_train),
            'validation_samples': len(X_test),
            'accuracy': accuracy,
            'features': X_train_vectorized.shape[1]
        }
    
    def predict_labels(self, texts: List[str]) -> List[str]:
        """Predict labels using trained ML model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Vectorize texts
        X_vectorized = self.vectorizer.transform(texts)
        
        # Make predictions
        predictions = self.classifier.predict(X_vectorized)
        
        return predictions.tolist()
    
    def get_prediction_confidence(self, texts: List[str]) -> List[float]:
        """Get prediction confidence scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting confidence scores")
        
        # Vectorize texts
        X_vectorized = self.vectorizer.transform(texts)
        
        # Get prediction probabilities
        probabilities = self.classifier.predict_proba(X_vectorized)
        
        # Return max probability for each prediction
        return [max(probs) for probs in probabilities]
    
    def validate_labels(self, labeled_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate labels using ML model and flag inconsistencies
        
        Args:
            labeled_data: List of dictionaries with existing labels
            
        Returns:
            List of dictionaries with ML validation results
        """
        logger.info(f"Validating {len(labeled_data)} labels using ML model")
        
        # Train model on existing data
        training_stats = self.train_model(labeled_data)
        
        if training_stats['status'] != 'success':
            logger.warning("ML model training failed, skipping ML validation")
            # Add empty ML fields
            for item in labeled_data:
                item['ml_label'] = None
                item['ml_confidence'] = None
                item['ml_flag'] = 'model_training_failed'
            return labeled_data
        
        # Get texts for prediction
        texts = [item['text'] for item in labeled_data]
        
        # Make predictions
        ml_predictions = self.predict_labels(texts)
        ml_confidences = self.get_prediction_confidence(texts)
        
        # Add ML validation results
        validated_data = []
        ml_disagreements = 0
        
        for i, item in enumerate(labeled_data):
            item_copy = item.copy()
            
            ml_label = ml_predictions[i]
            ml_confidence = ml_confidences[i]
            current_label = item.get('corrected_label', item.get('llm_label'))
            
            item_copy['ml_label'] = ml_label
            item_copy['ml_confidence'] = round(ml_confidence, 3)
            
            # Check for ML-current label disagreement
            if ml_label != current_label:
                item_copy['ml_flag'] = f"ml_suggests_{ml_label}_but_current_{current_label}"
                item_copy['needs_review'] = True
                ml_disagreements += 1
            else:
                item_copy['ml_flag'] = None
            
            validated_data.append(item_copy)
        
        logger.info(f"ML validation completed. Found {ml_disagreements} disagreements")
        
        return validated_data
    
    def get_validation_statistics(self, labeled_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics about ML validation"""
        total_items = len(labeled_data)
        ml_flags = sum(1 for item in labeled_data if item.get('ml_flag'))
        avg_confidence = np.mean([item.get('ml_confidence', 0) for item in labeled_data if item.get('ml_confidence')])
        
        return {
            'total_items': total_items,
            'ml_flagged_items': ml_flags,
            'ml_flagging_rate': ml_flags / total_items if total_items > 0 else 0,
            'average_ml_confidence': round(avg_confidence, 3) if not np.isnan(avg_confidence) else 0
        }

def validate_labels(labeled_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience function for ML model validation of labels
    
    Args:
        labeled_data: List of dictionaries with existing labels
        
    Returns:
        List of dictionaries with ML validation results
    """
    validator = ModelValidator()
    return validator.validate_labels(labeled_data)