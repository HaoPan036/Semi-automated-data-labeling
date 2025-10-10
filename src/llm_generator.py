"""
LLM Generator Module
Simulates LLM-based initial labeling using mock Gemini API functionality
"""

import numpy as np
import random
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class LLMGenerator:
    """Mock LLM generator simulating Gemini API for text classification"""
    
    def __init__(self, noise_rate: float = 0.15, categories: List[str] = None):
        """
        Initialize LLM generator with configurable noise rate
        
        Args:
            noise_rate: Fraction of labels to randomly flip (default 15%)
            categories: List of category labels
        """
        self.noise_rate = noise_rate
        self.categories = categories or ['sports', 'politics', 'tech', 'entertainment']
        self.keyword_mapping = self._build_keyword_mapping()
        
        # Set random seed for reproducible results
        random.seed(42)
        np.random.seed(123)
        
    def _build_keyword_mapping(self) -> Dict[str, str]:
        """Build keyword to category mapping for realistic predictions"""
        return {
            # Sports keywords
            'game': 'sports', 'match': 'sports', 'player': 'sports', 'team': 'sports',
            'sport': 'sports', 'athlete': 'sports', 'coach': 'sports', 'championship': 'sports',
            'victory': 'sports', 'defeat': 'sports', 'score': 'sports', 'goal': 'sports',
            'stadium': 'sports', 'fans': 'sports', 'basketball': 'sports', 'football': 'sports',
            'soccer': 'sports', 'tennis': 'sports', 'baseball': 'sports', 'hockey': 'sports',
            'olympic': 'sports', 'swimming': 'sports',
            
            # Politics keywords
            'election': 'politics', 'policy': 'politics', 'government': 'politics',
            'senate': 'politics', 'congress': 'politics', 'president': 'politics',
            'senator': 'politics', 'mayor': 'politics', 'governor': 'politics',
            'ambassador': 'politics', 'vote': 'politics', 'ballot': 'politics',
            'candidate': 'politics', 'debate': 'politics', 'legislation': 'politics',
            'bill': 'politics', 'law': 'politics', 'political': 'politics',
            
            # Tech keywords
            'ai': 'tech', 'artificial': 'tech', 'software': 'tech', 'startup': 'tech',
            'algorithm': 'tech', 'data': 'tech', 'cloud': 'tech', 'hardware': 'tech',
            'app': 'tech', 'device': 'tech', 'smartphone': 'tech', 'computer': 'tech',
            'technology': 'tech', 'cybersecurity': 'tech', 'virtual': 'tech',
            'developers': 'tech', 'programming': 'tech', 'digital': 'tech',
            
            # Entertainment keywords
            'movie': 'entertainment', 'actor': 'entertainment', 'film': 'entertainment',
            'cinema': 'entertainment', 'director': 'entertainment', 'celebrity': 'entertainment',
            'star': 'entertainment', 'show': 'entertainment', 'television': 'entertainment',
            'concert': 'entertainment', 'music': 'entertainment', 'musician': 'entertainment',
            'album': 'entertainment', 'artist': 'entertainment', 'awards': 'entertainment',
            'broadway': 'entertainment', 'streaming': 'entertainment'
        }
    
    def _predict_category(self, text: str) -> str:
        """Predict category based on keywords with some intelligence"""
        text_lower = text.lower()
        
        # Count keyword matches for each category
        scores = {category: 0 for category in self.categories}
        
        for keyword, category in self.keyword_mapping.items():
            if keyword in text_lower:
                scores[category] += 1
        
        # Return category with highest score, or random if no matches
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=scores.get)
        else:
            return random.choice(self.categories)
    
    def generate_labels(self, texts: List[str], gold_labels: List[str] = None) -> List[Dict[str, Any]]:
        """
        Generate initial labels using mock LLM (Gemini API simulation)
        
        Args:
            texts: List of input texts to label
            gold_labels: Optional gold standard labels for simulation
            
        Returns:
            List of dictionaries with text, predicted label, and confidence
        """
        logger.info(f"Generating LLM labels for {len(texts)} texts with {self.noise_rate*100:.1f}% noise rate")
        
        labeled_data = []
        
        for i, text in enumerate(texts):
            # Get intelligent prediction
            predicted_label = self._predict_category(text)
            
            # Add noise to simulate LLM errors
            if gold_labels and random.random() < self.noise_rate:
                # Introduce noise: randomly select a different label
                other_labels = [label for label in self.categories if label != gold_labels[i]]
                predicted_label = random.choice(other_labels)
            
            # Simulate confidence score
            confidence = round(random.uniform(0.7, 0.95), 3)
            
            labeled_data.append({
                'text': text,
                'llm_label': predicted_label,
                'confidence': confidence,
                'gold_label': gold_labels[i] if gold_labels else None
            })
        
        logger.info(f"LLM labeling completed. Generated {len(labeled_data)} predictions")
        return labeled_data

def generate_labels(texts: List[str], gold_labels: List[str] = None, noise_rate: float = 0.15) -> List[Dict[str, Any]]:
    """
    Convenience function for generating labels using mock Gemini API
    
    Args:
        texts: List of input texts to label
        gold_labels: Optional gold standard labels for simulation
        noise_rate: Fraction of labels to randomly flip
        
    Returns:
        List of labeled data dictionaries
    """
    generator = LLMGenerator(noise_rate=noise_rate)
    return generator.generate_labels(texts, gold_labels)