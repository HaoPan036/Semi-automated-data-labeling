"""
Rule Checker Module
Implements rule-based validation and correction of LLM predictions
"""

import logging
from typing import List, Dict, Any, Set

logger = logging.getLogger(__name__)

class RuleChecker:
    """Rule-based validator for checking and correcting LLM predictions"""
    
    def __init__(self, categories: List[str] = None):
        """
        Initialize rule checker with predefined rules
        
        Args:
            categories: List of valid category labels
        """
        self.categories = categories or ['sports', 'politics', 'tech', 'entertainment']
        self.rules = self._define_rules()
        
    def _define_rules(self) -> Dict[str, Dict[str, Any]]:
        """Define keyword-based rules for each category"""
        return {
            'sports': {
                'keywords': {
                    'game', 'match', 'player', 'team', 'sport', 'athlete', 'coach',
                    'championship', 'victory', 'defeat', 'score', 'goal', 'stadium',
                    'fans', 'basketball', 'football', 'soccer', 'tennis', 'baseball',
                    'hockey', 'olympic', 'swimming', 'league', 'tournament'
                },
                'min_confidence': 0.8,
                'priority': 1
            },
            'politics': {
                'keywords': {
                    'election', 'policy', 'government', 'senate', 'congress', 'president',
                    'senator', 'mayor', 'governor', 'ambassador', 'vote', 'ballot',
                    'candidate', 'debate', 'legislation', 'bill', 'law', 'supreme court',
                    'political', 'analyst', 'democracy', 'republic'
                },
                'min_confidence': 0.8,
                'priority': 1
            },
            'tech': {
                'keywords': {
                    'ai', 'artificial intelligence', 'software', 'startup', 'algorithm',
                    'data', 'cloud', 'hardware', 'app', 'device', 'smartphone', 'computer',
                    'technology', 'cybersecurity', 'virtual reality', 'developers',
                    'programming', 'digital', 'internet', 'platform', 'neural', 'machine learning'
                },
                'min_confidence': 0.8,
                'priority': 1
            },
            'entertainment': {
                'keywords': {
                    'movie', 'actor', 'film', 'cinema', 'director', 'celebrity', 'star',
                    'show', 'tv', 'television', 'concert', 'music', 'musician', 'album',
                    'artist', 'awards', 'broadway', 'streaming', 'netflix', 'entertainment',
                    'theater', 'drama', 'comedy'
                },
                'min_confidence': 0.8,
                'priority': 1
            }
        }
    
    def _apply_keyword_rules(self, text: str) -> str:
        """Apply keyword-based rules to determine category"""
        text_lower = text.lower()
        
        # Count keyword matches for each category
        scores = {}
        for category, rule_data in self.rules.items():
            score = sum(1 for keyword in rule_data['keywords'] if keyword in text_lower)
            if score > 0:
                scores[category] = score
        
        # Return category with highest score, or None if no matches
        if scores:
            return max(scores, key=scores.get)
        return None
    
    def _validate_text_constraints(self, text: str) -> bool:
        """Validate text meets basic constraints"""
        # Check text length (should be reasonable for classification)
        word_count = len(text.split())
        if word_count < 3 or word_count > 100:
            return False
        
        # Check for empty or whitespace-only text
        if not text.strip():
            return False
            
        return True
    
    def check_rules(self, labeled_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply rule-based checks to LLM predictions
        
        Args:
            labeled_data: List of dictionaries with LLM predictions
            
        Returns:
            List of dictionaries with rule-based corrections and flags
        """
        logger.info(f"Applying rule-based checks to {len(labeled_data)} predictions")
        
        corrected_data = []
        rule_corrections = 0
        rule_flags = 0
        
        for item in labeled_data:
            text = item['text']
            llm_label = item['llm_label']
            
            # Validate text constraints
            if not self._validate_text_constraints(text):
                item['rule_flag'] = 'invalid_text'
                item['rule_label'] = None
                item['needs_review'] = True
                rule_flags += 1
                corrected_data.append(item)
                continue
            
            # Apply keyword rules
            rule_predicted_label = self._apply_keyword_rules(text)
            item['rule_label'] = rule_predicted_label
            
            # Check for rule-LLM disagreement
            if rule_predicted_label and rule_predicted_label != llm_label:
                item['rule_flag'] = f"rule_suggests_{rule_predicted_label}_but_llm_predicted_{llm_label}"
                item['needs_review'] = True
                item['corrected_label'] = rule_predicted_label  # Prefer rule-based prediction
                rule_corrections += 1
                rule_flags += 1
            else:
                item['rule_flag'] = None
                item['needs_review'] = False
                item['corrected_label'] = llm_label
            
            corrected_data.append(item)
        
        logger.info(f"Rule checking completed. Made {rule_corrections} corrections, flagged {rule_flags} items for review")
        return corrected_data
    
    def get_rule_statistics(self, labeled_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics about rule application"""
        total_items = len(labeled_data)
        flagged_items = sum(1 for item in labeled_data if item.get('needs_review', False))
        rule_corrections = sum(1 for item in labeled_data if item.get('rule_label') != item.get('llm_label'))
        
        return {
            'total_items': total_items,
            'flagged_items': flagged_items,
            'rule_corrections': rule_corrections,
            'flagging_rate': flagged_items / total_items if total_items > 0 else 0,
            'correction_rate': rule_corrections / total_items if total_items > 0 else 0
        }

def check_rules(labeled_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience function for rule-based checking of LLM predictions
    
    Args:
        labeled_data: List of dictionaries with LLM predictions
        
    Returns:
        List of dictionaries with rule-based corrections
    """
    checker = RuleChecker()
    return checker.check_rules(labeled_data)