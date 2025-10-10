"""
Human Review Module
Implements command-line interface for human-in-the-loop review and correction
"""

import logging
from typing import List, Dict, Any, Optional
import sys

logger = logging.getLogger(__name__)

class HumanReviewer:
    """Interactive command-line interface for human review of flagged items"""
    
    def __init__(self, categories: List[str] = None, auto_mode: bool = False):
        """
        Initialize human reviewer
        
        Args:
            categories: List of valid category labels
            auto_mode: If True, automatically apply gold labels (for demo)
        """
        self.categories = categories or ['sports', 'politics', 'tech', 'entertainment']
        self.auto_mode = auto_mode
        
    def _display_item(self, item: Dict[str, Any], index: int, total: int) -> None:
        """Display item information for review"""
        print("\n" + "="*80)
        print(f"REVIEW ITEM {index + 1} of {total}")
        print("="*80)
        print(f"Text: {item['text']}")
        print("-"*80)
        
        # Show current predictions
        llm_label = item.get('llm_label', 'N/A')
        rule_label = item.get('rule_label', 'N/A')
        ml_label = item.get('ml_label', 'N/A')
        current_label = item.get('corrected_label', llm_label)
        
        print(f"LLM Prediction:   {llm_label}")
        print(f"Rule Prediction:  {rule_label}")
        print(f"ML Prediction:    {ml_label}")
        print(f"Current Label:    {current_label}")
        
        # Show flags
        flags = []
        if item.get('rule_flag'):
            flags.append(f"Rule: {item['rule_flag']}")
        if item.get('ml_flag'):
            flags.append(f"ML: {item['ml_flag']}")
        
        if flags:
            print(f"Flags: {'; '.join(flags)}")
        
        # Show gold label if available (for demo purposes)
        if item.get('gold_label'):
            print(f"Gold Label:       {item['gold_label']} (for reference)")
    
    def _get_user_input(self, item: Dict[str, Any]) -> str:
        """Get user input for label correction"""
        if self.auto_mode:
            # Automatically use gold label if available
            if item.get('gold_label'):
                return item['gold_label']
            else:
                # Use rule label if available, otherwise current label
                return item.get('rule_label', item.get('corrected_label', item.get('llm_label')))
        
        print("\nAvailable categories:")
        for i, category in enumerate(self.categories, 1):
            print(f"  {i}. {category}")
        
        print("\nOptions:")
        print("  Enter number (1-4) to select category")
        print("  Enter 'c' to keep current label")
        print("  Enter 's' to skip this item")
        print("  Enter 'q' to quit review process")
        
        while True:
            try:
                user_input = input("\nYour choice: ").strip().lower()
                
                if user_input == 'q':
                    print("Quitting review process...")
                    sys.exit(0)
                elif user_input == 's':
                    return item.get('corrected_label', item.get('llm_label'))
                elif user_input == 'c':
                    return item.get('corrected_label', item.get('llm_label'))
                elif user_input.isdigit():
                    choice = int(user_input)
                    if 1 <= choice <= len(self.categories):
                        return self.categories[choice - 1]
                    else:
                        print(f"Please enter a number between 1 and {len(self.categories)}")
                else:
                    print("Invalid input. Please try again.")
                    
            except KeyboardInterrupt:
                print("\nReview process interrupted.")
                sys.exit(0)
            except Exception as e:
                print(f"Error: {e}. Please try again.")
    
    def review_labels(self, labeled_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Conduct human review of flagged items
        
        Args:
            labeled_data: List of dictionaries with validation results
            
        Returns:
            List of dictionaries with human corrections applied
        """
        # Filter items that need review
        items_for_review = [item for item in labeled_data if item.get('needs_review', False)]
        
        logger.info(f"Starting human review for {len(items_for_review)} flagged items")
        
        if not items_for_review:
            print("\n🎉 No items require human review! All predictions are consistent.")
            return labeled_data
        
        print(f"\n👥 Human Review Required for {len(items_for_review)} items")
        
        if self.auto_mode:
            print("🤖 Running in auto-mode: applying gold labels automatically")
        else:
            print("📝 Interactive mode: please review each flagged item")
        
        # Create a mapping for quick lookup
        review_map = {id(item): item for item in items_for_review}
        
        # Review each flagged item
        corrections_made = 0
        
        for i, item in enumerate(items_for_review):
            if not self.auto_mode:
                self._display_item(item, i, len(items_for_review))
            
            # Get human input
            corrected_label = self._get_user_input(item)
            
            # Apply correction if changed
            original_label = item.get('corrected_label', item.get('llm_label'))
            if corrected_label != original_label:
                corrections_made += 1
            
            item['final_label'] = corrected_label
            item['human_reviewed'] = True
            
            if not self.auto_mode:
                print(f"✅ Label set to: {corrected_label}")
        
        # Apply corrections to original dataset
        corrected_data = []
        for item in labeled_data:
            item_copy = item.copy()
            
            if item.get('needs_review', False):
                # Find corresponding reviewed item
                reviewed_item = review_map.get(id(item))
                if reviewed_item:
                    item_copy['final_label'] = reviewed_item['final_label']
                    item_copy['human_reviewed'] = True
                else:
                    item_copy['final_label'] = item.get('corrected_label', item.get('llm_label'))
                    item_copy['human_reviewed'] = False
            else:
                item_copy['final_label'] = item.get('corrected_label', item.get('llm_label'))
                item_copy['human_reviewed'] = False
            
            corrected_data.append(item_copy)
        
        logger.info(f"Human review completed. Made {corrections_made} corrections")
        
        if not self.auto_mode:
            print(f"\n✅ Review complete! Made {corrections_made} corrections out of {len(items_for_review)} items")
        
        return corrected_data
    
    def get_review_statistics(self, labeled_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics about human review process"""
        total_items = len(labeled_data)
        reviewed_items = sum(1 for item in labeled_data if item.get('human_reviewed', False))
        items_needing_review = sum(1 for item in labeled_data if item.get('needs_review', False))
        
        return {
            'total_items': total_items,
            'items_needing_review': items_needing_review,
            'items_actually_reviewed': reviewed_items,
            'review_completion_rate': reviewed_items / items_needing_review if items_needing_review > 0 else 1.0,
            'human_effort_percentage': reviewed_items / total_items if total_items > 0 else 0
        }

def review_labels(labeled_data: List[Dict[str, Any]], auto_mode: bool = True) -> List[Dict[str, Any]]:
    """
    Convenience function for human review of labels
    
    Args:
        labeled_data: List of dictionaries with validation results
        auto_mode: If True, automatically apply corrections (for demo)
        
    Returns:
        List of dictionaries with human corrections applied
    """
    reviewer = HumanReviewer(auto_mode=auto_mode)
    return reviewer.review_labels(labeled_data)