import re
from typing import List, Dict, Tuple
import time


class CaptionReviewer:
    """Handle caption review, filtering, and keyword highlighting"""
    
    # Common negative words that might cause model bias
    NEGATIVE_WORDS = [
        # Emotions
        "sad", "angry", "upset", "depressed", "anxious", "worried", "scared",
        "frightened", "nervous", "stressed", "frustrated", "disappointed",
        "unhappy", "miserable", "gloomy", "melancholy", "distressed",
        # Physical states
        "tired", "exhausted", "sick", "ill", "weak", "injured", "hurt",
        "damaged", "broken", "dirty", "messy", "ugly", "old", "worn",
        # Social/behavioral
        "alone", "lonely", "isolated", "rejected", "ignored", "abandoned",
        "aggressive", "violent", "hostile", "rude", "mean", "cruel",
        # General negative
        "bad", "poor", "terrible", "horrible", "awful", "disgusting",
        "unpleasant", "negative", "wrong", "failed", "ruined"
    ]
    
    # Words that often indicate important features
    HIGHLIGHT_CATEGORIES = {
        "emotion": ["happy", "sad", "angry", "calm", "excited", "peaceful", 
                   "joyful", "content", "relaxed", "energetic", "serene"],
        "location": ["indoor", "outdoor", "studio", "street", "park", "beach",
                    "mountain", "forest", "city", "countryside", "home", "office"],
        "time": ["morning", "afternoon", "evening", "night", "sunset", "sunrise",
                "dawn", "dusk", "golden hour", "blue hour"],
        "style": ["portrait", "candid", "professional", "casual", "formal",
                 "artistic", "documentary", "fashion", "lifestyle", "editorial"]
    }
    
    def __init__(self):
        self.custom_negative_words = []
        self.trigger_words = []
        
    def set_trigger_words(self, words: List[str]):
        """Set trigger words for highlighting"""
        self.trigger_words = words
        
    def add_custom_negative_words(self, words: List[str]):
        """Add custom negative words to filter"""
        self.custom_negative_words.extend(words)
        
    def find_negative_words(self, caption: str) -> List[Tuple[str, int, int]]:
        """Find negative words in caption and return their positions"""
        found_words = []
        all_negative = self.NEGATIVE_WORDS + self.custom_negative_words
        
        for word in all_negative:
            pattern = r'\b' + re.escape(word) + r'\b'
            for match in re.finditer(pattern, caption, re.IGNORECASE):
                found_words.append((word, match.start(), match.end()))
                
        return sorted(found_words, key=lambda x: x[1])
    
    def remove_negative_words(self, caption: str) -> str:
        """Remove all negative words from caption"""
        all_negative = self.NEGATIVE_WORDS + self.custom_negative_words
        
        for word in all_negative:
            pattern = r'\b' + re.escape(word) + r'\b'
            caption = re.sub(pattern, '', caption, flags=re.IGNORECASE)
            
        # Clean up extra spaces
        caption = re.sub(r'\s+', ' ', caption).strip()
        caption = re.sub(r'\s+([.,!?])', r'\1', caption)
        
        return caption
    
    def highlight_keywords(self, caption: str) -> Dict[str, List[Tuple[str, int, int]]]:
        """Find positions of keywords for highlighting"""
        highlights = {
            "trigger": [],
            "negative": [],
            "emotion": [],
            "location": [],
            "time": [],
            "style": []
        }
        
        # Find trigger words
        for word in self.trigger_words:
            pattern = r'\b' + re.escape(word) + r'\b'
            for match in re.finditer(pattern, caption, re.IGNORECASE):
                highlights["trigger"].append((word, match.start(), match.end()))
        
        # Find negative words
        highlights["negative"] = self.find_negative_words(caption)
        
        # Find category words
        for category, words in self.HIGHLIGHT_CATEGORIES.items():
            for word in words:
                pattern = r'\b' + re.escape(word) + r'\b'
                for match in re.finditer(pattern, caption, re.IGNORECASE):
                    highlights[category].append((word, match.start(), match.end()))
        
        return highlights
    
    def batch_find_replace(self, captions: List[str], find_text: str, replace_text: str, 
                          case_sensitive: bool = False) -> List[str]:
        """Batch find and replace in all captions"""
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.escape(find_text)
        
        updated_captions = []
        for caption in captions:
            updated = re.sub(pattern, replace_text, caption, flags=flags)
            updated_captions.append(updated)
            
        return updated_captions
    
    def get_caption_stats(self, caption: str) -> Dict[str, any]:
        """Get statistics about a caption"""
        words = caption.split()
        negative_words = self.find_negative_words(caption)
        
        return {
            "word_count": len(words),
            "character_count": len(caption),
            "negative_word_count": len(negative_words),
            "has_trigger_word": any(word in caption.lower() for word in self.trigger_words),
            "negative_words": [w[0] for w in negative_words]
        }


class CaptionModelManager:
    """Manage different caption models and their configurations"""
    
    MODELS = {
        "florence2": {
            "name": "Florence-2 Large",
            "model_id": "multimodalart/Florence-2-large-no-flash-attn",
            "avg_inference_time": 2.5,  # seconds
            "accuracy": "High detail, sometimes includes emotions",
            "pros": ["Detailed descriptions", "Good object detection"],
            "cons": ["May add unwanted emotional descriptions", "Can be verbose"]
        },
        "joycaption": {
            "name": "JoyCaption (LLaVA-based)",
            "model_id": "llava-hf/llava-1.5-7b-hf",
            "avg_inference_time": 4.0,
            "accuracy": "Neutral, factual descriptions without emotional bias",
            "pros": ["No emotional descriptions", "Training-optimized", "Consistent quality"],
            "cons": ["Requires ~14GB VRAM", "Slower than Florence-2"]
        },
        "blip2": {
            "name": "BLIP-2",
            "model_id": "Salesforce/blip2-opt-2.7b",
            "avg_inference_time": 2.0,
            "accuracy": "Concise, factual descriptions",
            "pros": ["Fast", "Neutral descriptions"],
            "cons": ["Less detailed", "May miss some elements"]
        }
    }
    
    def __init__(self):
        self.current_model = "florence2"
        self.inference_times = []
        
    def get_model_info(self, model_key: str) -> Dict:
        """Get information about a specific model"""
        return self.MODELS.get(model_key, {})
    
    def get_all_models(self) -> Dict:
        """Get all available models"""
        return self.MODELS
    
    def set_current_model(self, model_key: str):
        """Set the current active model"""
        if model_key in self.MODELS:
            self.current_model = model_key
            
    def record_inference_time(self, time_seconds: float):
        """Record inference time for performance tracking"""
        self.inference_times.append(time_seconds)
        # Keep only last 100 times
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
            
    def get_avg_inference_time(self) -> float:
        """Get average inference time from recorded times"""
        if self.inference_times:
            return sum(self.inference_times) / len(self.inference_times)
        return self.MODELS[self.current_model]["avg_inference_time"]