"""
JoyCaption implementation for fluxgym
Based on fancyfeast's JoyCaption model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
import re


class JoyCaptionModel:
    """JoyCaption model wrapper for image captioning"""
    
    # Caption types supported by JoyCaption
    CAPTION_TYPES = {
        "descriptive": "Write a descriptive caption for this image in a formal tone.",
        "descriptive_informal": "Write a descriptive caption for this image in a casual tone.", 
        "training_prompt": "Write a stable diffusion prompt for this image.",
        "midjourney": "Write a MidJourney prompt for this image.",
        "booru_tags": "Write booru tags for this image.",
        "booru_tags_long": "Write booru tags for this image, include 20-30 tags.",
        "art_critic": "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "training_sd3": "Write a training prompt for Stable Diffusion 3 for this image.",
        "training_flux": "Write a training prompt for FLUX.1 [dev] for this image."
    }
    
    def __init__(self, device="cuda", torch_dtype=torch.float16, use_4bit=False):
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = None
        self.tokenizer = None
        self.image_adapter = None
        self.use_4bit = use_4bit
        
    def load_model(self):
        """Load JoyCaption model and tokenizer"""
        model_id = "fancyfeast/llama-joycaption-alpha-one-vqa-test"
        
        # Configure quantization if using 4-bit
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
            
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            quantization_config=bnb_config,
            device_map="auto" if self.use_4bit else None,
            trust_remote_code=True
        )
        
        if not self.use_4bit:
            self.model.to(self.device)
            
        self.model.eval()
        
    def generate_caption(self, image: Image.Image, caption_type: str = "training_flux", 
                        max_tokens: int = 256, temperature: float = 0.5) -> str:
        """Generate caption for an image"""
        
        if caption_type not in self.CAPTION_TYPES:
            caption_type = "training_flux"
            
        prompt = self.CAPTION_TYPES[caption_type]
        
        # Prepare the conversation
        conversation = [
            {
                "role": "system",
                "content": "You are a helpful image captioning assistant."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process image - JoyCaption expects specific image preprocessing
        # This is a simplified version - the actual model may need specific preprocessing
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        if not self.use_4bit:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                min_length=20,
                repetition_penalty=1.1,
                length_penalty=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated caption (remove the prompt)
        caption = output_text.split(input_text)[-1].strip()
        
        # Clean up the caption
        caption = self._clean_caption(caption)
        
        return caption
    
    def _clean_caption(self, caption: str) -> str:
        """Clean up generated caption"""
        # Remove any remaining special tokens
        caption = re.sub(r'<[^>]+>', '', caption)
        
        # Remove multiple spaces
        caption = re.sub(r'\s+', ' ', caption)
        
        # Remove leading/trailing whitespace
        caption = caption.strip()
        
        # Remove any prompt leakage
        for prompt in self.CAPTION_TYPES.values():
            caption = caption.replace(prompt, "").strip()
            
        return caption
    
    def unload_model(self):
        """Free up memory by unloading the model"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()