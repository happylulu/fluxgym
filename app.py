import os
import sys
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), 'sd-scripts'))
import subprocess
import gradio as gr
from PIL import Image
import torch
import uuid
import shutil
import json
import yaml
import time
from slugify import slugify
from transformers import AutoProcessor, AutoModelForCausalLM
from gradio_logsview import LogsView, LogsViewRunner
from huggingface_hub import hf_hub_download, HfApi
from library import flux_train_utils, huggingface_util
from argparse import Namespace
import train_network
import toml
import re
import asyncio
from training_monitor import TrainingMonitor
from caption_utils import CaptionReviewer, CaptionModelManager
MAX_IMAGES = 150

with open('models.yaml', 'r') as file:
    models = yaml.safe_load(file)

def readme(base_model, lora_name, instance_prompt, sample_prompts):

    # model license
    model_config = models[base_model]
    model_file = model_config["file"]
    base_model_name = model_config["base"]
    license = None
    license_name = None
    license_link = None
    license_items = []
    if "license" in model_config:
        license = model_config["license"]
        license_items.append(f"license: {license}")
    if "license_name" in model_config:
        license_name = model_config["license_name"]
        license_items.append(f"license_name: {license_name}")
    if "license_link" in model_config:
        license_link = model_config["license_link"]
        license_items.append(f"license_link: {license_link}")
    license_str = "\n".join(license_items)
    print(f"license_items={license_items}")
    print(f"license_str = {license_str}")

    # tags
    tags = [ "text-to-image", "flux", "lora", "diffusers", "template:sd-lora", "fluxgym" ]

    # widgets
    widgets = []
    sample_image_paths = []
    output_name = slugify(lora_name)
    samples_dir = resolve_path_without_quotes(f"outputs/{output_name}/sample")
    try:
        for filename in os.listdir(samples_dir):
            # Filename Schema: [name]_[steps]_[index]_[timestamp].png
            match = re.search(r"_(\d+)_(\d+)_(\d+)\.png$", filename)
            if match:
                steps, index, timestamp = int(match.group(1)), int(match.group(2)), int(match.group(3))
                sample_image_paths.append((steps, index, f"sample/{filename}"))

        # Sort by numeric index
        sample_image_paths.sort(key=lambda x: x[0], reverse=True)

        final_sample_image_paths = sample_image_paths[:len(sample_prompts)]
        final_sample_image_paths.sort(key=lambda x: x[1])
        for i, prompt in enumerate(sample_prompts):
            _, _, image_path = final_sample_image_paths[i]
            widgets.append(
                {
                    "text": prompt,
                    "output": {
                        "url": image_path
                    },
                }
            )
    except:
        print(f"no samples")
    dtype = "torch.bfloat16"
    # Construct the README content
    readme_content = f"""---
tags:
{yaml.dump(tags, indent=4).strip()}
{"widget:" if os.path.isdir(samples_dir) else ""}
{yaml.dump(widgets, indent=4).strip() if widgets else ""}
base_model: {base_model_name}
{"instance_prompt: " + instance_prompt if instance_prompt else ""}
{license_str}
---

# {lora_name}

A Flux LoRA trained on a local computer with [Fluxgym](https://github.com/cocktailpeanut/fluxgym)

<Gallery />

## Trigger words

{"You should use `" + instance_prompt + "` to trigger the image generation." if instance_prompt else "No trigger words defined."}

## Download model and use it with ComfyUI, AUTOMATIC1111, SD.Next, Invoke AI, Forge, etc.

Weights for this model are available in Safetensors format.

"""
    return readme_content

def account_hf():
    try:
        with open("HF_TOKEN", "r") as file:
            token = file.read()
            api = HfApi(token=token)
            try:
                account = api.whoami()
                return { "token": token, "account": account['name'] }
            except:
                return None
    except:
        return None

"""
hf_logout.click(fn=logout_hf, outputs=[hf_token, hf_login, hf_logout, repo_owner])
"""
def logout_hf():
    os.remove("HF_TOKEN")
    global current_account
    current_account = account_hf()
    print(f"current_account={current_account}")
    return gr.update(value=""), gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=False)


"""
hf_login.click(fn=login_hf, inputs=[hf_token], outputs=[hf_token, hf_login, hf_logout, repo_owner])
"""
def login_hf(hf_token):
    api = HfApi(token=hf_token)
    try:
        account = api.whoami()
        if account != None:
            if "name" in account:
                with open("HF_TOKEN", "w") as file:
                    file.write(hf_token)
                global current_account
                current_account = account_hf()
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(value=current_account["account"], visible=True)
        return gr.update(), gr.update(), gr.update(), gr.update()
    except:
        print(f"incorrect hf_token")
        return gr.update(), gr.update(), gr.update(), gr.update()

def upload_hf(base_model, lora_rows, repo_owner, repo_name, repo_visibility, hf_token):
    src = lora_rows
    repo_id = f"{repo_owner}/{repo_name}"
    gr.Info(f"Uploading to Huggingface. Please Stand by...", duration=None)
    args = Namespace(
        huggingface_repo_id=repo_id,
        huggingface_repo_type="model",
        huggingface_repo_visibility=repo_visibility,
        huggingface_path_in_repo="",
        huggingface_token=hf_token,
        async_upload=False
    )
    print(f"upload_hf args={args}")
    huggingface_util.upload(args=args, src=src)
    gr.Info(f"[Upload Complete] https://huggingface.co/{repo_id}", duration=None)

def load_captioning(uploaded_files, concept_sentence):
    # Handle both file objects and file paths
    uploaded_images = []
    txt_files = []
    
    for file in uploaded_files:
        # Check if it's a file object or a string path
        if hasattr(file, 'name'):
            file_path = file.name
        else:
            file_path = file
            
        if file_path.endswith('.txt'):
            txt_files.append(file_path)
        else:
            uploaded_images.append(file_path)
    
    txt_files_dict = {os.path.splitext(os.path.basename(txt_file))[0]: txt_file for txt_file in txt_files}
    updates = []
    
    if len(uploaded_images) <= 1:
        raise gr.Error(
            "Please upload at least 2 images to train your model (the ideal number with default settings is between 4-30)"
        )
    elif len(uploaded_images) > MAX_IMAGES:
        raise gr.Error(f"For now, only {MAX_IMAGES} or less images are allowed for training")
    
    # Update for the captioning_area
    updates.append(gr.update(visible=True))
    
    # Update visibility and image for each captioning row and image
    for i in range(1, MAX_IMAGES + 1):
        # Determine if the current row and image should be visible
        visible = i <= len(uploaded_images)

        # Update visibility of the captioning row
        updates.append(gr.update(visible=visible))

        # Update for image component - display image if available, otherwise hide
        if visible:
            image_path = uploaded_images[i - 1]
            # Ensure the image exists and is accessible
            if os.path.exists(image_path):
                # Read the image using PIL to ensure it's valid
                try:
                    from PIL import Image
                    img = Image.open(image_path)
                    # Convert to RGB if needed
                    if img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGB')
                    updates.append(gr.update(value=img, visible=True))
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    updates.append(gr.update(value=None, visible=False))
            else:
                print(f"Image path does not exist: {image_path}")
                updates.append(gr.update(value=None, visible=False))
        else:
            updates.append(gr.update(value=None, visible=False))

        corresponding_caption = False
        if visible and i <= len(uploaded_images):
            image_path = uploaded_images[i - 1]
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            if base_name in txt_files_dict:
                try:
                    with open(txt_files_dict[base_name], 'r') as file:
                        corresponding_caption = file.read()
                except Exception as e:
                    print(f"Error reading caption file: {e}")

        # Update value of captioning area
        text_value = corresponding_caption if visible and corresponding_caption else concept_sentence if visible and concept_sentence else None
        updates.append(gr.update(value=text_value, visible=visible))
        
        # Update for caption_stats component
        updates.append(gr.update(visible=visible))

    # Update for the start button
    updates.append(gr.update(visible=True))
    
    print(f"DEBUG: load_captioning returning {len(updates)} updates")
    print(f"DEBUG: Expected structure: 1 (captioning_area) + {MAX_IMAGES}*4 (images) + 1 (start) = {1 + MAX_IMAGES*4 + 1}")
    
    return updates

def hide_captioning():
    return gr.update(visible=False), gr.update(visible=False)

def resize_image(image_path, output_path, size):
    with Image.open(image_path) as img:
        width, height = img.size
        if width < height:
            new_width = size
            new_height = int((size/width) * height)
        else:
            new_height = size
            new_width = int((size/height) * width)
        print(f"resize {image_path} : {new_width}x{new_height}")
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_resized.save(output_path)

def create_dataset(destination_folder, size, *inputs):
    print("Creating dataset")
    images = inputs[0]
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for index, image in enumerate(images):
        # copy the images to the datasets folder
        new_image_path = shutil.copy(image, destination_folder)

        # if it's a caption text file skip the next bit
        ext = os.path.splitext(new_image_path)[-1].lower()
        if ext == '.txt':
            continue

        # resize the images
        resize_image(new_image_path, new_image_path, size)

        # copy the captions

        original_caption = inputs[index + 1]

        image_file_name = os.path.basename(new_image_path)
        caption_file_name = os.path.splitext(image_file_name)[0] + ".txt"
        caption_path = resolve_path_without_quotes(os.path.join(destination_folder, caption_file_name))
        print(f"image_path={new_image_path}, caption_path = {caption_path}, original_caption={original_caption}")
        # if caption_path exists, do not write
        if os.path.exists(caption_path):
            print(f"{caption_path} already exists. use the existing .txt file")
        else:
            print(f"{caption_path} create a .txt caption file")
            with open(caption_path, 'w') as file:
                file.write(original_caption)

    print(f"destination_folder {destination_folder}")
    return destination_folder


def run_captioning(images, concept_sentence, model_choice, *captions):
    print(f"run_captioning with model: {model_choice}")
    print(f"concept sentence {concept_sentence}")
    print(f"captions {captions}")
    
    manager = CaptionModelManager()
    manager.set_current_model(model_choice)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    torch_dtype = torch.float16
    
    start_time = time.time()
    
    # Load model based on choice
    model = None
    processor = None
    
    try:
        if model_choice == "florence2":
            model = AutoModelForCausalLM.from_pretrained(
                "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
            ).to(device)
            processor = AutoProcessor.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)
        elif model_choice == "joycaption":
            # JoyCaption - Using Llava-based model
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            gr.Info("Loading JoyCaption model... This may take a moment.")
            
            # Using llava-1.5 as JoyCaption base
            model_id = "llava-hf/llava-1.5-7b-hf"
            processor = AutoProcessor.from_pretrained(model_id)
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            ).to(device)
            
        elif model_choice == "blip2":
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            gr.Info("Loading BLIP-2 model...")
            
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch_dtype
            ).to(device)
        else:
            # Default to Florence-2
            gr.Info(f"Model {model_choice} not configured, using Florence-2")
            model = AutoModelForCausalLM.from_pretrained(
                "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
            ).to(device)
            processor = AutoProcessor.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)
            model_choice = "florence2"
    except Exception as e:
        print(f"Error loading {model_choice}: {str(e)}")
        gr.Info(f"Error loading {model_choice}, falling back to Florence-2")
        # Fallback to Florence-2
        model = AutoModelForCausalLM.from_pretrained(
            "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
        ).to(device)
        processor = AutoProcessor.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)
        model_choice = "florence2"

    captions = list(captions)
    for i, image_path in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}")
        if isinstance(image_path, str):  # If image is a file path
            image = Image.open(image_path).convert("RGB")

        image_start = time.time()
        caption_text = ""
        
        try:
            if model_choice == "florence2":
                prompt = "<DETAILED_CAPTION>"
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
                
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"], 
                    pixel_values=inputs["pixel_values"], 
                    max_new_tokens=1024, 
                    num_beams=3
                )
                
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = processor.post_process_generation(
                    generated_text, task=prompt, image_size=(image.width, image.height)
                )
                caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
                
            elif model_choice == "joycaption":
                # JoyCaption-style prompt for neutral, training-optimized captions
                prompt = "USER: <image>\nDescribe this image focusing on visual elements, composition, style, and details without emotional interpretation.\nASSISTANT:"
                
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
                if hasattr(inputs, 'pixel_values') and inputs.pixel_values.dtype != torch_dtype:
                    inputs['pixel_values'] = inputs['pixel_values'].to(torch_dtype)
                
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.95,
                    repetition_penalty=1.1
                )
                
                caption_text = processor.decode(generated_ids[0], skip_special_tokens=True)
                # Extract only the assistant's response
                if "ASSISTANT:" in caption_text:
                    caption_text = caption_text.split("ASSISTANT:")[-1].strip()
                # Clean up any remaining prompt artifacts
                caption_text = caption_text.replace("USER:", "").replace("<image>", "").strip()
                
            elif model_choice == "blip2":
                prompt = "Describe this image in detail:"
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch_dtype)
                
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_beams=3
                )
                
                caption_text = processor.decode(generated_ids[0], skip_special_tokens=True)
                
        except Exception as e:
            print(f"Error generating caption: {str(e)}")
            caption_text = "Error generating caption"
        
        image_time = time.time() - image_start
        manager.record_inference_time(image_time)
        
        print(f"Generated caption: {caption_text}")
        if concept_sentence and concept_sentence not in caption_text:
            caption_text = f"{concept_sentence}, {caption_text}"
        captions[i] = caption_text

        yield captions
        
    total_time = time.time() - start_time
    gr.Info(f"Caption generation completed in {total_time:.1f}s (avg: {manager.get_avg_inference_time():.1f}s per image)")
    
    if model:
        model.to("cpu")
        del model
    if processor:
        del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def download(base_model):
    model = models[base_model]
    model_file = model["file"]
    repo = model["repo"]

    # download unet
    if base_model == "flux-dev" or base_model == "flux-schnell":
        unet_folder = "models/unet"
    else:
        unet_folder = f"models/unet/{repo}"
    unet_path = os.path.join(unet_folder, model_file)
    if not os.path.exists(unet_path):
        os.makedirs(unet_folder, exist_ok=True)
        gr.Info(f"Downloading base model: {base_model}. Please wait. (You can check the terminal for the download progress)", duration=None)
        print(f"download {base_model}")
        hf_hub_download(repo_id=repo, local_dir=unet_folder, filename=model_file)

    # download vae
    vae_folder = "models/vae"
    vae_path = os.path.join(vae_folder, "ae.sft")
    if not os.path.exists(vae_path):
        os.makedirs(vae_folder, exist_ok=True)
        gr.Info(f"Downloading vae")
        print(f"downloading ae.sft...")
        hf_hub_download(repo_id="cocktailpeanut/xulf-dev", local_dir=vae_folder, filename="ae.sft")

    # download clip
    clip_folder = "models/clip"
    clip_l_path = os.path.join(clip_folder, "clip_l.safetensors")
    if not os.path.exists(clip_l_path):
        os.makedirs(clip_folder, exist_ok=True)
        gr.Info(f"Downloading clip...")
        print(f"download clip_l.safetensors")
        hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", local_dir=clip_folder, filename="clip_l.safetensors")

    # download t5xxl
    t5xxl_path = os.path.join(clip_folder, "t5xxl_fp16.safetensors")
    if not os.path.exists(t5xxl_path):
        print(f"download t5xxl_fp16.safetensors")
        gr.Info(f"Downloading t5xxl...")
        hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", local_dir=clip_folder, filename="t5xxl_fp16.safetensors")


def resolve_path(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.normpath(os.path.join(current_dir, p))
    return f"\"{norm_path}\""
def resolve_path_without_quotes(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.normpath(os.path.join(current_dir, p))
    return norm_path

def gen_sh(
    base_model,
    output_name,
    resolution,
    seed,
    workers,
    learning_rate,
    network_dim,
    max_train_epochs,
    save_every_n_epochs,
    timestep_sampling,
    guidance_scale,
    vram,
    sample_prompts,
    sample_every_n_steps,
    *advanced_components
):

    print(f"gen_sh: network_dim:{network_dim}, max_train_epochs={max_train_epochs}, save_every_n_epochs={save_every_n_epochs}, timestep_sampling={timestep_sampling}, guidance_scale={guidance_scale}, vram={vram}, sample_prompts={sample_prompts}, sample_every_n_steps={sample_every_n_steps}")

    output_dir = resolve_path(f"outputs/{output_name}")
    sample_prompts_path = resolve_path(f"outputs/{output_name}/sample_prompts.txt")

    line_break = "\\"
    file_type = "sh"
    if sys.platform == "win32":
        line_break = "^"
        file_type = "bat"

    ############# Sample args ########################
    sample = ""
    if len(sample_prompts) > 0 and sample_every_n_steps > 0:
        sample = f"""--sample_prompts={sample_prompts_path} --sample_every_n_steps="{sample_every_n_steps}" {line_break}"""


    ############# Optimizer args ########################
#    if vram == "8G":
#        optimizer = f"""--optimizer_type adafactor {line_break}
#    --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
#        --split_mode {line_break}
#        --network_args "train_blocks=single" {line_break}
#        --lr_scheduler constant_with_warmup {line_break}
#        --max_grad_norm 0.0 {line_break}"""
    if vram == "16G":
        # 16G VRAM
        optimizer = f"""--optimizer_type adafactor {line_break}
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
  --lr_scheduler constant_with_warmup {line_break}
  --max_grad_norm 0.0 {line_break}"""
    elif vram == "12G":
      # 12G VRAM
        optimizer = f"""--optimizer_type adafactor {line_break}
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
  --split_mode {line_break}
  --network_args "train_blocks=single" {line_break}
  --lr_scheduler constant_with_warmup {line_break}
  --max_grad_norm 0.0 {line_break}"""
    else:
        # 20G+ VRAM
        optimizer = f"--optimizer_type adamw8bit {line_break}"


    #######################################################
    model_config = models[base_model]
    model_file = model_config["file"]
    repo = model_config["repo"]
    if base_model == "flux-dev" or base_model == "flux-schnell":
        model_folder = "models/unet"
    else:
        model_folder = f"models/unet/{repo}"
    model_path = os.path.join(model_folder, model_file)
    pretrained_model_path = resolve_path(model_path)

    clip_path = resolve_path("models/clip/clip_l.safetensors")
    t5_path = resolve_path("models/clip/t5xxl_fp16.safetensors")
    ae_path = resolve_path("models/vae/ae.sft")
    sh = f"""accelerate launch {line_break}
  --mixed_precision bf16 {line_break}
  --num_cpu_threads_per_process 1 {line_break}
  sd-scripts/flux_train_network.py {line_break}
  --pretrained_model_name_or_path {pretrained_model_path} {line_break}
  --clip_l {clip_path} {line_break}
  --t5xxl {t5_path} {line_break}
  --ae {ae_path} {line_break}
  --cache_latents_to_disk {line_break}
  --save_model_as safetensors {line_break}
  --sdpa --persistent_data_loader_workers {line_break}
  --max_data_loader_n_workers {workers} {line_break}
  --seed {seed} {line_break}
  --gradient_checkpointing {line_break}
  --mixed_precision bf16 {line_break}
  --save_precision bf16 {line_break}
  --network_module networks.lora_flux {line_break}
  --network_dim {network_dim} {line_break}
  {optimizer}{sample}
  --learning_rate {learning_rate} {line_break}
  --cache_text_encoder_outputs {line_break}
  --cache_text_encoder_outputs_to_disk {line_break}
  --fp8_base {line_break}
  --highvram {line_break}
  --max_train_epochs {max_train_epochs} {line_break}
  --save_every_n_epochs {save_every_n_epochs} {line_break}
  --dataset_config {resolve_path(f"outputs/{output_name}/dataset.toml")} {line_break}
  --output_dir {output_dir} {line_break}
  --output_name {output_name} {line_break}
  --timestep_sampling {timestep_sampling} {line_break}
  --discrete_flow_shift 3.1582 {line_break}
  --model_prediction_type raw {line_break}
  --guidance_scale {guidance_scale} {line_break}
  --loss_type l2 {line_break}"""
   


    ############# Advanced args ########################
    global advanced_component_ids
    global original_advanced_component_values
   
    # check dirty
    print(f"original_advanced_component_values = {original_advanced_component_values}")
    advanced_flags = []
    for i, current_value in enumerate(advanced_components):
#        print(f"compare {advanced_component_ids[i]}: old={original_advanced_component_values[i]}, new={current_value}")
        if original_advanced_component_values[i] != current_value:
            # dirty
            if current_value == True:
                # Boolean
                advanced_flags.append(advanced_component_ids[i])
            else:
                # string
                advanced_flags.append(f"{advanced_component_ids[i]} {current_value}")

    if len(advanced_flags) > 0:
        advanced_flags_str = f" {line_break}\n  ".join(advanced_flags)
        sh = sh + "\n  " + advanced_flags_str

    return sh

def gen_toml(
  dataset_folder,
  resolution,
  class_tokens,
  num_repeats
):
    toml = f"""[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = '{resolve_path_without_quotes(dataset_folder)}'
  class_tokens = '{class_tokens}'
  num_repeats = {num_repeats}"""
    return toml

def update_total_steps(max_train_epochs, num_repeats, images):
    try:
        num_images = len(images)
        total_steps = max_train_epochs * num_images * num_repeats
        print(f"max_train_epochs={max_train_epochs} num_images={num_images}, num_repeats={num_repeats}, total_steps={total_steps}")
        return gr.update(value = total_steps)
    except:
        print("")

def set_repo(lora_rows):
    selected_name = os.path.basename(lora_rows)
    return gr.update(value=selected_name)

def get_loras():
    try:
        outputs_path = resolve_path_without_quotes(f"outputs")
        files = os.listdir(outputs_path)
        folders = [os.path.join(outputs_path, item) for item in files if os.path.isdir(os.path.join(outputs_path, item)) and item != "sample"]
        folders.sort(key=lambda file: os.path.getctime(file), reverse=True)
        return folders
    except Exception as e:
        return []

def get_samples(lora_name):
    output_name = slugify(lora_name)
    try:
        samples_path = resolve_path_without_quotes(f"outputs/{output_name}/sample")
        files = [os.path.join(samples_path, file) for file in os.listdir(samples_path)]
        files.sort(key=lambda file: os.path.getctime(file), reverse=True)
        return files
    except:
        return []

async def start_training_async(
    base_model,
    lora_name,
    train_script,
    train_config,
    sample_prompts,
    progress_bar,
    gpu_status,
):
    """Async training with real-time monitoring"""
    # write custom script and toml
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    if not os.path.exists("outputs"):
        os.makedirs("outputs", exist_ok=True)
    output_name = slugify(lora_name)
    output_dir = resolve_path_without_quotes(f"outputs/{output_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    download(base_model)

    file_type = "sh"
    if sys.platform == "win32":
        file_type = "bat"

    sh_filename = f"train.{file_type}"
    sh_filepath = resolve_path_without_quotes(f"outputs/{output_name}/{sh_filename}")
    with open(sh_filepath, 'w', encoding="utf-8") as file:
        file.write(train_script)
    gr.Info(f"Generated train script at {sh_filename}")

    dataset_path = resolve_path_without_quotes(f"outputs/{output_name}/dataset.toml")
    with open(dataset_path, 'w', encoding="utf-8") as file:
        file.write(train_config)
    gr.Info(f"Generated dataset.toml")

    sample_prompts_path = resolve_path_without_quotes(f"outputs/{output_name}/sample_prompts.txt")
    with open(sample_prompts_path, 'w', encoding='utf-8') as file:
        file.write(sample_prompts)
    gr.Info(f"Generated sample_prompts.txt")

    # Train with monitoring
    if sys.platform == "win32":
        command = sh_filepath
    else:
        command = f"bash \"{sh_filepath}\""

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['LOG_LEVEL'] = 'DEBUG'
    env['PYTHONUNBUFFERED'] = '1'

    monitor = TrainingMonitor()
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    gr.Info(f"Started training")
    
    # Start the training process
    proc = await monitor.run_training_with_monitoring(command, cwd, env)
    
    # Monitor GPU in background
    async def gpu_monitor_task():
        while proc.returncode is None:
            gpu_info = await monitor.monitor_gpu()
            monitor.gpu_usage = gpu_info['gpu_usage_percent']
            monitor.memory_usage = gpu_info['memory_used_gb']
            
            # Update GPU status display
            gpu_text = f"GPU: {gpu_info['gpu_usage_percent']:.1f}% | Memory: {gpu_info['memory_used_gb']:.1f}/{gpu_info['memory_total_gb']:.1f} GB"
            yield gr.update(value=gpu_text), progress_bar
            
            await asyncio.sleep(2)  # Update every 2 seconds
    
    # Start GPU monitoring
    gpu_task = asyncio.create_task(gpu_monitor_task())
    
    # Process output lines
    log_output = ""
    async for line in proc.stdout:
        line = line.decode('utf-8', errors='ignore').strip()
        if line:
            log_output += line + "\n"
            
            # Parse training progress
            updates = monitor.parse_training_log(line)
            
            if updates:
                # Update progress bar
                if 'progress_percent' in updates:
                    progress = updates['progress_percent']
                    progress_text = f"Step {monitor.current_step}/{monitor.total_steps} ({progress:.1f}%)"
                    yield gr.update(value=progress, label=progress_text), gpu_status
                    
            # Yield log line
            yield log_output, gpu_status
    
    # Wait for process to complete
    await proc.wait()
    gpu_task.cancel()
    
    # Generate Readme
    config = toml.loads(train_config)
    concept_sentence = config['datasets'][0]['subsets'][0]['class_tokens']
    sample_prompts_path = resolve_path_without_quotes(f"outputs/{output_name}/sample_prompts.txt")
    with open(sample_prompts_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    sample_prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    md = readme(base_model, lora_name, concept_sentence, sample_prompts)
    readme_path = resolve_path_without_quotes(f"outputs/{output_name}/README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(md)

    gr.Info(f"Training Complete. Check the outputs folder for the LoRA files.", duration=None)
    
    # Send completion notification
    yield gr.update(value=100, label="Training Complete!"), gr.update(value="Training Complete!")


def start_training(
    base_model,
    lora_name,
    train_script,
    train_config,
    sample_prompts,
):
    """Wrapper to run async training in sync context"""
    # For now, keep the original implementation as fallback
    # write custom script and toml
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    if not os.path.exists("outputs"):
        os.makedirs("outputs", exist_ok=True)
    output_name = slugify(lora_name)
    output_dir = resolve_path_without_quotes(f"outputs/{output_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    download(base_model)

    file_type = "sh"
    if sys.platform == "win32":
        file_type = "bat"

    sh_filename = f"train.{file_type}"
    sh_filepath = resolve_path_without_quotes(f"outputs/{output_name}/{sh_filename}")
    with open(sh_filepath, 'w', encoding="utf-8") as file:
        file.write(train_script)
    gr.Info(f"Generated train script at {sh_filename}")


    dataset_path = resolve_path_without_quotes(f"outputs/{output_name}/dataset.toml")
    with open(dataset_path, 'w', encoding="utf-8") as file:
        file.write(train_config)
    gr.Info(f"Generated dataset.toml")

    sample_prompts_path = resolve_path_without_quotes(f"outputs/{output_name}/sample_prompts.txt")
    with open(sample_prompts_path, 'w', encoding='utf-8') as file:
        file.write(sample_prompts)
    gr.Info(f"Generated sample_prompts.txt")

    # Train
    if sys.platform == "win32":
        command = sh_filepath
    else:
        command = f"bash \"{sh_filepath}\""

    # Use Popen to run the command and capture output in real-time
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['LOG_LEVEL'] = 'DEBUG'
    env['PYTHONUNBUFFERED'] = '1'  # Ensure unbuffered Python output
    runner = LogsViewRunner()
    cwd = os.path.dirname(os.path.abspath(__file__))
    gr.Info(f"Started training")
    yield from runner.run_command([command], cwd=cwd, env=env)
    yield runner.log(f"Runner: {runner}")

    # Generate Readme
    config = toml.loads(train_config)
    concept_sentence = config['datasets'][0]['subsets'][0]['class_tokens']
    print(f"concept_sentence={concept_sentence}")
    print(f"lora_name {lora_name}, concept_sentence={concept_sentence}, output_name={output_name}")
    sample_prompts_path = resolve_path_without_quotes(f"outputs/{output_name}/sample_prompts.txt")
    with open(sample_prompts_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    sample_prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    md = readme(base_model, lora_name, concept_sentence, sample_prompts)
    readme_path = resolve_path_without_quotes(f"outputs/{output_name}/README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(md)

    gr.Info(f"Training Complete. Check the outputs folder for the LoRA files.", duration=None)


def update(
    base_model,
    lora_name,
    resolution,
    seed,
    workers,
    class_tokens,
    learning_rate,
    network_dim,
    max_train_epochs,
    save_every_n_epochs,
    timestep_sampling,
    guidance_scale,
    vram,
    num_repeats,
    sample_prompts,
    sample_every_n_steps,
    *advanced_components,
):
    output_name = slugify(lora_name)
    dataset_folder = str(f"datasets/{output_name}")
    sh = gen_sh(
        base_model,
        output_name,
        resolution,
        seed,
        workers,
        learning_rate,
        network_dim,
        max_train_epochs,
        save_every_n_epochs,
        timestep_sampling,
        guidance_scale,
        vram,
        sample_prompts,
        sample_every_n_steps,
        *advanced_components,
    )
    toml = gen_toml(
        dataset_folder,
        resolution,
        class_tokens,
        num_repeats
    )
    return gr.update(value=sh), gr.update(value=toml), dataset_folder

"""
demo.load(fn=loaded, js=js, outputs=[hf_token, hf_login, hf_logout, hf_account])
"""
def loaded():
    global current_account
    current_account = account_hf()
    print(f"current_account={current_account}")
    if current_account != None:
        return gr.update(value=current_account["token"]), gr.update(visible=False), gr.update(visible=True), gr.update(value=current_account["account"], visible=True)
    else:
        return gr.update(value=""), gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=False)

def update_sample(concept_sentence):
    return gr.update(value=concept_sentence)

def update_model_info(model_key):
    """Update model information display"""
    manager = CaptionModelManager()
    info = manager.get_model_info(model_key)
    if info:
        info_text = f"**{info['name']}**\n\n"
        info_text += f"🕐 Avg. inference time: {info['avg_inference_time']}s\n\n"
        info_text += f"📊 {info['accuracy']}\n\n"
        info_text += "**Pros:**\n"
        for pro in info['pros']:
            info_text += f"• {pro}\n"
        info_text += "\n**Cons:**\n"
        for con in info['cons']:
            info_text += f"• {con}\n"
        return gr.update(value=info_text)
    return gr.update(value="")

def batch_find_replace(find_text, replace_text, case_sensitive, *captions):
    """Batch find and replace in all captions"""
    if not find_text:
        return captions
    
    reviewer = CaptionReviewer()
    updated_captions = reviewer.batch_find_replace(
        list(captions), find_text, replace_text, case_sensitive
    )
    return updated_captions

def remove_negative_words(concept_sentence, custom_negative, *captions):
    """Remove negative words from all captions"""
    reviewer = CaptionReviewer()
    
    # Add trigger word
    if concept_sentence:
        reviewer.set_trigger_words([concept_sentence])
    
    # Add custom negative words
    if custom_negative:
        custom_words = [w.strip() for w in custom_negative.split(',') if w.strip()]
        reviewer.add_custom_negative_words(custom_words)
    
    # Process captions
    updated_captions = []
    total_removed = 0
    
    for caption in captions:
        if caption:
            negative_words = reviewer.find_negative_words(caption)
            total_removed += len(negative_words)
            updated_caption = reviewer.remove_negative_words(caption)
            updated_captions.append(updated_caption)
        else:
            updated_captions.append(caption)
    
    # Update info
    info_text = f"Removed {total_removed} negative words from captions"
    
    return [gr.update(value=info_text)] + updated_captions

def analyze_caption(caption, concept_sentence):
    """Analyze a single caption and return stats"""
    if not caption:
        return gr.update(visible=False)
    
    reviewer = CaptionReviewer()
    if concept_sentence:
        reviewer.set_trigger_words([concept_sentence])
    
    stats = reviewer.get_caption_stats(caption)
    
    stats_text = f"Words: {stats['word_count']} | "
    stats_text += f"Characters: {stats['character_count']} | "
    
    if stats['negative_word_count'] > 0:
        stats_text += f"⚠️ Negative words: {', '.join(stats['negative_words'])}"
    else:
        stats_text += "✓ No negative words"
    
    if stats['has_trigger_word']:
        stats_text += " | ✓ Has trigger word"
    else:
        stats_text += " | ⚠️ Missing trigger word"
    
    return gr.update(value=stats_text, visible=True)

def refresh_publish_tab():
    loras = get_loras()
    return gr.Dropdown(label="Trained LoRAs", choices=loras)

def init_advanced():
    # if basic_args
    basic_args = {
        'pretrained_model_name_or_path',
        'clip_l',
        't5xxl',
        'ae',
        'cache_latents_to_disk',
        'save_model_as',
        'sdpa',
        'persistent_data_loader_workers',
        'max_data_loader_n_workers',
        'seed',
        'gradient_checkpointing',
        'mixed_precision',
        'save_precision',
        'network_module',
        'network_dim',
        'learning_rate',
        'cache_text_encoder_outputs',
        'cache_text_encoder_outputs_to_disk',
        'fp8_base',
        'highvram',
        'max_train_epochs',
        'save_every_n_epochs',
        'dataset_config',
        'output_dir',
        'output_name',
        'timestep_sampling',
        'discrete_flow_shift',
        'model_prediction_type',
        'guidance_scale',
        'loss_type',
        'optimizer_type',
        'optimizer_args',
        'lr_scheduler',
        'sample_prompts',
        'sample_every_n_steps',
        'max_grad_norm',
        'split_mode',
        'network_args'
    }

    # generate a UI config
    # if not in basic_args, create a simple form
    parser = train_network.setup_parser()
    flux_train_utils.add_flux_train_arguments(parser)
    args_info = {}
    for action in parser._actions:
        if action.dest != 'help':  # Skip the default help argument
            # if the dest is included in basic_args
            args_info[action.dest] = {
                "action": action.option_strings,  # Option strings like '--use_8bit_adam'
                "type": action.type,              # Type of the argument
                "help": action.help,              # Help message
                "default": action.default,        # Default value, if any
                "required": action.required       # Whether the argument is required
            }
    temp = []
    for key in args_info:
        temp.append({ 'key': key, 'action': args_info[key] })
    temp.sort(key=lambda x: x['key'])
    advanced_component_ids = []
    advanced_components = []
    for item in temp:
        key = item['key']
        action = item['action']
        if key in basic_args:
            print("")
        else:
            action_type = str(action['type'])
            component = None
            with gr.Column(min_width=300):
                if action_type == "None":
                    # radio
                    component = gr.Checkbox()
    #            elif action_type == "<class 'str'>":
    #                component = gr.Textbox()
    #            elif action_type == "<class 'int'>":
    #                component = gr.Number(precision=0)
    #            elif action_type == "<class 'float'>":
    #                component = gr.Number()
    #            elif "int_or_float" in action_type:
    #                component = gr.Number()
                else:
                    component = gr.Textbox(value="")
                if component != None:
                    component.interactive = True
                    component.elem_id = action['action'][0]
                    component.label = component.elem_id
                    component.elem_classes = ["advanced"]
                if action['help'] != None:
                    component.info = action['help']
            advanced_components.append(component)
            advanced_component_ids.append(component.elem_id)
    return advanced_components, advanced_component_ids


theme = gr.themes.Monochrome(
    text_size=gr.themes.Size(lg="18px", md="15px", sm="13px", xl="22px", xs="12px", xxl="24px", xxs="9px"),
    font=[gr.themes.GoogleFont("Source Sans Pro"), "ui-sans-serif", "system-ui", "sans-serif"],
)
css = """
@keyframes rotate {
    0% {
        transform: rotate(0deg);
    }
    100% {
        transform: rotate(360deg);
    }
}
#advanced_options .advanced:nth-child(even) { background: rgba(0,0,100,0.04) !important; }
h1{font-family: georgia; font-style: italic; font-weight: bold; font-size: 30px; letter-spacing: -1px;}
h3{margin-top: 0}
.tabitem{border: 0px}
.group_padding{}
nav{position: fixed; top: 0; left: 0; right: 0; z-index: 1000; text-align: center; padding: 10px; box-sizing: border-box; display: flex; align-items: center; backdrop-filter: blur(10px); }
nav button { background: none; color: firebrick; font-weight: bold; border: 2px solid firebrick; padding: 5px 10px; border-radius: 5px; font-size: 14px; }
nav img { height: 40px; width: 40px; border-radius: 40px; }
nav img.rotate { animation: rotate 2s linear infinite; }
.flexible { flex-grow: 1; }
.tast-details { margin: 10px 0 !important; }
.toast-wrap { bottom: var(--size-4) !important; top: auto !important; border: none !important; backdrop-filter: blur(10px); }
.toast-title, .toast-text, .toast-icon, .toast-close { color: black !important; font-size: 14px; }
.toast-body { border: none !important; }
#terminal { box-shadow: none !important; margin-bottom: 25px; background: rgba(0,0,0,0.03); }
#terminal .generating { border: none !important; }
#terminal label { position: absolute !important; }
.tabs { margin-top: 50px; }
.hidden { display: none !important; }
.codemirror-wrapper .cm-line { font-size: 12px !important; }
label { font-weight: bold !important; }
#start_training.clicked { background: silver; color: black; }

/* Progress bar styling */
.progress-bar {
    background: linear-gradient(to right, #4CAF50, #45a049);
    border-radius: 5px;
}

/* GPU status styling */
.gpu-status {
    font-family: monospace;
    background: rgba(0, 0, 0, 0.05);
    padding: 10px;
    border-radius: 5px;
    border: 1px solid rgba(0, 0, 0, 0.1);
}

/* Caption review styling */
.caption-text textarea {
    font-size: 14px;
    line-height: 1.6;
}

.caption-stats {
    font-size: 12px;
    color: #666;
    margin-top: 5px;
}

.model-info {
    font-size: 13px;
    background: rgba(0, 0, 0, 0.03);
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
}

.negative-info {
    font-size: 12px;
    color: #d32f2f;
}

/* Keyword highlighting */
.highlight-trigger {
    background-color: #ffeb3b;
    font-weight: bold;
    padding: 2px 4px;
    border-radius: 3px;
}

.highlight-negative {
    background-color: #ffcdd2;
    color: #d32f2f;
    padding: 2px 4px;
    border-radius: 3px;
}

.highlight-emotion {
    background-color: #e1bee7;
    padding: 2px 4px;
    border-radius: 3px;
}

.highlight-location {
    background-color: #c5e1a5;
    padding: 2px 4px;
    border-radius: 3px;
}

.highlight-time {
    background-color: #b3e5fc;
    padding: 2px 4px;
    border-radius: 3px;
}

.highlight-style {
    background-color: #ffe0b2;
    padding: 2px 4px;
    border-radius: 3px;
}
"""

js = """
function() {
    let autoscroll = document.querySelector("#autoscroll")
    if (window.iidxx) {
        window.clearInterval(window.iidxx);
    }
    window.iidxx = window.setInterval(function() {
        let text=document.querySelector(".codemirror-wrapper .cm-line").innerText.trim()
        let img = document.querySelector("#logo")
        if (text.length > 0) {
            autoscroll.classList.remove("hidden")
            if (autoscroll.classList.contains("on")) {
                autoscroll.textContent = "Autoscroll ON"
                window.scrollTo(0, document.body.scrollHeight, { behavior: "smooth" });
                img.classList.add("rotate")
            } else {
                autoscroll.textContent = "Autoscroll OFF"
                img.classList.remove("rotate")
            }
        }
    }, 500);
    console.log("autoscroll", autoscroll)
    autoscroll.addEventListener("click", (e) => {
        autoscroll.classList.toggle("on")
    })
    function debounce(fn, delay) {
        let timeoutId;
        return function(...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => fn(...args), delay);
        };
    }

    function handleClick() {
        console.log("refresh")
        document.querySelector("#refresh").click();
    }
    const debouncedClick = debounce(handleClick, 1000);
    document.addEventListener("input", debouncedClick);

    document.querySelector("#start_training").addEventListener("click", (e) => {
      e.target.classList.add("clicked")
      e.target.innerHTML = "Training..."
    })

}
"""

current_account = account_hf()
print(f"current_account={current_account}")

with gr.Blocks(elem_id="app", theme=theme, css=css, fill_width=True) as demo:
    with gr.Tabs() as tabs:
        with gr.TabItem("Gym"):
            output_components = []
            with gr.Row():
                gr.HTML("""<nav>
            <img id='logo' src='/file=icon.png' width='80' height='80'>
            <div class='flexible'></div>
            <button id='autoscroll' class='on hidden'></button>
        </nav>
        """)
            with gr.Row(elem_id='container'):
                with gr.Column():
                    gr.Markdown(
                        """# Step 1. LoRA Info
        <p style="margin-top:0">Configure your LoRA train settings.</p>
        """, elem_classes="group_padding")
                    lora_name = gr.Textbox(
                        label="The name of your LoRA",
                        info="This has to be a unique name",
                        placeholder="e.g.: Persian Miniature Painting style, Cat Toy",
                    )
                    concept_sentence = gr.Textbox(
                        elem_id="--concept_sentence",
                        label="Trigger word/sentence",
                        info="Trigger word or sentence to be used",
                        placeholder="uncommon word like p3rs0n or trtcrd, or sentence like 'in the style of CNSTLL'",
                        interactive=True,
                    )
                    model_names = list(models.keys())
                    print(f"model_names={model_names}")
                    base_model = gr.Dropdown(label="Base model (edit the models.yaml file to add more to this list)", choices=model_names, value=model_names[0])
                    vram = gr.Radio(["20G", "16G", "12G" ], value="20G", label="VRAM", interactive=True)
                    num_repeats = gr.Number(value=10, precision=0, label="Repeat trains per image", interactive=True)
                    max_train_epochs = gr.Number(label="Max Train Epochs", value=16, interactive=True)
                    total_steps = gr.Number(0, interactive=False, label="Expected training steps")
                    sample_prompts = gr.Textbox("", lines=5, label="Sample Image Prompts (Separate with new lines)", interactive=True)
                    sample_every_n_steps = gr.Number(0, precision=0, label="Sample Image Every N Steps", interactive=True)
                    resolution = gr.Number(value=512, precision=0, label="Resize dataset images")
                with gr.Column():
                    gr.Markdown(
                        """# Step 2. Dataset
        <p style="margin-top:0">Make sure the captions include the trigger word.</p>
        """, elem_classes="group_padding")
                    with gr.Group():
                        images = gr.File(
                            file_types=["image", ".txt"],
                            label="Upload your images",
                            #info="If you want, you can also manually upload caption files that match the image names (example: img0.png => img0.txt)",
                            file_count="multiple",
                            interactive=True,
                            visible=True,
                            scale=1,
                        )
                    with gr.Group(visible=False) as captioning_area:
                        # Caption model selection
                        with gr.Row():
                            caption_model_manager = CaptionModelManager()
                            model_choices = list(caption_model_manager.MODELS.keys())
                            caption_model = gr.Dropdown(
                                label="Caption Model",
                                choices=model_choices,
                                value="florence2",
                                interactive=True
                            )
                            model_info = gr.Markdown("", elem_classes="model-info")
                        
                        do_captioning = gr.Button("Generate AI Captions")
                        
                        # Caption review tools
                        with gr.Accordion("Caption Review Tools", open=False) as review_tools:
                            with gr.Row():
                                # Find and replace
                                find_text = gr.Textbox(label="Find", placeholder="Text to find", scale=3)
                                replace_text = gr.Textbox(label="Replace with", placeholder="Replacement text", scale=3)
                                case_sensitive = gr.Checkbox(label="Case sensitive", value=False)
                                batch_replace_btn = gr.Button("Replace All", scale=1)
                            
                            with gr.Row():
                                # Negative word removal
                                remove_negative_btn = gr.Button("Remove Negative Words", scale=1)
                                negative_words_info = gr.Markdown("", elem_classes="negative-info", scale=2)
                            
                            with gr.Row():
                                # Custom negative words
                                custom_negative_words = gr.Textbox(
                                    label="Custom negative words (comma-separated)",
                                    placeholder="e.g., dark, gloomy, old",
                                    scale=3
                                )
                                add_custom_negative_btn = gr.Button("Add Custom", scale=1)
                        
                        output_components.append(captioning_area)
                        
                        # Caption display with review features
                        caption_list = []
                        for i in range(1, MAX_IMAGES + 1):
                            locals()[f"captioning_row_{i}"] = gr.Row(visible=False)
                            with locals()[f"captioning_row_{i}"]:
                                locals()[f"image_{i}"] = gr.Image(
                                    type="pil",
                                    width=111,
                                    height=111,
                                    min_width=111,
                                    interactive=False,
                                    scale=2,
                                    show_label=False,
                                    show_share_button=False,
                                    show_download_button=False,
                                )
                                with gr.Column(scale=15):
                                    locals()[f"caption_{i}"] = gr.Textbox(
                                        label=f"Caption {i}", 
                                        interactive=True,
                                        elem_classes="caption-text"
                                    )
                                    locals()[f"caption_stats_{i}"] = gr.Markdown(
                                        "", 
                                        elem_classes="caption-stats",
                                        visible=False
                                    )

                            output_components.append(locals()[f"captioning_row_{i}"])
                            output_components.append(locals()[f"image_{i}"])
                            output_components.append(locals()[f"caption_{i}"])
                            output_components.append(locals()[f"caption_stats_{i}"])
                            caption_list.append(locals()[f"caption_{i}"])
                with gr.Column():
                    gr.Markdown(
                        """# Step 3. Train
        <p style="margin-top:0">Press start to start training.</p>
        """, elem_classes="group_padding")
                    refresh = gr.Button("Refresh", elem_id="refresh", visible=False)
                    start = gr.Button("Start training", visible=False, elem_id="start_training")
                    output_components.append(start)
                    
                    # Progress monitoring components
                    with gr.Row():
                        progress_bar = gr.Slider(minimum=0, maximum=100, value=0, label="Training Progress", interactive=False)
                    with gr.Row():
                        gpu_status = gr.Textbox(label="GPU Status", value="Waiting to start...", interactive=False)
                    
                    train_script = gr.Textbox(label="Train script", max_lines=100, interactive=True)
                    train_config = gr.Textbox(label="Train config", max_lines=100, interactive=True)
            with gr.Accordion("Advanced options", elem_id='advanced_options', open=False):
                with gr.Row():
                    with gr.Column(min_width=300):
                        seed = gr.Number(label="--seed", info="Seed", value=42, interactive=True)
                    with gr.Column(min_width=300):
                        workers = gr.Number(label="--max_data_loader_n_workers", info="Number of Workers", value=2, interactive=True)
                    with gr.Column(min_width=300):
                        learning_rate = gr.Textbox(label="--learning_rate", info="Learning Rate", value="8e-4", interactive=True)
                    with gr.Column(min_width=300):
                        save_every_n_epochs = gr.Number(label="--save_every_n_epochs", info="Save every N epochs", value=4, interactive=True)
                    with gr.Column(min_width=300):
                        guidance_scale = gr.Number(label="--guidance_scale", info="Guidance Scale", value=1.0, interactive=True)
                    with gr.Column(min_width=300):
                        timestep_sampling = gr.Textbox(label="--timestep_sampling", info="Timestep Sampling", value="shift", interactive=True)
                    with gr.Column(min_width=300):
                        network_dim = gr.Number(label="--network_dim", info="LoRA Rank", value=4, minimum=4, maximum=128, step=4, interactive=True)
                    advanced_components, advanced_component_ids = init_advanced()
            with gr.Row():
                terminal = LogsView(label="Train log", elem_id="terminal")
            with gr.Row():
                gallery = gr.Gallery(get_samples, inputs=[lora_name], label="Samples", every=10, columns=6)

        with gr.TabItem("Publish") as publish_tab:
            hf_token = gr.Textbox(label="Huggingface Token")
            hf_login = gr.Button("Login")
            hf_logout = gr.Button("Logout")
            with gr.Row() as row:
                gr.Markdown("**LoRA**")
                gr.Markdown("**Upload**")
            loras = get_loras()
            with gr.Row():
                lora_rows = refresh_publish_tab()
                with gr.Column():
                    with gr.Row():
                        repo_owner = gr.Textbox(label="Account", interactive=False)
                        repo_name = gr.Textbox(label="Repository Name")
                    repo_visibility = gr.Textbox(label="Repository Visibility ('public' or 'private')", value="public")
                    upload_button = gr.Button("Upload to HuggingFace")
                    upload_button.click(
                        fn=upload_hf,
                        inputs=[
                            base_model,
                            lora_rows,
                            repo_owner,
                            repo_name,
                            repo_visibility,
                            hf_token,
                        ]
                    )
            hf_login.click(fn=login_hf, inputs=[hf_token], outputs=[hf_token, hf_login, hf_logout, repo_owner])
            hf_logout.click(fn=logout_hf, outputs=[hf_token, hf_login, hf_logout, repo_owner])


    publish_tab.select(refresh_publish_tab, outputs=lora_rows)
    lora_rows.select(fn=set_repo, inputs=[lora_rows], outputs=[repo_name])

    dataset_folder = gr.State()

    listeners = [
        base_model,
        lora_name,
        resolution,
        seed,
        workers,
        concept_sentence,
        learning_rate,
        network_dim,
        max_train_epochs,
        save_every_n_epochs,
        timestep_sampling,
        guidance_scale,
        vram,
        num_repeats,
        sample_prompts,
        sample_every_n_steps,
        *advanced_components
    ]
    advanced_component_ids = [x.elem_id for x in advanced_components]
    original_advanced_component_values = [comp.value for comp in advanced_components]
    
    print(f"DEBUG: output_components has {len(output_components)} elements")
    
    images.upload(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components
    )
    images.delete(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components
    )
    images.clear(
        hide_captioning,
        outputs=[captioning_area, start]
    )
    max_train_epochs.change(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    num_repeats.change(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    images.upload(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    images.delete(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    images.clear(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    concept_sentence.change(fn=update_sample, inputs=[concept_sentence], outputs=sample_prompts)
    start.click(fn=create_dataset, inputs=[dataset_folder, resolution, images] + caption_list, outputs=dataset_folder).then(
        fn=start_training,
        inputs=[
            base_model,
            lora_name,
            train_script,
            train_config,
            sample_prompts,
        ],
        outputs=terminal,
    )
    # Caption generation with model choice
    do_captioning.click(
        fn=run_captioning, 
        inputs=[images, concept_sentence, caption_model] + caption_list, 
        outputs=caption_list
    )
    
    # Model info update
    caption_model.change(
        fn=update_model_info,
        inputs=[caption_model],
        outputs=[model_info]
    )
    
    # Batch find/replace
    batch_replace_btn.click(
        fn=batch_find_replace,
        inputs=[find_text, replace_text, case_sensitive] + caption_list,
        outputs=caption_list
    )
    
    # Remove negative words
    remove_negative_btn.click(
        fn=remove_negative_words,
        inputs=[concept_sentence, custom_negative_words] + caption_list,
        outputs=[negative_words_info] + caption_list
    )
    
    # Analyze captions on change
    for i in range(1, MAX_IMAGES + 1):
        locals()[f"caption_{i}"].change(
            fn=analyze_caption,
            inputs=[locals()[f"caption_{i}"], concept_sentence],
            outputs=[locals()[f"caption_stats_{i}"]]
        )
    demo.load(fn=loaded, js=js, outputs=[hf_token, hf_login, hf_logout, repo_owner]).then(
        fn=update_model_info,
        inputs=[caption_model],
        outputs=[model_info]
    )
    refresh.click(update, inputs=listeners, outputs=[train_script, train_config, dataset_folder])
if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    # Set Gradio temp directory for better file handling in containers
    if os.environ.get('GRADIO_TEMP_DIR') is None:
        os.environ['GRADIO_TEMP_DIR'] = os.path.join(cwd, 'temp')
        os.makedirs(os.environ['GRADIO_TEMP_DIR'], exist_ok=True)
    
    # Launch with proper configuration for RunPod
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7863, 
        debug=True, 
        show_error=True, 
        allowed_paths=[cwd, os.environ.get('GRADIO_TEMP_DIR', '/tmp')]
    )
