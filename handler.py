#!/usr/bin/env python3
import runpod
import os
import logging
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globálne premenomnená na cacheovanie modelu
llm = None

def load_model():
    global llm
    if llm is None:
        logger.info("Loading Gemma 3-27B model with vLLM...")
        model_name = "mlabonne/gemma-3-27b-it-abliterated"
        
        # vLLM - najrýchlejší runtime bez kvantizácie
        llm = LLM(
            model=model_name,
            dtype="float16",  # float16 na šetrenie VRAM, bez stráty kvality
            gpu_memory_utilization=0.95,  # Maximálne využitie GPU
            max_model_len=2048,  # Max token dĺžka
            trust_remote_code=True,
        )
        logger.info("Model loaded successfully")
    return llm

def handler(event):
    try:
        load_model()
        
        prompt = event["input"].get("prompt", "")
        max_tokens = event["input"].get("max_tokens", 512)
        temperature = event["input"].get("temperature", 0.7)
        top_p = event["input"].get("top_p", 0.95)
        
        if not prompt:
            return {
                "output": "",
                "error": "Prompt is required",
                "status": "error"
            }
        
        # Nastavenie sampling parametrov
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        # Inference
        logger.info(f"Processing prompt: {prompt[:50]}...")
        outputs = llm.generate(
            [prompt],
            sampling_params
        )
        
        # Výsledok
        response = outputs[0].outputs[0].text
        
        return {
            "output": response,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}", exc_info=True)
        return {
            "output": "",
            "error": str(e),
            "status": "error"
        }

# Spustenie Runpod serverless
if __name__ == "__main__":
    logger.info("Starting RunPod Serverless handler...")
    runpod.serverless.start({"handler": handler})
