import requests
import json
import time
import base64
import subprocess
import threading
import pandas as pd
from datetime import datetime

# --- Configuration ---
OLLAMA_URL = "http://localhost:11700/v1/chat/completions"
LLAMACPP_URL = "http://localhost:11701/v1/chat/completions"

# Model names as defined in your systems
MODEL_OLLAMA = "qwen3-vl:8b-instruct"
MODEL_LLAMACPP = "qwen3-vl:8b" # Name is arbitrary, it doesnt need it

# Test Prompts
TEXT_PROMPT = "Explain the physics of why the sky appears blue in 3 sentences."
IMAGE_PROMPT = "Describe what is in this image in detail."
TEST_IMAGE_PATH = "test_image.jpg" # Ensure this exist

# Headers
HEADERS_OLLAMA = {"Content-Type": "application/json"}
HEADERS_LLAMACPP = {"Content-Type": "application/json", "Authorization": "Bearer no-key"}

# WHETHER TO RESTART CONTAINER
RESTART = False

class VRAMMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.monitoring = False
        self.peak_vram = 0
        self.lock = threading.Lock()

    def get_gpu_memory(self):
        """Queries nvidia-smi for used memory in MiB."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            return int(result.stdout.strip().split('\n')[0])
        except (ValueError, FileNotFoundError):
            return 0

    def start(self):
        self.monitoring = True
        self.peak_vram = 0
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def _monitor_loop(self):
        while self.monitoring:
            current_mem = self.get_gpu_memory()
            with self.lock:
                if current_mem > self.peak_vram:
                    self.peak_vram = current_mem
            time.sleep(self.interval)

    def stop(self):
        self.monitoring = False
        self.thread.join()
        return self.peak_vram

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def purge_vram(engine_name):
    """
    Attempts to purge VRAM. 
    Note: A true VRAM purge often requires restarting the container.
    This function attempts API-level unloading where possible.
    """
    print(f"[*] Attempting to purge VRAM for {engine_name}...")
    
    if engine_name == "Ollama":
        # # Ollama allows unloading via setting keep_alive to 0
        # try:
        #     requests.post("http://localhost:11700/api/generate", 
        #                   json={"model": MODEL_OLLAMA, "keep_alive": 0})
        # except Exception as e:
        #     print(f"    Warning: Could not unload Ollama: {e}")

        # To make it fair
        if RESTART:
            subprocess.run(["docker", "restart", "benchmark-ollama"])

    elif engine_name == "llama.cpp":
        # llama.cpp server is persistent. 
        # Ideally, should restart the docker container here.
        if RESTART:
            subprocess.run(["docker", "restart", "benchmark-llamacpp"])
        else:
            print("    Note: llama.cpp keeps models loaded. For a Cold Start test, restart the container.")
    
    # Wait a moment for memory to clear
    time.sleep(2)

def run_test(engine, url, model, headers, prompt, image_path=None):
    monitor = VRAMMonitor()
    
    messages = [{"role": "user", "content": []}]
    
    if image_path:
        base64_image = encode_image(image_path)
        messages[0]["content"].append({
            "type": "text",
            "text": prompt
        })
        messages[0]["content"].append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
    else:
        messages[0]["content"] = prompt

    payload = {
        "model": model,
        "messages": messages,
        "stream": True, # Streaming is required for accurate TTFT
        "stream_options": {"include_usage": True}
    }

    print(f"--- Testing {engine} ({'Vision' if image_path else 'Text'}) ---")
    purge_vram(engine)
    
    start_time = time.time()
    monitor.start()
    
    ttft = None
    response = requests.post(url, headers=headers, json=payload, stream=True)
    
    full_content = ""
    token_count = 0
    
    first_chunk_time = None
    
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8').strip()
            if decoded_line.startswith("data: ") and decoded_line != "data: [DONE]":
                json_str = decoded_line[6:]
                try:
                    data = json.loads(json_str)
                    
                    # Calculate TTFT on first content chunk
                    if first_chunk_time is None and len(data['choices']) > 0:
                        delta = data['choices'][0].get('delta', {})
                        if 'content' in delta and delta['content']:
                            first_chunk_time = time.time()
                            ttft = first_chunk_time - start_time
                            
                    # Accumulate content
                    if len(data['choices']) > 0:
                        delta = data['choices'][0].get('delta', {})

                        # We safely get content and check if it is not None/Empty
                        chunk_content = delta.get('content')
                        
                        if chunk_content:
                            # 1. Calculate TTFT on the very first valid text chunk
                            if first_chunk_time is None:
                                first_chunk_time = time.time()
                                ttft = first_chunk_time - start_time
                            
                            # 2. Append content
                            full_content += chunk_content
                             
                    # Try to get exact usage if provided in the last chunk
                    if data.get('usage'):
                        token_count = data['usage']['completion_tokens']
                        
                except json.JSONDecodeError:
                    continue

    end_time = time.time()
    peak_vram = monitor.stop()
    
    total_time = end_time - start_time
    
    # Fallback token counting if usage not returned
    if token_count == 0:
        # Rough estimation: 1 token ~= 4 chars
        token_count = len(full_content) / 4 

    # Metrics
    tps = token_count / (total_time - (ttft if ttft else 0))
    
    return {
        "Engine": engine,
        "Task": "Image + Text" if image_path else "Text Only",
        "TTFT (s)": round(ttft, 4) if ttft else 0,
        "Total Time (s)": round(total_time, 2),
        "Tokens/Sec": round(tps, 2),
        "Peak VRAM (MiB)": peak_vram,
        "Output Length": len(full_content)
    }

def main():
    results = []
    
    # --- Text Tests ---
    results.append(run_test("Ollama", OLLAMA_URL, MODEL_OLLAMA, HEADERS_OLLAMA, TEXT_PROMPT))
    results.append(run_test("llama.cpp", LLAMACPP_URL, MODEL_LLAMACPP, HEADERS_LLAMACPP, TEXT_PROMPT))
    
    # --- Vision Tests ---
    # Ensure you have a 'test_image.jpg' in the directory
    try:
        results.append(run_test("Ollama", OLLAMA_URL, MODEL_OLLAMA, HEADERS_OLLAMA, IMAGE_PROMPT, TEST_IMAGE_PATH))
        results.append(run_test("llama.cpp", LLAMACPP_URL, MODEL_LLAMACPP, HEADERS_LLAMACPP, IMAGE_PROMPT, TEST_IMAGE_PATH))
    except FileNotFoundError:
        print("\n[!] Image test skipped: 'test_image.jpg' not found.")

    # --- Reporting ---
    df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("INFERENCE BENCHMARK RESULTS")
    print("="*50)
    print(df.to_markdown(index=False))
    
    # Export for stakeholders
    df.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to 'benchmark_results.csv'")

if __name__ == "__main__":
    main()