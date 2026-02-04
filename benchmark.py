import requests
import json
import time
import base64
import subprocess
import threading
import pandas as pd
import glob
import os
import statistics
from datetime import datetime

# --- Configuration ---
CONFIG = {
    "ITERATIONS": 5,          # How many times to run each test for averaging
    "WARMUP": True,           # Run once before measuring to load weights
    "COOLDOWN": 2,            # Seconds to sleep between runs
    "IMAGE_DIR": "test_images", # Folder containing images (jpg/png)
    
    # Define your prompts (Label: Prompt)
    "TEXT_PROMPTS": {
        "Short (Physics)": "Explain the physics of why the sky appears blue in 3 sentences.",
        "Long (Creative)": "Write a 200-word short story about a robot discovering a flower on Mars."
    },
    
    # Generic prompt for every image found in IMAGE_DIR
    "VISION_PROMPT": "Describe this image in extreme detail, focusing on lighting and object placement.",
    
    "ENGINES": {
        "Ollama": {
            "url": "http://localhost:11700/v1/chat/completions",
            "model": "qwen3-vl:8b-instruct",
            "headers": {"Content-Type": "application/json"}
        },
        "llama.cpp": {
            "url": "http://localhost:11701/v1/chat/completions",
            "model": "qwen3-vl:8b",
            "headers": {"Content-Type": "application/json", "Authorization": "Bearer no-key"}
        }
    }
}

class VRAMMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.monitoring = False
        self.peak_vram = 0
        self.lock = threading.Lock()

    def get_gpu_memory(self):
        try:
            # Queries nvidia-smi for used memory in MiB
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            return int(result.stdout.strip().split('\n')[0])
        except (ValueError, FileNotFoundError, IndexError):
            return 0

    def start(self):
        self.monitoring = True
        self.peak_vram = self.get_gpu_memory() # Start with current usage
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

def api_call(url, model, headers, prompt, image_path=None):
    """
    Performs the actual API request and yields metrics for a single run.
    """
    monitor = VRAMMonitor()
    
    messages = [{"role": "user", "content": []}]
    
    if image_path:
        base64_image = encode_image(image_path)
        messages[0]["content"].append({"type": "text", "text": prompt})
        messages[0]["content"].append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
    else:
        messages[0]["content"] = prompt

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.1 # Low temp for more consistent benchmark results
    }

    start_time = time.time()
    monitor.start()
    
    first_chunk_time = None
    ttft = 0
    token_count = 0
    full_content = ""
    
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=300)
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8').strip()
                if decoded_line.startswith("data: ") and decoded_line != "data: [DONE]":
                    try:
                        data = json.loads(decoded_line[6:])
                        
                        # Capture TTFT
                        if first_chunk_time is None and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            if delta.get('content'):
                                first_chunk_time = time.time()
                                ttft = first_chunk_time - start_time
                                full_content += delta.get('content')
                        
                        # Capture subsequent content
                        elif len(data['choices']) > 0:
                            content = data['choices'][0].get('delta', {}).get('content')
                            if content:
                                full_content += content

                        # Capture final usage stats if available
                        if data.get('usage'):
                            token_count = data['usage']['completion_tokens']
                            
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"    [!] Error during API call: {e}")
        monitor.stop()
        return None

    end_time = time.time()
    peak_vram = monitor.stop()
    total_time = end_time - start_time

    # Fallback token counting
    if token_count == 0 and full_content:
        token_count = len(full_content) / 4 

    # Calculate TPS
    gen_time = total_time - ttft
    tps = token_count / gen_time if gen_time > 0 else 0

    return {
        "ttft": ttft,
        "total_time": total_time,
        "tps": tps,
        "peak_vram": peak_vram,
        "tokens": token_count
    }

def run_benchmark_suite():
    raw_results = []
    
    # 1. Setup Image List
    image_files = []
    if os.path.exists(CONFIG["IMAGE_DIR"]):
        types = ('*.jpg', '*.jpeg', '*.png')
        for t in types:
            image_files.extend(glob.glob(os.path.join(CONFIG["IMAGE_DIR"], t)))
    else:
        print(f"[!] Warning: directory '{CONFIG['IMAGE_DIR']}' not found. Skipping vision tests.")

    # 2. Iterate Engines
    for engine_name, config in CONFIG["ENGINES"].items():
        print(f"\nSTARTING ENGINE: {engine_name}")
        
        # --- Text Tests ---
        for test_name, prompt in CONFIG["TEXT_PROMPTS"].items():
            print(f"\n  Running Text Test: {test_name}")
            
            # Warmup
            if CONFIG["WARMUP"]:
                print("    [Warmup] Loading model...", end="\r")
                api_call(config['url'], config['model'], config['headers'], prompt)
                print("    [Warmup] Done.           ")

            # Iterations
            for i in range(CONFIG["ITERATIONS"]):
                print(f"    Iteration {i+1}/{CONFIG['ITERATIONS']}...", end="\r")
                metrics = api_call(config['url'], config['model'], config['headers'], prompt)
                
                if metrics:
                    metrics.update({
                        "Engine": engine_name,
                        "Type": "Text",
                        "Input": test_name
                    })
                    raw_results.append(metrics)
                
                time.sleep(CONFIG["COOLDOWN"])

        # --- Vision Tests ---
        if image_files:
            print(f"\n  Running Vision Tests on {len(image_files)} images")
            
            # Warmup with first image
            if CONFIG["WARMUP"]:
                print("    [Warmup] Loading vision encoder...", end="\r")
                api_call(config['url'], config['model'], config['headers'], 
                        CONFIG["VISION_PROMPT"], image_files[0])
                print("    [Warmup] Done.                   ")

            for img_path in image_files:
                img_name = os.path.basename(img_path)
                
                for i in range(CONFIG["ITERATIONS"]):
                    print(f"    [{img_name}] Iteration {i+1}/{CONFIG['ITERATIONS']}...", end="\r")
                    metrics = api_call(config['url'], config['model'], config['headers'], 
                                     CONFIG["VISION_PROMPT"], img_path)
                    
                    if metrics:
                        metrics.update({
                            "Engine": engine_name,
                            "Type": "Vision",
                            "Input": img_name
                        })
                        raw_results.append(metrics)
                    
                    time.sleep(CONFIG["COOLDOWN"])

    return raw_results

def process_results(raw_data):
    if not raw_data:
        print("No results collected.")
        return

    df = pd.DataFrame(raw_data)
    
    # Calculate Aggregates (Mean & StdDev)
    summary = df.groupby(['Engine', 'Type', 'Input']).agg({
        'tps': ['mean', 'std'],
        'ttft': ['mean', 'std'],
        'peak_vram': 'max',
        'tokens': 'mean'
    }).reset_index()

    # Flatten MultiIndex columns
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    
    # Rename for readability
    summary = summary.rename(columns={
        'tps_mean': 'Avg TPS',
        'tps_std': 'TPS (σ)',
        'ttft_mean': 'Avg TTFT (s)',
        'ttft_std': 'TTFT (σ)',
        'peak_vram_max': 'Peak VRAM (MiB)',
        'tokens_mean': 'Avg Tokens'
    })

    # Formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.2f}'.format)

    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(summary.to_markdown(index=False))
    
    # Save full raw data and summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"benchmark_raw_{timestamp}.csv", index=False)
    summary.to_csv(f"benchmark_summary_{timestamp}.csv", index=False)
    print(f"\nFiles saved: benchmark_raw_{timestamp}.csv & benchmark_summary_{timestamp}.csv")

if __name__ == "__main__":
    # Ensure image directory exists or create it
    if not os.path.exists(CONFIG["IMAGE_DIR"]):
        os.makedirs(CONFIG["IMAGE_DIR"])
        print(f"[!] Created '{CONFIG['IMAGE_DIR']}'. Please put images there and run again.")
    else:
        results = run_benchmark_suite()
        process_results(results)