import argparse, csv, json, os, platform, re, subprocess, sys, threading, time
from pathlib import Path
import requests
import psutil

def nvidia_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return out
    except Exception:
        return "nvidia-smi unavailable"

def main():
    ap = argparse.ArgumentParser(description="Reproducible llama.cpp FP16/Q4 benchmark")
    ap.add_argument("--model", required=True, help="Path to the .gguf model file")
    ap.add_argument("--label", required=True, help="For example FP16 or Q4_K_M")
    ap.add_argument("--server", default="llama-server.exe", help="Path/name of llama-server")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--prompts", default="data/prompts/prompts.json")
    ap.add_argument("--out", default="data/results")
    args = ap.parse_args()

    model = Path(args.model).resolve()
    if not model.exists():
        sys.exit(f"Model file not found: {model}")
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^A-Za-z0-9_-]", "_", args.label)
    log_path = outdir / f"{safe_label}_server.log"
    json_path = outdir / f"{safe_label}_results.json"
    csv_path = outdir / f"{safe_label}_results.csv"

    cmd = [args.server, "-m", str(model), "--host", "127.0.0.1",
           "--port", str(args.port), "-c", str(args.ctx),
           "-t", str(args.threads), "-ngl", "0"]
    print("Starting CPU-only llama.cpp server:", " ".join(cmd))
    print("GPU layers set to 0 so both runs use the same CPU execution path.")
    log_file = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    ps_proc = psutil.Process(proc.pid)
    peak_rss = 0
    stop_sample = False

    def sample_memory():
        nonlocal peak_rss
        while not stop_sample:
            try:
                peak_rss = max(peak_rss, ps_proc.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.2)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    url = f"http://127.0.0.1:{args.port}"
    try:
        ready = False
        for _ in range(180):
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited. Check {log_path}")
            try:
                r = requests.get(url + "/health", timeout=2)
                if r.status_code == 200:
                    ready = True
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        if not ready:
            raise RuntimeError(f"Server did not become ready. Check {log_path}")

        rows = []
        for item in prompts:
            prompt = ("<|im_start|>system\nYou are a helpful assistant. Answer accurately and clearly."
                      "<|im_end|>\n<|im_start|>user\n" + item["prompt"] +
                      "<|im_end|>\n<|im_start|>assistant\n")
            payload = {"prompt": prompt, "n_predict": args.tokens, "temperature": 0,
                       "seed": 42, "cache_prompt": False, "stop": ["<|im_end|>"]}
            t0 = time.perf_counter()
            resp = requests.post(url + "/completion", json=payload, timeout=600)
            elapsed = time.perf_counter() - t0
            resp.raise_for_status()
            data = resp.json()
            timings = data.get("timings", {})
            row = {
                "id": item["id"], "prompt": item["prompt"],
                "output": data.get("content", "").strip(),
                "prompt_tokens": timings.get("prompt_n"),
                "generated_tokens": timings.get("predicted_n"),
                "generation_tokens_per_second": timings.get("predicted_per_second"),
                "request_elapsed_seconds": round(elapsed, 3),
                "finish_reason": data.get("stop_type")
            }
            rows.append(row)
            rate = row["generation_tokens_per_second"]
            print(f'{item["id"]}: {rate if rate is not None else "N/A"} tokens/s; '
                  f'{row["generated_tokens"]} generated tokens')
        time.sleep(1)
        peak_rss = max(peak_rss, ps_proc.memory_info().rss if ps_proc.is_running() else 0)
        result = {
            "label": args.label, "model_path": str(model), "model_size_bytes": model.stat().st_size,
            "model_size_GB_decimal": round(model.stat().st_size / 1e9, 3),
            "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                         "logical_cpu_count": psutil.cpu_count(), "ram_total_GB": round(psutil.virtual_memory().total/1e9, 2),
                         "nvidia_smi_before_run": nvidia_info()},
            "runtime": {"llama_server_command": cmd, "gpu_layers": 0, "threads": args.threads,
                        "context_size": args.ctx, "max_new_tokens": args.tokens},
            "peak_server_process_rss_GB": round(peak_rss/1e9, 3),
            "generation_tokens_per_second_mean": round(
                sum(x["generation_tokens_per_second"] for x in rows if x["generation_tokens_per_second"] is not None) /
                max(1, sum(1 for x in rows if x["generation_tokens_per_second"] is not None)), 3),
            "results": rows
        }
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {json_path} and {csv_path}")
        print(f"Peak server RSS: {result['peak_server_process_rss_GB']} GB")
        print(f"Mean generation speed: {result['generation_tokens_per_second_mean']} tokens/s")
    finally:
        stop_sample = True
        try:
            requests.post(url + "/slots", timeout=2)
        except Exception:
            pass
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill()
        log_file.close()

if __name__ == "__main__":
    main()
