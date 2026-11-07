#!/usr/bin/env python3
"""getcompute — the getgpu idea, rebuilt for Joshua's Mint PC.

What getgpu does on a shared NVIDIA lab machine, this does on a
single-user CPU-only machine (Ryzen 3 3250U, 8 GB RAM, Vega 3 shared
graphics, HDD, Linux Mint):

  * rshin/getgpu side: find free compute, claim it, set the env, run
    your program on it — `getcompute -- <command>`
  * DongShan03/GetGPU (gg.py) side: hold/reserve the resource so a
    second run can't steal it mid-job, and retry until it's free
    instead of crashing — `--wait`, `--retry-interval`

Why NOT a straight copy of either repo:
  * Both are NVIDIA/CUDA only. rshin/getgpu reads NVML for "GPUs with
    no compute processes" and sets CUDA_VISIBLE_DEVICES. gg.py grabs
    VRAM with PyTorch CUDA tensors (75% by default) and runs
    torch.distributed.
  * This PC has no NVIDIA card, no CUDA, no discrete VRAM. Vega 3
    shares system RAM and has no ROCm support, so Ollama/llama.cpp
    run on CPU here. Copying either script would find zero GPUs and
    do nothing. The scarce resource here is RAM + CPU threads, so
    that's what this claims.

What it claims instead:
  * Free RAM (from /proc/meminfo MemAvailable) — enough for the model
    file + KV cache + ~1.5 GB left for Mint itself.
  * CPU headroom (from /proc/loadavg vs thread count).
  * The Ollama slot — one model at a time on 8 GB; a second agent run
    while one model is loaded is what pushes the box into swap.
  * A lock dir (/tmp/getcompute.lock) so two `getcompute` runs can't
    both claim the same RAM at once. Stale locks (dead PID) are
    cleared; live ones make a new run wait or exit.

Models it knows (Joshua's roster, 2026-11-05 — all pass his 3
constants: 2B-class >= Bonsai 2 2B, fits his PC, abliterated):
  * minicpm5 — maternion/minicpm5-heretic:2b, Q8_0, ~2.7 GB (driver)
  * qwen-dolphin — dolphinlabs/qwen3.5-uncensored, ~2.1 GB (reviewer)
  * bonsai2b — eriktonon/bonsai-2-uncensored:2b, ~1.0 GB (fast seat)
  * bonsai4b — bonsai-2:4b, ~1.5 GB est. (the original stock default,
    kept for governor/control runs; stock, not abliterated)

Usage:
  python3 getcompute.py --check
      Just report: free RAM, load, Ollama status, which models fit.
  python3 getcompute.py --model minicpm5 -- ollama run maternion/minicpm5-heretic:2b "hello"
      Claim, set OLLAMA_MODEL / OLLAMA_NUM_PARALLEL=1 /
      OMP_NUM_THREADS=4, run the command, release on exit.
  python3 getcompute.py --model bonsai2b --wait 600 -- ollama run eriktonon/bonsai-2-uncensored:2b "hello"
      Wait up to 600s (retry every 30s) for enough free RAM, then run.
  python3 getcompute.py --model minicpm5 --reserve-gb 3.5 -- python3 -m forge.main
      Same, but demand 3.5 GB free for the model instead of the default.

Stdlib only — no pip install, no PyTorch, no nvidia-smi.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

LOCK_DIR = "/tmp/getcompute.lock"
OS_RESERVE_GB = 1.5          # leave this much RAM for Mint itself
CTX_GB = 0.5                 # rough KV-cache headroom at 4K-8K context
RETRY_DEFAULT = 30

MODELS = {
    "minicpm5": {
        "ollama": "maternion/minicpm5-heretic:2b",
        "gb": 2.7,
        "note": "The driver. Best coder of the roster; Q8_0.",
    },
    "qwen-dolphin": {
        "ollama": "dolphinlabs/qwen3.5-uncensored:latest",
        "gb": 2.1,
        "note": "The reviewer. Full-weight Qwen3.5-2B, abliterated; Q8_0.",
    },
    "bonsai2b": {
        "ollama": "eriktonon/bonsai-2-uncensored:2b",
        "gb": 1.0,
        "note": "The fast seat. Bonsai 2 2B itself, abliterated; lightest.",
    },
    "bonsai4b": {
        "ollama": "bonsai-2:4b",
        "gb": 1.5,
        "note": "Original stock default (estimate). Governor/control runs only.",
    },
}


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def mem_total_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def load_1min():
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def ollama_status():
    """Return (running: bool, loaded_models: list[str])."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=2) as r:
            data = json.load(r)
        return True, [m.get("name", "?") for m in data.get("models", [])]
    except Exception:
        return False, []


def lock_is_live():
    pid_file = os.path.join(LOCK_DIR, "pid")
    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def acquire_lock():
    """Atomic claim (mkdir). Returns True if we hold it."""
    try:
        os.mkdir(LOCK_DIR)
    except FileExistsError:
        if lock_is_live():
            return False
        # Stale lock from a dead process — clear it, then claim.
        subprocess.run(["rm", "-rf", LOCK_DIR], check=False)
        try:
            os.mkdir(LOCK_DIR)
        except FileExistsError:
            return False
    with open(os.path.join(LOCK_DIR, "pid"), "w") as f:
        f.write(str(os.getpid()))
    return True


def release_lock():
    subprocess.run(["rm", "-rf", LOCK_DIR], check=False)


def check(model_key, reserve_gb=None):
    info = MODELS[model_key]
    need = reserve_gb if reserve_gb else info["gb"] + CTX_GB
    avail = mem_available_gb()
    total = mem_total_gb()
    load = load_1min()
    threads = os.cpu_count() or 4
    running, loaded = ollama_status()
    fits = (avail - OS_RESERVE_GB) >= need
    busy = load > threads
    return {
        "model": model_key,
        "ollama_tag": info["ollama"],
        "need_gb": round(need, 2),
        "avail_gb": round(avail, 2),
        "total_gb": round(total, 2),
        "load_1min": load,
        "threads": threads,
        "ollama_running": running,
        "ollama_loaded": loaded,
        "fits": fits and not busy,
        "fits_ram": fits,
        "cpu_busy": busy,
    }


def report(status):
    print(f"RAM: {status['avail_gb']} GB free of {status['total_gb']} GB "
          f"| need {status['need_gb']} GB for {status['model']} "
          f"({status['ollama_tag']}) + {OS_RESERVE_GB} GB left for Mint")
    print(f"CPU: load {status['load_1min']} on {status['threads']} threads"
          f"{' — BUSY, wait' if status['cpu_busy'] else ' — OK'}")
    if status["ollama_running"]:
        loaded = ", ".join(status["ollama_loaded"]) or "none loaded"
        print(f"Ollama: running, loaded: {loaded}")
    else:
        print("Ollama: not running (start it with `ollama serve`, or a first `ollama run` will start it)")
    print("VERDICT: " + ("fits — safe to run" if status["fits"]
          else "does NOT fit right now — close browser tabs / wait for the other run to finish"))
    return status["fits"]


def main():
    p = argparse.ArgumentParser(description="getgpu equivalent for a CPU-only Mint PC")
    p.add_argument("--model", default="minicpm5", choices=list(MODELS),
                   help="which model to size the claim for (default: minicpm5)")
    p.add_argument("--check", action="store_true", help="report only, claim nothing")
    p.add_argument("--wait", type=int, default=0,
                   help="seconds to keep retrying until the machine is free (0 = try once)")
    p.add_argument("--retry-interval", type=int, default=RETRY_DEFAULT,
                   help="seconds between retries (default 30, like getgpu's --startup-time)")
    p.add_argument("--reserve-gb", type=float, default=None,
                   help="GB to demand free for the model (default: model size + 0.5 GB context)")
    p.add_argument("command", nargs=argparse.REMAINDER,
                   help="command to run after `--`, with the claim held")
    args = p.parse_args()
    cmd = [c for c in args.command if c != "--"]

    if args.check or not cmd:
        ok = report(check(args.model, args.reserve_gb))
        return 0 if ok else 1

    deadline = time.time() + args.wait if args.wait else time.time()
    while True:
        status = check(args.model, args.reserve_gb)
        if status["fits"]:
            if not acquire_lock():
                print("Another getcompute run holds the claim — waiting…")
            else:
                break
        else:
            report(status)
        if time.time() >= deadline:
            print(f"Gave up after waiting {args.wait}s — machine still busy. "
                  "Close browser tabs or wait for the other Ollama run to finish, then retry.")
            return 1
        print(f"Retrying in {args.retry_interval}s… (Ctrl-C to stop)")
        try:
            time.sleep(args.retry_interval)
        except KeyboardInterrupt:
            return 1

    print(f"Claimed: {args.model} ({MODELS[args.model]['ollama']}) — "
          f"{status['avail_gb']} GB free, load {status['load_1min']}. Running your command…")
    env = dict(os.environ)
    env["OLLAMA_MODEL"] = MODELS[args.model]["ollama"]
    env["OLLAMA_NUM_PARALLEL"] = "1"   # one model at a time on 8 GB
    env["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)
    try:
        rc = subprocess.run(cmd, env=env).returncode
    finally:
        release_lock()
        print("Claim released.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
