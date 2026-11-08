# getcompute

**RAM instead of VRAM.** A getgpu equivalent for CPU-only machines.

## What it is

`getcompute` is a small Python tool that does for a CPU-only home PC what the getgpu scripts do for shared NVIDIA lab machines: before a local AI model runs, it checks the machine actually has room — free RAM for the model plus its context plus ~1.5 GB left for the OS, CPU headroom, and the Ollama slot — claims those resources, and refuses (or waits) instead of letting a run shove the machine into swap and freeze.

One file. Python standard library only — no pip install, no PyTorch, no nvidia-smi.

## Why it exists

The existing getgpu tools (rshin/getgpu, DongShan03/GetGPU) are NVIDIA/CUDA only. On an integrated-graphics machine (Ryzen 3 3250U, 8 GB RAM, Vega 3 sharing system memory, spinning HDD), there is no VRAM to grab — RAM and CPU threads are the scarce resource. getcompute claims what's actually scarce.

## What it checks

- **Free RAM** (from `/proc/meminfo`) — enough for the model file + KV cache + ~1.5 GB left for the OS itself.
- **CPU headroom** (from `/proc/loadavg` vs thread count).
- **The Ollama slot** — one model at a time on 8 GB; a second run while one is loaded is what pushes the box into swap.
- **A lock** (`/tmp/getcompute.lock`) so two `getcompute` runs can't both claim the same RAM at once. Stale locks from dead processes are cleared; live ones make a new run wait or exit.

## Usage

Just report — free RAM, load, Ollama status, whether the model fits:

```sh
python3 getcompute.py --check
```

Claim the machine, then run a command with the claim held (sets `OLLAMA_MODEL`, `OLLAMA_NUM_PARALLEL=1`, `OMP_NUM_THREADS` for the run, releases on exit):

```sh
python3 getcompute.py --model minicpm5 -- ollama run maternion/minicpm5-heretic:2b "hello"
```

Wait up to 600 seconds (retrying every 30s) for enough free RAM, then run:

```sh
python3 getcompute.py --model bonsai2b --wait 600 -- ollama run eriktonon/bonsai-2-uncensored:2b "hello"
```

Demand a specific amount of free RAM for whatever you're running:

```sh
python3 getcompute.py --model minicpm5 --reserve-gb 3.5 -- python3 -m forge.main
```

The model table inside the script is sized for one specific roster on one specific 8 GB machine — edit `MODELS` to match yours. The checks themselves are generic: any Linux box with `/proc` and Ollama.

## Requirements

- Linux (reads `/proc/meminfo` and `/proc/loadavg`)
- Python 3, standard library only
- Ollama, if you want the loaded-model check (it degrades fine without it)
