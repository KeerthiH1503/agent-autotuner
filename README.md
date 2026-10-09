# Agent Autotuner

Profile-guided optimizer for agent tool-calling settings. Given one open-weight
model and one toolset, it searches schema compression level x confidence signal x
escalation threshold on synthetic tasks (ground truth by construction), then
checks on held-out tasks whether the tuned settings beat pre-registered defaults.

## What it measures
Automation yield, silent errors, unsafe calls (wrong destructive call executed
without escalation), unnecessary escalations, prompt tokens, latency. Comparison:
default vs independently-tuned vs jointly-tuned, with paired-bootstrap 95% CIs.
The verdict prints NOT ESTABLISHED unless the yield CI excludes zero.

## Run
    pip install openai numpy matplotlib pyyaml
    set NVIDIA_API_KEY (PowerShell: $env:NVIDIA_API_KEY="nvapi-...")
    python autotuner.py --list-models --base-url https://integrate.api.nvidia.com/v1
    python autotuner.py --check --base-url https://integrate.api.nvidia.com/v1 --model auto
    python autotuner.py --base-url https://integrate.api.nvidia.com/v1 --model auto

`--model auto` picks a small live instruct model; pass `--model <id>` to choose.
Works with any OpenAI-compatible endpoint that returns logprobs (e.g. Ollama,
llama.cpp server) via `--base-url`. `--mock` tests the code path with a simulated
model (numbers meaningless).

## Model and dependencies
Open-weight model served through NVIDIA's OpenAI-compatible API (model id is
recorded in runs/results.json). Dependencies: openai, numpy, matplotlib, pyyaml.

## Outputs
runs/pareto.png, runs/tuned_config.yaml, runs/tuned_tools_prompt.txt, runs/results.json

## Limitations
- Small held-out set gives wide confidence intervals.
- Hosted-API latency is noisy; token counts are exact.
- Synthetic tasks may not reflect real tool use.
- Needs a server that returns token logprobs; otherwise only compression is tuned.

## Results
Pending: fill in only after a real run.
