# Pause-and-Think: A Dataset and Benchmark for Video-Grounded Assistive Action Suggestion

This repository contains the code accompanying the paper **"Pause and Think: A Dataset and Benchmark for Video-Grounded Assistive Action Suggestion"** by Shivam Singh, Saptarshi Majumdar, Pratik Prabhanjan, Zicheng Liu, and Emad Barsoum (AMD).

> Recent Vision-Language Models (VLMs) struggle with grounded reasoning, temporal consistency, and context-aware planning in videos. We introduce **pause-and-think-T**, a reasoning-centric training dataset that encourages models to pause, reason over visual evidence, and produce concise, actionable responses. A fine-tuned compact **4B-parameter** model achieves **58.0%** accuracy on our **pause-and-think-B** benchmark — **59× fewer parameters than Qwen3-VL-235B (58.9%)** — matching GPT-5.2 on scene understanding and surpassing GPT-4o.

## Links

- **Paper (this repo):** [`pause-and-think-arxiv-amd.pdf`](./pause-and-think-arxiv-amd.pdf)
- **Fine-tuned checkpoint (Hugging Face):** [shivammmmm/pause-and-think-best-checkpoint-hf](https://huggingface.co/shivammmmm/pause-and-think-best-checkpoint-hf)
- **Training / evaluation / benchmark data:** *To be released soon on Hugging Face.*

## Repository Structure

```
.
├── benchmarking/                 # Performance + evaluation code for pause-and-think-B
│   ├── benchmark_300_files.json  # 300-sample benchmark manifest
│   ├── evaluation/               # GPT-5.1 automated evaluator (binary validity scoring)
│   │   ├── evaluation.py
│   │   └── evaluation_prompt.py
│   └── performance/              # Per-model inference scripts on the benchmark
│       ├── gpt-5.2/
│       ├── gemini-robotics-er-1.5/
│       ├── qwen-3vl-235B/
│       └── qwen-3vl-4B-finetuned_frozen_vit_and_projector_32768_250/
│
├── q3vl_llamafactory/            # Training code (fork of LLaMA-Factory)
│   ├── train_robotics/
│   │   ├── qwen3_vl/             # SFT configs WITH <thinking> supervision
│   │   └── qwen3_vl_nothink/     # SFT configs WITHOUT <thinking> supervision
│   ├── run.sh                    # Full SFT launch script
│   ├── run_lora.sh               # LoRA SFT launch script
│   └── ...                       # LLaMA-Factory source
│
└── pause-and-think-arxiv-amd.pdf
```

## What's Included

This release covers three components of the paper:

1. **Training code** (`q3vl_llamafactory/`) — based on the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework, with our reasoning-centric SFT recipes for Qwen3-VL (2B/4B) and Qwen2.5-VL (3B) in `train_robotics/`. Both thinking (`qwen3_vl/`) and non-thinking (`qwen3_vl_nothink/`) variants are provided.
2. **Benchmark inference** (`benchmarking/performance/`) — scripts to run each evaluated model (GPT-5.2, Gemini-Robotics-ER-1.5, Qwen3-VL-235B, and our fine-tuned Qwen3-VL-4B) on the 300-sample pause-and-think-B benchmark.
3. **Automated evaluation** (`benchmarking/evaluation/`) — the GPT-5.1-based evaluator that performs binary validity scoring against ground truth, as described in Section 4.1.4 of the paper.

The training, evaluation, and benchmarking **datasets** themselves will be uploaded to Hugging Face shortly — this repository will be updated with direct links once available.

## Quick Start

### 1. Run the fine-tuned model

The best checkpoint from the paper is available on Hugging Face:

```bash
huggingface-cli download shivammmmm/pause-and-think-best-checkpoint-hf
```

### 2. Run benchmark inference

Each subdirectory under `benchmarking/performance/` contains a self-contained inference script:

```bash
cd benchmarking/performance/qwen-3vl-4B-finetuned_frozen_vit_and_projector_32768_250
python inference_qwen3vl4b.py
```

Note: inference scripts currently expect API keys / model paths to be set inside the script (look for the `API_KEY` placeholder string).

### 3. Run evaluation

```bash
cd benchmarking/evaluation
python evaluation.py
```

The evaluator uses GPT-5.1 as a multimodal judge under binary validity scoring (see Section 4.1.4 of the paper).

### 4. Fine-tune your own model

Training uses LLaMA-Factory. Before launching, export your Hugging Face token:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxx
cd q3vl_llamafactory
bash run.sh        # full SFT
# or
bash run_lora.sh   # LoRA SFT
```

The config used inside `run.sh` / `run_lora.sh` can be swapped to any YAML under `train_robotics/qwen3_vl/` or `train_robotics/qwen3_vl_nothink/`.

## Headline Results

On **pause-and-think-B** (300 samples, binary validity, averaged over 3 runs):

| Model | Params | Overall | Scene | Goal |
|---|---:|---:|---:|---:|
| GPT-5.2 (closed) | — | 64.24 | 55.13 | 86.52 |
| Qwen3-VL-235B-Instruct | 235B | 58.89 | 53.03 | 72.26 |
| **Ours: Qwen3-VL-4B + pause-and-think-T (T)** | **4B** | **58.00** | **55.02** | **64.84** |
| GPT-4o (closed) | — | 50.33 | 42.58 | 68.13 |
| Qwen3-VL-4B baseline | 4B | 49.00 | 46.86 | 53.76 |

Our 4B model is **Pareto-optimal among open-weight models** and generalises out-of-distribution on **EgoThink** and **TempCompass** (see Table II in the paper).

## Citation

If you use this code, the dataset, or the released checkpoint, please cite:

```bibtex
@article{singh2026pauseandthink,
  title   = {Pause and Think: A Dataset and Benchmark for Video-Grounded Assistive Action Suggestion},
  author  = {Singh, Shivam and Majumdar, Saptarshi and Prabhanjan, Pratik and Liu, Zicheng and Barsoum, Emad},
  year    = {2026},
  note    = {Work done during internship at AMD}
}
```

## Acknowledgements

- Training infrastructure built on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
- Inference accelerated with [vLLM](https://github.com/vllm-project/vllm).
- Training data sourced from [Epic-Kitchens](https://epic-kitchens.github.io/), [Assembly101](https://assembly-101.github.io/), and [Ego4D](https://ego4d-data.org/).
- Fine-tuning performed on **8× AMD Instinct™ MI325** GPUs; the deployed model was further optimised via the **AMD Ryzen™ AI** stack for edge inference on a Strix Halo Ryzen AI chip.

## License

The training code in `q3vl_llamafactory/` is a fork of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and is distributed under its original license (see `q3vl_llamafactory/LICENSE`). The rest of this repository follows the same license unless stated otherwise.
