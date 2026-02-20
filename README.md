# SKaiNET-nanoGPT

A faithful Kotlin port of [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) built on the [SKaiNET](https://github.com/SKaiNET-developers/SKaiNET) deep learning framework. Loads real GPT-2 weights from HuggingFace and generates text — entirely on the JVM, with zero Python dependencies.

## What is this?

Karpathy's `model.py` is a clean, single-file GPT-2 implementation in ~330 lines of Python/PyTorch. This project ports it to Kotlin using SKaiNET's tensor primitives and DSLs, producing a standalone GPT-2 inference engine that runs on any JVM 21+ platform.

The port is structurally faithful to the original — the same pre-norm transformer architecture, the same weight-tying scheme, the same causal attention with additive masking — but expressed through SKaiNET's type-safe Kotlin APIs instead of PyTorch.

## Why SKaiNET?

SKaiNET is a Kotlin Multiplatform deep learning framework designed for edge and on-device AI. Using it instead of PyTorch gives you:

### Run anywhere the JVM runs
No CUDA, no Python, no native library installation. The CPU backend uses JDK 21's Vector API for SIMD-accelerated tensor operations. The same code runs on macOS, Linux, Windows — and with SKaiNET's multiplatform targets, the core model logic can compile to Android, iOS, JS, WASM, and native binaries.

### Type-safe tensor programming
SKaiNET tensors are parameterized as `Tensor<T, V>` where `T` is the precision type (`FP32`, `FP16`, `Int8`) and `V` is the JVM value type. Shape mismatches and type errors are caught at compile time, not at runtime.

### Expressive DSLs for deep learning
The port showcases several SKaiNET DSLs that make tensor code readable and concise:

- **Slicing DSL** — zero-copy tensor views with a builder syntax:
  ```kotlin
  // Split combined QKV projection into Q, K, V
  val q = qkv.sliceView {
      segment { all() }              // all positions
      segment { range(0, nEmbd) }    // first nEmbd features
  }
  ```

- **LayerNormalization modules** — pre-initialized with loaded weights:
  ```kotlin
  val ln = LayerNormalization<T, Float>(
      normalizedShape = intArrayOf(nEmbd),
      initGamma = weights.lnFWeight,
      initBeta = weights.lnFBias
  )
  ```

- **Tensor extension functions** — idiomatic operator chaining:
  ```kotlin
  val fc = (input.matmul(tw.mlpFcWT) + layer.mlpFcBias).gelu()
  ```

### SafeTensors weight loading
SKaiNET's I/O module reads HuggingFace SafeTensors files directly, with automatic FP16/BF16 dequantization. No model conversion step needed — download from HuggingFace and run.

### Lightweight and embeddable
The compiled fat JAR is self-contained. No framework servers, no model registries, no GPU drivers. Suitable for embedding GPT-2 inference into any JVM application — backend services, Android apps, desktop tools, or CI pipelines.

## Architecture

```
model.py (Python/PyTorch)          SKaiNET-nanoGPT (Kotlin/SKaiNET)
─────────────────────────          ──────────────────────────────────
GPTConfig                    →     GPTConfig.kt
nn.Embedding                 →     Embedding<T, Float>
LayerNorm                    →     LayerNormalization<T, Float>
CausalSelfAttention          →     causalSelfAttention() + sliceView DSL
MLP                          →     mlp() with .gelu() extension
Block                        →     transformerBlock() (pre-norm + residuals)
GPT.forward(idx)             →     GPTRuntime.forward(tokenIds)
GPT.generate()               →     GPTRuntime.generate() with temp + top-k
from_pretrained()            →     loadGPTWeights() + GPTWeightMapper
weight tying (lm_head=wte)   →     lmHeadWT = weights.wte.t()
```

## Quick start

### Prerequisites

- JDK 21+
- ~2 GB RAM (for GPT-2 base)

### Download the model

```bash
pip install huggingface_hub
huggingface-cli download openai-community/gpt2 --local-dir ~/models/gpt2
```

You need three files: `model.safetensors`, `vocab.json`, `merges.txt`.

### Build and run

```bash
cd SKaiNET-nanoGPT

# Set JDK 21
jenv local 21.0          # or: export JAVA_HOME=/path/to/jdk-21

# Run via Gradle
JAVA_HOME=$(jenv javahome) ./gradlew run \
  --args="--model-dir ~/models/gpt2 -p 'The meaning of life is' -n 100 -t 0.8"
```

### Run the fat JAR

```bash
JAVA_HOME=$(jenv javahome) ./gradlew shadowJar

java --enable-preview --add-modules jdk.incubator.vector \
  -jar build/libs/nanogpt-all.jar \
  --model-dir ~/models/gpt2 \
  --prompt "The meaning of life is" \
  --max-tokens 100 \
  --temperature 0.7 \
  --top-k 40
```

### CLI options

| Flag | Description | Default |
|---|---|---|
| `--model-dir`, `-d` | Directory with model files (required) | — |
| `--model-type` | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl` | `gpt2` |
| `--prompt`, `-p` | Input prompt | `Once upon a time` |
| `--max-tokens`, `-n` | Tokens to generate | `100` |
| `--temperature`, `-t` | Sampling temperature (0 = greedy) | `0.8` |
| `--top-k`, `-k` | Top-k sampling | disabled |

### Supported models

| Model | HuggingFace repo | Parameters | Download |
|---|---|---|---|
| GPT-2 | `openai-community/gpt2` | 124M | 548 MB |
| GPT-2 Medium | `openai-community/gpt2-medium` | 350M | 1.4 GB |
| GPT-2 Large | `openai-community/gpt2-large` | 774M | 3.1 GB |
| GPT-2 XL | `openai-community/gpt2-xl` | 1558M | 6.2 GB |

## Project structure

```
src/main/kotlin/sk/ainet/nanogpt/
├── GPTConfig.kt           Model configuration (GPT2, Medium, Large, XL presets)
├── GPTWeights.kt          Weight data classes (per-layer + model-level)
├── GPTRuntime.kt          Transformer forward pass and autoregressive generation
├── GPTWeightLoader.kt     SafeTensors loading with Conv1D weight transposition
├── GPT2Tokenizer.kt       Byte-level BPE tokenizer (vocab.json + merges.txt)
└── Main.kt                CLI entry point
```

## How it works

The runtime follows the same pattern as SKaiNET's existing LlamaRuntime and BertRuntime: direct tensor operations without Module composition for encoder/decoder layers.

**Forward pass** processes a full token sequence (matching `model.py`'s semantics):
1. Token embedding + learned position embedding
2. N transformer blocks, each: `x = x + attn(ln_1(x))` then `x = x + mlp(ln_2(x))`
3. Final LayerNorm
4. Linear projection to vocabulary (weight-tied with token embeddings)

**Attention** splits the combined QKV projection using SKaiNET's slicing DSL, iterates over heads, applies a pre-computed additive causal mask, and concatenates head outputs.

**Generation** re-runs the full forward pass each step (no KV cache), cropping to the context window when the sequence exceeds `blockSize`. This matches `model.py`'s `generate()` exactly.

## Acknowledgements

This project is a direct port of [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT). If you want to understand the GPT architecture from scratch, watch his excellent video lecture:

- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy
- [nanoGPT](https://github.com/karpathy/nanoGPT) — the simplest, fastest repository for training/finetuning medium-sized GPTs

## Licence

[MIT](LICENCE)
