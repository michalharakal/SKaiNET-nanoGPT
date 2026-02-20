package sk.ainet.nanogpt.cli

import kotlinx.coroutines.runBlocking
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readByteArray
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.lang.types.FP32
import sk.ainet.nanogpt.*
import kotlin.time.measureTime

private fun usage(): Nothing {
    println("Usage: nanogpt <model-dir> [prompt] [max-tokens] [temperature] [--model-type=gpt2]")
    println("  <model-dir>     Directory with model.safetensors, vocab.json, merges.txt")
    println("  [prompt]        Text prompt (default: \"Once upon a time\")")
    println("  [max-tokens]    Number of tokens to generate (default: 100)")
    println("  [temperature]   Sampling temperature (default: 0.8)")
    println("  --model-type=X  gpt2, gpt2-medium, gpt2-large, gpt2-xl (default: gpt2)")
    throw IllegalArgumentException("Invalid arguments")
}

fun main(args: Array<String>) = runBlocking {
    var modelType = "gpt2"
    val filteredArgs = args.filter { arg ->
        when {
            arg.startsWith("--model-type=") -> { modelType = arg.substringAfter("="); false }
            else -> true
        }
    }.toTypedArray()

    if (filteredArgs.isEmpty()) usage()

    val modelDirStr = filteredArgs[0]
    val prompt = filteredArgs.getOrNull(1) ?: "Once upon a time"
    val maxTokens = filteredArgs.getOrNull(2)?.toIntOrNull() ?: 100
    val temperature = filteredArgs.getOrNull(3)?.toFloatOrNull() ?: 0.8f

    val modelDir = Path(modelDirStr)
    val safetensorsPath = Path(modelDirStr, "model.safetensors")
    val vocabPath = Path(modelDirStr, "vocab.json")
    val mergesPath = Path(modelDirStr, "merges.txt")

    if (!SystemFileSystem.exists(safetensorsPath)) error("model.safetensors not found in $modelDirStr")
    if (!SystemFileSystem.exists(vocabPath)) error("vocab.json not found in $modelDirStr")
    if (!SystemFileSystem.exists(mergesPath)) error("merges.txt not found in $modelDirStr")

    val config = when (modelType) {
        "gpt2" -> GPTConfig.GPT2
        "gpt2-medium" -> GPTConfig.GPT2_MEDIUM
        "gpt2-large" -> GPTConfig.GPT2_LARGE
        "gpt2-xl" -> GPTConfig.GPT2_XL
        else -> error("Unknown model type: $modelType. Use: gpt2, gpt2-medium, gpt2-large, gpt2-xl")
    }

    println("SKaiNET-nanoGPT (native)")
    println("  Model dir:   $modelDirStr")
    println("  Model type:  $modelType")
    println("  Config:      ${config.nLayer}L / ${config.nHead}H / ${config.nEmbd}D")
    println("  Temperature: $temperature")
    println("  Max tokens:  $maxTokens")
    println()

    // ── Load tokenizer ────────────────────────────────────────────────
    println("Loading tokenizer...")
    val vocabJson = SystemFileSystem.source(vocabPath).buffered().readByteArray().decodeToString()
    val mergesTxt = SystemFileSystem.source(mergesPath).buffered().readByteArray().decodeToString()
    val tokenizer = GPT2Tokenizer.fromFiles(vocabJson, mergesTxt)
    println("  Tokenizer ready")

    // ── Initialize execution context ──────────────────────────────────
    val ctx = createExecutionContext()

    // ── Load weights from SafeTensors ─────────────────────────────────
    println("Loading weights (reading into memory)...")
    val loadElapsed = measureTime {
        // Read entire safetensors file into memory
    }
    val modelBytes = SystemFileSystem.source(safetensorsPath).buffered().readByteArray()
    println("  Read ${modelBytes.size / 1_000_000} MB into memory")

    val source = ByteArrayRandomAccessSource(modelBytes)
    val loader = SafeTensorsParametersLoader(sourceProvider = { source })

    val weightsElapsed = measureTime {
        // placeholder — actual load happens below
    }
    val weights = loadGPTWeights(loader, ctx, FP32::class, config) { msg ->
        println("  $msg")
    }
    println("  Weights loaded")

    // ── Create runtime ────────────────────────────────────────────────
    val runtime = GPTRuntime(ctx, weights, FP32::class)
    val numParams = runtime.getNumParams()
    val paramsMil = ((numParams / 1e6) * 100).toLong() / 100.0
    println("Model loaded: ${paramsMil}M parameters")
    println()

    // ── Tokenize and generate ─────────────────────────────────────────
    val promptTokens = tokenizer.encode(prompt)
    println("Prompt: \"$prompt\" (${promptTokens.size} tokens)")
    print("Generated: ")

    var tokenCount = 0
    val genElapsed = measureTime {
        runtime.generate(
            prompt = promptTokens,
            maxNewTokens = maxTokens,
            temperature = temperature
        ) { tokenId ->
            print(tokenizer.decode(tokenId))
            tokenCount++
        }
    }

    val seconds = genElapsed.inWholeMilliseconds / 1000.0
    val secRounded = ((seconds * 100).toLong()) / 100.0
    val tokPerSec = ((tokenCount / seconds * 10).toLong()) / 10.0
    println("\n\n--- $tokenCount tokens in ${secRounded}s ($tokPerSec tok/s) ---")
}
