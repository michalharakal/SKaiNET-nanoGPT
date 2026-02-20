/**
 * SKaiNET-nanoGPT CLI — GPT-2 inference on the JVM.
 *
 * Usage:
 *   ./gradlew run --args="--model-dir /path/to/gpt2 --prompt 'Hello world'"
 *
 * The model directory should contain files downloaded from HuggingFace:
 *   model.safetensors, vocab.json, merges.txt
 *
 * Supports: gpt2, gpt2-medium, gpt2-large, gpt2-xl.
 */
package sk.ainet.nanogpt

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.lang.types.FP32
import java.io.File

fun main(args: Array<String>) {
    // ── Parse arguments ───────────────────────────────────────────────
    val parsed = parseArgs(args)
    val modelDir = parsed["--model-dir"] ?: parsed["-d"]
    val prompt = parsed["--prompt"] ?: parsed["-p"] ?: "Once upon a time"
    val maxTokens = (parsed["--max-tokens"] ?: parsed["-n"] ?: "100").toInt()
    val temperature = (parsed["--temperature"] ?: parsed["-t"] ?: "0.8").toFloat()
    val topK = (parsed["--top-k"] ?: parsed["-k"])?.toInt()
    val modelType = parsed["--model-type"] ?: "gpt2"

    if (modelDir == null) {
        printUsage()
        return
    }

    val resolvedDir = if (modelDir.startsWith("~")) {
        modelDir.replaceFirst("~", System.getProperty("user.home"))
    } else {
        modelDir
    }
    val dir = File(resolvedDir)
    val safetensorsFile = File(dir, "model.safetensors")
    require(dir.isDirectory) { "Not a directory: $modelDir" }
    require(safetensorsFile.exists()) { "model.safetensors not found in $modelDir" }

    val config = when (modelType) {
        "gpt2" -> GPTConfig.GPT2
        "gpt2-medium" -> GPTConfig.GPT2_MEDIUM
        "gpt2-large" -> GPTConfig.GPT2_LARGE
        "gpt2-xl" -> GPTConfig.GPT2_XL
        else -> error("Unknown model type: $modelType. Use: gpt2, gpt2-medium, gpt2-large, gpt2-xl")
    }

    println("SKaiNET-nanoGPT")
    println("  Model dir:   $resolvedDir")
    println("  Model type:  $modelType")
    println("  Config:      ${config.nLayer}L / ${config.nHead}H / ${config.nEmbd}D")
    println("  Temperature: $temperature")
    println("  Top-k:       ${topK ?: "disabled"}")
    println("  Max tokens:  $maxTokens")
    println()

    // ── Load tokenizer ────────────────────────────────────────────────
    val vocabFile = File(dir, "vocab.json")
    val mergesFile = File(dir, "merges.txt")
    println("Loading tokenizer...")
    println("  vocab:  ${vocabFile.absolutePath} (${vocabFile.length() / 1024} KB)")
    println("  merges: ${mergesFile.absolutePath} (${mergesFile.length() / 1024} KB)")
    val tokenizer = GPT2Tokenizer.fromFiles(vocabFile.readText(), mergesFile.readText())
    println("  Tokenizer ready")

    // ── Initialize SKaiNET execution context (CPU backend with SIMD) ──
    val ctx = DirectCpuExecutionContext()

    // ── Load weights from SafeTensors ─────────────────────────────────
    println("Loading weights...")
    println("  file: ${safetensorsFile.absolutePath} (%.1f MB)".format(safetensorsFile.length() / 1e6))
    val loader = SafeTensorsParametersLoader(
        sourceProvider = { JvmRandomAccessSource.open(safetensorsFile.absolutePath) }
    )

    val startLoad = System.nanoTime()
    val weights = runBlocking {
        loadGPTWeights(loader, ctx, FP32::class, config) { msg ->
            println("  $msg")
        }
    }
    val loadTime = (System.nanoTime() - startLoad) / 1e9
    println("  Weights loaded in %.2fs".format(loadTime))

    // ── Create runtime ────────────────────────────────────────────────
    val runtime = GPTRuntime(ctx, weights, FP32::class)
    val numParams = runtime.getNumParams()
    println("Model loaded: %.2fM parameters".format(numParams / 1e6))
    println()

    // ── Tokenize and generate ─────────────────────────────────────────
    val promptTokens = tokenizer.encode(prompt)
    println("Prompt: \"$prompt\" (${promptTokens.size} tokens)")
    print("Generated: ")

    val startTime = System.nanoTime()
    var tokenCount = 0

    runtime.generate(
        prompt = promptTokens,
        maxNewTokens = maxTokens,
        temperature = temperature,
        topK = topK
    ) { tokenId ->
        print(tokenizer.decode(tokenId))
        System.out.flush()
        tokenCount++
    }

    val elapsed = (System.nanoTime() - startTime) / 1e9
    println("\n\n--- $tokenCount tokens in %.2fs (%.1f tok/s) ---".format(elapsed, tokenCount / elapsed))
}

// ── Argument parsing ──────────────────────────────────────────────────

private fun parseArgs(args: Array<String>): Map<String, String> {
    val map = mutableMapOf<String, String>()
    var i = 0
    while (i < args.size) {
        if (args[i].startsWith("-") && i + 1 < args.size) {
            map[args[i]] = args[i + 1]
            i += 2
        } else {
            i++
        }
    }
    return map
}

private fun printUsage() {
    println("""
        |SKaiNET-nanoGPT — GPT-2 inference on the JVM via SKaiNET
        |
        |Setup:
        |  pip install huggingface_hub
        |  huggingface-cli download openai-community/gpt2 --local-dir ~/models/gpt2
        |
        |Usage:
        |  ./gradlew run --args="--model-dir <path> [options]"
        |
        |Required:
        |  --model-dir, -d <path>   Directory with model.safetensors, vocab.json, merges.txt
        |
        |Options:
        |  --model-type <type>      gpt2, gpt2-medium, gpt2-large, gpt2-xl (default: gpt2)
        |  --prompt, -p <text>      Prompt text (default: "Once upon a time")
        |  --max-tokens, -n <int>   Max tokens to generate (default: 100)
        |  --temperature, -t <f>    Sampling temperature (default: 0.8)
        |  --top-k, -k <int>        Top-k sampling (default: disabled)
        |
        |Example:
        |  ./gradlew run --args="--model-dir ~/models/gpt2 -p 'The meaning of life is' -n 200 -t 0.7"
    """.trimMargin())
}
