@file:OptIn(ExperimentalWasmJsInterop::class)

package sk.ainet.nanogpt.browser

import kotlin.js.ExperimentalWasmJsInterop
import kotlinx.browser.document
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.khronos.webgl.Int8Array
import org.khronos.webgl.toByteArray
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.lang.types.FP32
import sk.ainet.nanogpt.*
import sk.ainet.lang.tensor.Tensor
import kotlin.time.TimeSource

// ── JS interop: read file data from window._nanogpt ────────────────────────

@JsFun("() => { const b = window._nanogpt.modelBytes; return b ? new Int8Array(b) : null }")
private external fun getModelInt8Array(): Int8Array?

@JsFun("() => window._nanogpt.vocabText || null")
private external fun getVocabText(): JsString?

@JsFun("() => window._nanogpt.mergesText || null")
private external fun getMergesText(): JsString?

@JsFun("() => window._nanogpt.modelSize || 0")
private external fun getModelSize(): Int

@JsFun("() => window._nanogpt.filesVersion || 0")
private external fun getFilesVersion(): Int

@JsFun("() => !!(window._nanogpt.filesReady)")
private external fun isFilesReady(): Boolean

@JsFun("(arr, begin, end) => arr.subarray(begin, end)")
private external fun jsSubarray(arr: Int8Array, begin: Int, end: Int): Int8Array

// ── JS interop: DOM helpers ────────────────────────────────────────────────

@JsFun("(el) => el.value || ''")
private external fun getInputValue(el: JsAny): JsString

@JsFun("(el, v) => { el.disabled = v }")
private external fun setDisabled(el: JsAny, disabled: JsBoolean)

@JsFun("(msg) => { console.log('[nanogpt] ' + msg) }")
private external fun jsLog(msg: JsString)

// ── JS interop: progress bar ───────────────────────────────────────────────

@JsFun("""(pct) => {
    const wrap = document.getElementById('progress-wrap');
    const bar  = document.getElementById('progress-bar');
    if (pct < 0) { wrap.classList.remove('visible'); bar.style.width = '0%'; }
    else { wrap.classList.add('visible'); bar.style.width = Math.min(100, pct) + '%'; }
}""")
private external fun setProgress(pct: Int)

// ── JS interop: generate button click detection ────────────────────────────

@JsFun("""() => {
    if (!window._nanogpt._genWired) {
        window._nanogpt._genClicked = false;
        document.getElementById('generate').addEventListener('click', () => {
            window._nanogpt._genClicked = true;
        });
        window._nanogpt._genWired = true;
    }
}""")
private external fun wireGenerateClick()

@JsFun("() => { const v = window._nanogpt._genClicked; window._nanogpt._genClicked = false; return !!v }")
private external fun consumeGenerateClick(): Boolean

// ── Application state ──────────────────────────────────────────────────────

private val scope = MainScope()

private var loadedRuntime: GPTRuntime<FP32>? = null
private var loadedTokenizer: GPT2Tokenizer? = null

// ── Entry point ────────────────────────────────────────────────────────────

fun main() {
    jsLog("WASM main() started".toJsString())
    statusText("WASM loaded. Drop or select your GPT-2 files.")
    wireGenerateClick()

    scope.launch {
        var lastFilesVersion = 0
        while (true) {
            delay(200)

            val currentVersion = getFilesVersion()
            if (currentVersion > lastFilesVersion && isFilesReady()) {
                lastFilesVersion = currentVersion
                loadedRuntime = null
                loadedTokenizer = null
                setGenerateEnabled(false)
                loadModel()
            }

            if (loadedRuntime != null && consumeGenerateClick()) {
                generate()
            }
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

private suspend fun yieldToBrowser() {
    delay(1)
}

private fun statusText(msg: String) {
    document.getElementById("status")?.let { it.textContent = msg }
}

private fun setGenerateEnabled(enabled: Boolean) {
    document.getElementById("generate")?.let {
        setDisabled(it as JsAny, (!enabled).toJsBoolean())
    }
}

// ── Load Model ─────────────────────────────────────────────────────────────

private suspend fun loadModel() {
    jsLog("loadModel() start".toJsString())
    val jsModelArray = getModelInt8Array()
    val vocabText = getVocabText()?.toString()
    val mergesText = getMergesText()?.toString()

    if (jsModelArray == null || vocabText == null || mergesText == null) {
        statusText("Waiting for files...")
        return
    }

    val config = detectConfig(getModelSize())
    // Total expected tensors: wte + wpe + ln_f_weight + ln_f_bias + nLayer * 12
    val totalTensors = 4 + config.nLayer * 12

    try {
        val loadStart = TimeSource.Monotonic.markNow()

        // Phase 1: tokenizer (fast, ~2% of bar)
        setProgress(0)
        statusText("Loading tokenizer...")
        yieldToBrowser()

        loadedTokenizer = GPT2Tokenizer.fromFiles(vocabText, mergesText)
        val tokenizerMs = loadStart.elapsedNow().inWholeMilliseconds
        jsLog("Phase 1 (tokenizer): ${tokenizerMs}ms".toJsString())
        setProgress(2)
        statusText("Tokenizer ready. Copying model bytes...")
        yieldToBrowser()

        // Phase 2: byte copy from JS → Kotlin (30% of bar)
        val copyStart = TimeSource.Monotonic.markNow()
        val safetensorsBytes = jsArrayToByteArrayWithProgress(jsModelArray, 2, 30)
        val sizeMb = safetensorsBytes.size / 1_000_000
        val copyMs = copyStart.elapsedNow().inWholeMilliseconds
        jsLog("Phase 2 (byte copy ${sizeMb}MB): ${copyMs}ms".toJsString())

        jsLog("Config: ${config.nLayer}L / ${config.nHead}H / ${config.nEmbd}D".toJsString())

        val source = ByteArrayRandomAccessSource(safetensorsBytes)
        val loader = SafeTensorsParametersLoader(sourceProvider = { source })
        val ctx = DirectCpuExecutionContext()

        // Phase 3: parse safetensors into raw tensor map (32% → 70%)
        setProgress(32)
        statusText("Parsing safetensors ($sizeMb MB) — browser may freeze briefly...")
        yieldToBrowser()

        val parseStart = TimeSource.Monotonic.markNow()
        var tensorCount = 0
        val tensors = mutableMapOf<String, Tensor<FP32, Float>>()
        loader.load(ctx, FP32::class) { name, tensor ->
            tensors[name] = tensor
            tensorCount++
        }
        val parseMs = parseStart.elapsedNow().inWholeMilliseconds
        jsLog("Phase 3 (parse $tensorCount tensors): ${parseMs}ms".toJsString())

        // Phase 3b: yield after parse, update progress
        setProgress(70)
        statusText("Parsed $tensorCount tensors in ${parseMs / 1000}s. Mapping weights...")
        yieldToBrowser()

        // Phase 4: map tensors to GPT-2 architecture (70% → 95%)
        // GPTWeightMapper.map does Conv1D → Linear transposition per layer
        val mapStart = TimeSource.Monotonic.markNow()
        val weights = GPTWeightMapper.map(tensors, config)
        val mapMs = mapStart.elapsedNow().inWholeMilliseconds
        jsLog("Phase 4 (weight mapping + transpose): ${mapMs}ms".toJsString())

        // Phase 5: create runtime (95% → 100%)
        setProgress(96)
        statusText("Initializing runtime...")
        yieldToBrowser()

        val runtime = GPTRuntime(ctx, weights, FP32::class)
        loadedRuntime = runtime

        val numParams = runtime.getNumParams()
        val paramsMil = ((numParams / 1e6) * 100).toLong() / 100.0
        val totalMs = loadStart.elapsedNow().inWholeMilliseconds
        val totalSec = totalMs / 1000

        setProgress(100)
        yieldToBrowser()
        setProgress(-1) // hide bar

        statusText("Model loaded: ${paramsMil}M params in ${totalSec}s. Ready to generate.")
        jsLog("Model loaded: ${paramsMil}M params, total ${totalMs}ms".toJsString())
        setGenerateEnabled(true)
    } catch (t: Throwable) {
        setProgress(-1)
        statusText("Error loading model: ${t.message}")
        jsLog("Error: ${t.message}".toJsString())
        println("Error loading model: ${t.stackTraceToString()}")
    }
}

/**
 * Copy JS Int8Array → Kotlin ByteArray in chunks, yielding to update the progress bar.
 * Uses Int8Array.subarray() + toByteArray() for fast bulk copy instead of per-byte interop.
 * [startPct]..[startPct+rangePct] is the progress bar range for this phase.
 */
private suspend fun jsArrayToByteArrayWithProgress(
    jsArr: Int8Array, startPct: Int, rangePct: Int
): ByteArray {
    val len = jsArr.length
    val bytes = ByteArray(len)
    val chunkSize = 4_000_000 // 4 MB chunks (bulk copy is fast, fewer yields needed)
    var offset = 0
    while (offset < len) {
        val end = minOf(offset + chunkSize, len)
        val chunk = jsSubarray(jsArr, offset, end).toByteArray()
        chunk.copyInto(bytes, offset)
        offset = end
        val pct = startPct + (offset.toLong() * rangePct / len).toInt()
        setProgress(pct)
        statusText("Copying model bytes... ${offset / 1_000_000}/${len / 1_000_000} MB")
        yieldToBrowser()
    }
    return bytes
}

// ── Generate ───────────────────────────────────────────────────────────────

private suspend fun generate() {
    jsLog("generate() start".toJsString())
    val runtime = loadedRuntime ?: return
    val tokenizer = loadedTokenizer ?: return
    val output = document.getElementById("output") ?: return

    setGenerateEnabled(false)

    val prompt = document.getElementById("prompt")
        ?.let { getInputValue(it as JsAny).toString() }
        ?: "Once upon a time"
    val maxTokens = document.getElementById("max-tokens")
        ?.let { getInputValue(it as JsAny).toString().toIntOrNull() }
        ?: 100
    val temperature = document.getElementById("temperature")
        ?.let { getInputValue(it as JsAny).toString().toFloatOrNull() }
        ?: 0.8f

    output.textContent = ""
    statusText("Encoding prompt...")
    setProgress(0)
    yieldToBrowser()

    try {
        val promptTokens = tokenizer.encode(prompt)
        output.textContent = prompt
        statusText("Generating...")
        setProgress(0)
        yieldToBrowser()

        var tokenCount = 0
        val mark = TimeSource.Monotonic.markNow()

        runtime.generate(
            prompt = promptTokens,
            maxNewTokens = maxTokens,
            temperature = temperature
        ) { tokenId ->
            val text = tokenizer.decode(tokenId)
            output.appendChild(document.createTextNode(text))
            tokenCount++
            val pct = (tokenCount * 100) / maxTokens
            setProgress(pct)
        }

        setProgress(-1)
        val elapsed = mark.elapsedNow()
        val seconds = elapsed.inWholeMilliseconds / 1000.0
        val secRounded = ((seconds * 100).toLong()) / 100.0
        val tokPerSec = if (seconds > 0) ((tokenCount / seconds * 10).toLong() / 10.0) else 0.0
        statusText("Done: $tokenCount tokens in ${secRounded}s ($tokPerSec tok/s)")
        jsLog("Done: $tokenCount tokens in ${secRounded}s".toJsString())
    } catch (t: Throwable) {
        setProgress(-1)
        statusText("Error: ${t.message}")
        jsLog("Error: ${t.message}".toJsString())
        println("Error during generation: ${t.stackTraceToString()}")
        output.textContent = (output.textContent ?: "") + "\n\nError: ${t.message}"
    } finally {
        setGenerateEnabled(true)
    }
}

// ── Utility ────────────────────────────────────────────────────────────────

private fun detectConfig(fileSize: Int): GPTConfig {
    val mb = fileSize / 1_000_000
    return when {
        mb < 600  -> GPTConfig.GPT2
        mb < 1600 -> GPTConfig.GPT2_MEDIUM
        mb < 3200 -> GPTConfig.GPT2_LARGE
        else      -> GPTConfig.GPT2_XL
    }
}
