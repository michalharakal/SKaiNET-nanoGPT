/**
 * GPT-2 runtime — full-sequence forward pass and autoregressive generation.
 *
 * Port of Karpathy's nanoGPT `GPT` class using SKaiNET tensor DSLs:
 * - **Slicing DSL** (`sliceView {}`) for QKV splitting and head extraction
 * - **Tensor creation DSL** (`tensor {}`) for causal mask construction
 * - **LayerNormalization** modules for pre-norm architecture
 * - **Tensor extension functions** (`.gelu()`, `.matmul()`, `.softmax()`, `.t()`, `.tril()`)
 *
 * Follows the [BertRuntime] pattern: direct tensor ops, no Module composition for layers.
 */
package sk.ainet.nanogpt

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.nn.normalization.LayerNormalization
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.div
import sk.ainet.lang.tensor.gelu
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.minus
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.sliceView
import sk.ainet.lang.tensor.softmax
import sk.ainet.lang.tensor.t
import sk.ainet.lang.tensor.times
import sk.ainet.lang.tensor.tril
import sk.ainet.lang.types.DType
import kotlin.math.exp
import kotlin.math.sqrt as mathSqrt
import kotlin.random.Random
import kotlin.reflect.KClass

/**
 * GPT-2 inference runtime.
 *
 * Supports two modes:
 * - **Full-sequence forward** ([forward]): processes a complete token sequence, returns logits
 *   for every position. Matches model.py's `forward(idx)` semantics.
 * - **Autoregressive generation** ([generate]): feeds tokens one at a time, samples new tokens.
 *
 * @param T The data type for weight tensors (typically [FP32])
 * @param ctx SKaiNET execution context (CPU backend with SIMD acceleration)
 * @param weights Pre-loaded GPT-2 weights
 * @param dtype KClass token for the data type
 */
public class GPTRuntime<T : DType>(
    private val ctx: ExecutionContext,
    private val weights: GPTWeights<T>,
    private val dtype: KClass<T>,
    private val random: Random = Random.Default
) {
    private val config: GPTConfig get() = weights.config
    private val nEmbd: Int = config.nEmbd
    private val nHead: Int = config.nHead
    private val headDim: Int = config.headDim

    // ── Embedding modules ─────────────────────────────────────────────────

    private val tokenEmbedding = Embedding<T, Float>(
        numEmbeddings = config.vocabSize,
        embeddingDim = nEmbd,
        initWeight = weights.wte,
        name = "wte"
    )

    private val positionEmbedding = Embedding<T, Float>(
        numEmbeddings = config.blockSize,
        embeddingDim = nEmbd,
        initWeight = weights.wpe,
        name = "wpe"
    )

    // ── LayerNorm modules ─────────────────────────────────────────────────

    private val lnF = LayerNormalization<T, Float>(
        normalizedShape = intArrayOf(nEmbd),
        eps = 1e-5,
        elementwiseAffine = true,
        name = "ln_f",
        initGamma = weights.lnFWeight,
        initBeta = weights.lnFBias
    )

    private val ln1s: List<LayerNormalization<T, Float>> = weights.layers.mapIndexed { i, layer ->
        LayerNormalization(
            normalizedShape = intArrayOf(nEmbd),
            eps = 1e-5,
            elementwiseAffine = true,
            name = "h.$i.ln_1",
            initGamma = layer.ln1Weight,
            initBeta = layer.ln1Bias
        )
    }

    private val ln2s: List<LayerNormalization<T, Float>> = weights.layers.mapIndexed { i, layer ->
        LayerNormalization(
            normalizedShape = intArrayOf(nEmbd),
            eps = 1e-5,
            elementwiseAffine = true,
            name = "h.$i.ln_2",
            initGamma = layer.ln2Weight,
            initBeta = layer.ln2Bias
        )
    }

    // ── Pre-transposed weights (following LlamaRuntime pattern) ───────────

    private class TransposedWeights<T : DType>(
        val cAttnWT: Tensor<T, Float>,      // [nEmbd, 3*nEmbd]
        val cProjWT: Tensor<T, Float>,      // [nEmbd, nEmbd]
        val mlpFcWT: Tensor<T, Float>,      // [nEmbd, 4*nEmbd]
        val mlpProjWT: Tensor<T, Float>     // [4*nEmbd, nEmbd]
    )

    private val transposed: List<TransposedWeights<T>> = weights.layers.map { layer ->
        TransposedWeights(
            cAttnWT = layer.cAttnWeight.t(),
            cProjWT = layer.cProjWeight.t(),
            mlpFcWT = layer.mlpFcWeight.t(),
            mlpProjWT = layer.mlpProjWeight.t()
        )
    }

    /** lm_head shares weights with wte (weight tying). */
    private val lmHeadWT: Tensor<T, Float> = weights.wte.t()   // [nEmbd, vocabSize]

    // ── Causal mask (tensor creation DSL) ─────────────────────────────────

    /**
     * Pre-computed additive causal mask using the SKaiNET **tensor creation DSL**.
     *
     * Creates a lower-triangular ones matrix via `.tril()`, then converts to an
     * additive mask: 0.0 where attention is allowed, -1e9 where it's blocked.
     *
     * Shape: [blockSize, blockSize]
     */
    private val causalMask: Tensor<T, Float> = run {
        // ── Tensor creation DSL ──────────────────────────────────────
        // Creates a lower-triangular ones matrix, then converts to additive mask.
        val bs = config.blockSize
        val ones = ctx.ones<T, Float>(Shape(bs, bs), dtype)
        val tril = ones.tril()
        val negInf = ctx.full<T, Float>(Shape(bs, bs), dtype, -1e9f)

        // Additive causal mask: lower triangle → 0.0, upper triangle → -1e9
        // Formula: (1 - tril) * (-1e9)
        (ones - tril) * negInf
    }

    // ── Full-sequence forward pass ────────────────────────────────────────

    /**
     * Forward pass over a complete token sequence.
     *
     * Mirrors model.py's `forward(idx, targets=None)`:
     * ```python
     * tok_emb = self.transformer.wte(idx)
     * pos_emb = self.transformer.wpe(pos)
     * x = self.transformer.drop(tok_emb + pos_emb)
     * for block in self.transformer.h:
     *     x = block(x)
     * x = self.transformer.ln_f(x)
     * logits = self.lm_head(x)
     * ```
     *
     * @param tokenIds Token ID sequence, shape [seqLen]
     * @return Logits tensor, shape [seqLen, vocabSize]
     */
    public fun forward(tokenIds: IntArray): Tensor<T, Float> {
        val seqLen = tokenIds.size
        require(seqLen <= config.blockSize) {
            "Sequence length $seqLen exceeds block size ${config.blockSize}"
        }

        // Embeddings: token + position
        val positionIds = IntArray(seqLen) { it }
        val tokEmb = tokenEmbedding.forward(tokenIds, ctx)     // [seqLen, nEmbd]
        val posEmb = positionEmbedding.forward(positionIds, ctx) // [seqLen, nEmbd]
        var x = tokEmb + posEmb

        // Transformer blocks
        for (i in weights.layers.indices) {
            x = transformerBlock(i, x, seqLen)
        }

        // Final LayerNorm + lm_head projection (weight-tied with wte)
        x = lnF.forward(x, ctx)
        return x.matmul(lmHeadWT)   // [seqLen, vocabSize]
    }

    // ── Transformer block ─────────────────────────────────────────────────

    /**
     * Single transformer block: pre-norm architecture with residual connections.
     *
     * ```
     * x = x + attn(ln_1(x))
     * x = x + mlp(ln_2(x))
     * ```
     */
    private fun transformerBlock(
        layerIdx: Int,
        x: Tensor<T, Float>,
        seqLen: Int
    ): Tensor<T, Float> {
        val layer = weights.layers[layerIdx]
        val tw = transposed[layerIdx]

        // Self-attention with pre-norm
        val normed1 = ln1s[layerIdx].forward(x, ctx)
        val attnOut = causalSelfAttention(normed1, layer, tw, seqLen)
        val afterAttn = x + attnOut

        // MLP with pre-norm
        val normed2 = ln2s[layerIdx].forward(afterAttn, ctx)
        val mlpOut = mlp(normed2, layer, tw)
        return afterAttn + mlpOut
    }

    // ── Causal self-attention (slicing DSL) ───────────────────────────────

    /**
     * Multi-head causal self-attention.
     *
     * Uses the SKaiNET **slicing DSL** (`sliceView {}`) for:
     * - Splitting the combined QKV projection into Q, K, V
     * - Extracting per-head slices
     * - Cropping the causal mask to the current sequence length
     *
     * Follows the BERT attention pattern with an added causal (lower-triangular) mask.
     */
    private fun causalSelfAttention(
        input: Tensor<T, Float>,
        layer: GPTLayerWeights<T>,
        tw: TransposedWeights<T>,
        seqLen: Int
    ): Tensor<T, Float> {
        // Combined QKV projection: [seqLen, nEmbd] @ [nEmbd, 3*nEmbd] → [seqLen, 3*nEmbd]
        val qkv = input.matmul(tw.cAttnWT) + layer.cAttnBias

        // ── Slicing DSL: split QKV into Q, K, V ──────────────────────────
        val q = qkv.sliceView {
            segment { all() }                       // dim 0: all positions
            segment { range(0, nEmbd) }             // dim 1: first nEmbd → Q
        }
        val k = qkv.sliceView {
            segment { all() }                       // dim 0: all positions
            segment { range(nEmbd, 2 * nEmbd) }    // dim 1: middle nEmbd → K
        }
        val v = qkv.sliceView {
            segment { all() }                       // dim 0: all positions
            segment { range(2 * nEmbd, 3 * nEmbd) } // dim 1: last nEmbd → V
        }

        // ── Slicing DSL: crop causal mask to current sequence length ─────
        val mask = causalMask.sliceView {
            segment { range(0, seqLen) }    // rows [0, seqLen)
            segment { range(0, seqLen) }    // cols [0, seqLen)
        }

        // ── Per-head attention (BERT-style with causal mask) ─────────────
        val scale = mathSqrt(headDim.toDouble()).toFloat()
        val headOutputs = ArrayList<Tensor<T, Float>>(nHead)

        for (h in 0 until nHead) {
            val offset = h * headDim

            // Slicing DSL: extract per-head Q, K, V slices
            val qh = q.sliceView {
                segment { all() }
                segment { range(offset, offset + headDim) }
            }
            val kh = k.sliceView {
                segment { all() }
                segment { range(offset, offset + headDim) }
            }
            val vh = v.sliceView {
                segment { all() }
                segment { range(offset, offset + headDim) }
            }

            // Scaled dot-product attention with causal mask:
            //   attn = softmax(Q @ K^T / sqrt(d) + mask) @ V
            val scores = qh.matmul(kh.t()) / scale  // [seqLen, seqLen]
            val masked = scores + mask                // apply causal mask (additive -1e9)
            val attnWeights = masked.softmax(dim = 1) // softmax over key dimension
            val headOut = attnWeights.matmul(vh)       // [seqLen, headDim]

            headOutputs.add(headOut)
        }

        // Concatenate heads along feature dimension → [seqLen, nEmbd]
        val attnOutput = q.ops.concat(headOutputs, dim = 1)

        // Output projection: [seqLen, nEmbd] @ [nEmbd, nEmbd] → [seqLen, nEmbd]
        return attnOutput.matmul(tw.cProjWT) + layer.cProjBias
    }

    // ── MLP (feed-forward network) ────────────────────────────────────────

    /**
     * GPT-2 MLP: two linear projections with GELU activation.
     *
     * ```python
     * x = self.c_fc(x)        # [seqLen, nEmbd] → [seqLen, 4*nEmbd]
     * x = self.gelu(x)
     * x = self.c_proj(x)      # [seqLen, 4*nEmbd] → [seqLen, nEmbd]
     * ```
     */
    private fun mlp(
        input: Tensor<T, Float>,
        layer: GPTLayerWeights<T>,
        tw: TransposedWeights<T>
    ): Tensor<T, Float> {
        val fc = (input.matmul(tw.mlpFcWT) + layer.mlpFcBias).gelu()
        return fc.matmul(tw.mlpProjWT) + layer.mlpProjBias
    }

    // ── Autoregressive generation ─────────────────────────────────────────

    /**
     * Generate new tokens autoregressively.
     *
     * Mirrors model.py's `generate(idx, max_new_tokens, temperature, top_k)`:
     * re-runs the full forward pass each step (no KV cache), cropping to
     * `blockSize` if the sequence grows too long.
     *
     * @param prompt Initial token IDs
     * @param maxNewTokens Number of tokens to generate
     * @param temperature Sampling temperature (0.0 = greedy)
     * @param topK If non-null, restrict sampling to top-k logits
     * @param onToken Callback invoked for each generated token
     */
    public fun generate(
        prompt: IntArray,
        maxNewTokens: Int,
        temperature: Float = 1.0f,
        topK: Int? = null,
        onToken: (Int) -> Unit
    ) {
        var sequence = prompt.copyOf()

        for (step in 0 until maxNewTokens) {
            // Crop to block size if sequence exceeds context window
            val input = if (sequence.size > config.blockSize) {
                sequence.copyOfRange(sequence.size - config.blockSize, sequence.size)
            } else {
                sequence
            }

            // Forward pass → logits for all positions
            val logits = forward(input)
            val seqLen = input.size

            // Extract logits at the last position using slicing DSL
            val lastLogits = logits.sliceView {
                segment { at(seqLen - 1) }  // select last position
                segment { all() }           // all vocab logits
            }

            // Sample next token
            val nextToken = sample(lastLogits, temperature, topK)
            onToken(nextToken)

            // Append to sequence
            sequence = sequence + nextToken
        }
    }

    /**
     * Sample a token from logits with temperature scaling and optional top-k filtering.
     */
    private fun sample(logits: Tensor<T, Float>, temperature: Float, topK: Int?): Int {
        val buf = logits.data.copyToFloatArray()

        // Greedy (argmax)
        if (temperature <= 1e-6f) {
            var best = 0
            for (i in 1 until buf.size) {
                if (buf[i] > buf[best]) best = i
            }
            return best
        }

        // Temperature scaling
        for (i in buf.indices) buf[i] /= temperature

        // Top-k filtering: set logits outside top-k to -inf
        if (topK != null && topK < buf.size) {
            val threshold = buf.sortedDescending()[topK - 1]
            for (i in buf.indices) {
                if (buf[i] < threshold) buf[i] = Float.NEGATIVE_INFINITY
            }
        }

        // Softmax → categorical sample
        var maxLogit = Float.NEGATIVE_INFINITY
        for (v in buf) if (v > maxLogit) maxLogit = v
        var sum = 0f
        for (i in buf.indices) {
            val e = exp((buf[i] - maxLogit).toDouble()).toFloat()
            buf[i] = e
            sum += e
        }
        val r = random.nextFloat() * sum
        var acc = 0f
        for (i in buf.indices) {
            acc += buf[i]
            if (acc >= r) return i
        }
        return buf.lastIndex
    }

    // ── Utility ───────────────────────────────────────────────────────────

    /** Count total model parameters (excluding position embeddings, like model.py). */
    public fun getNumParams(nonEmbedding: Boolean = true): Long {
        var total = 0L
        total += config.vocabSize.toLong() * nEmbd   // wte
        total += config.blockSize.toLong() * nEmbd    // wpe
        total += nEmbd * 2L                            // ln_f (weight + bias)

        for (i in weights.layers.indices) {
            total += nEmbd * 2L                        // ln_1
            total += 3L * nEmbd * nEmbd + 3L * nEmbd  // c_attn (weight + bias)
            total += nEmbd.toLong() * nEmbd + nEmbd    // c_proj
            total += nEmbd * 2L                        // ln_2
            total += 4L * nEmbd * nEmbd + 4L * nEmbd  // mlp.c_fc
            total += 4L * nEmbd * nEmbd + nEmbd        // mlp.c_proj
        }

        if (nonEmbedding) {
            total -= config.blockSize.toLong() * nEmbd // subtract wpe
        }
        return total
    }
}
