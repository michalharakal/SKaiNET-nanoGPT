/**
 * GPT-2 model configuration.
 *
 * Port of Karpathy's nanoGPT `GPTConfig` dataclass.
 * Default values match GPT-2 base (124M parameters).
 */
package sk.ainet.nanogpt

public data class GPTConfig(
    /** Maximum sequence length (context window). */
    val blockSize: Int = 1024,
    /** Vocabulary size. 50257 for GPT-2, padded to 50304 for efficiency in training. */
    val vocabSize: Int = 50304,
    /** Number of transformer layers. */
    val nLayer: Int = 12,
    /** Number of attention heads. */
    val nHead: Int = 12,
    /** Embedding dimension. */
    val nEmbd: Int = 768,
    /** Dropout rate (0.0 = no dropout). */
    val dropout: Float = 0.0f,
    /** Whether to use bias in Linear layers and LayerNorms. */
    val bias: Boolean = true
) {
    /** Dimension per attention head. */
    val headDim: Int get() = nEmbd / nHead

    init {
        require(nEmbd % nHead == 0) {
            "nEmbd ($nEmbd) must be divisible by nHead ($nHead)"
        }
    }

    public companion object {
        /** GPT-2 base: 12 layers, 12 heads, 768 dim — 124M params. */
        val GPT2 = GPTConfig(
            nLayer = 12, nHead = 12, nEmbd = 768,
            vocabSize = 50257, blockSize = 1024, bias = true
        )
        /** GPT-2 medium: 24 layers, 16 heads, 1024 dim — 350M params. */
        val GPT2_MEDIUM = GPTConfig(
            nLayer = 24, nHead = 16, nEmbd = 1024,
            vocabSize = 50257, blockSize = 1024, bias = true
        )
        /** GPT-2 large: 36 layers, 20 heads, 1280 dim — 774M params. */
        val GPT2_LARGE = GPTConfig(
            nLayer = 36, nHead = 20, nEmbd = 1280,
            vocabSize = 50257, blockSize = 1024, bias = true
        )
        /** GPT-2 XL: 48 layers, 25 heads, 1600 dim — 1558M params. */
        val GPT2_XL = GPTConfig(
            nLayer = 48, nHead = 25, nEmbd = 1600,
            vocabSize = 50257, blockSize = 1024, bias = true
        )
    }
}
