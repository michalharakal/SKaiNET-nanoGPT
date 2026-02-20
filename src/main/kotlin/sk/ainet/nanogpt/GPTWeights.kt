/**
 * Weight structures for GPT-2.
 *
 * Follows the [BertRuntimeWeights] pattern from SKaiNET.
 * All weight matrices are stored in standard Linear format: [outFeatures, inFeatures].
 * The weight loader handles Conv1D transposition from HuggingFace format.
 */
package sk.ainet.nanogpt

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Per-layer weights for a GPT-2 transformer block.
 *
 * Maps to model.py's `Block` which contains:
 * - `ln_1` / `ln_2`: pre-norm LayerNorms
 * - `attn.c_attn`: combined QKV projection [nEmbd -> 3*nEmbd]
 * - `attn.c_proj`: attention output projection [nEmbd -> nEmbd]
 * - `mlp.c_fc`: FFN up-projection [nEmbd -> 4*nEmbd]
 * - `mlp.c_proj`: FFN down-projection [4*nEmbd -> nEmbd]
 */
public data class GPTLayerWeights<T : DType>(
    // Pre-attention LayerNorm
    val ln1Weight: Tensor<T, Float>,        // [nEmbd]
    val ln1Bias: Tensor<T, Float>,          // [nEmbd]
    // Causal self-attention
    val cAttnWeight: Tensor<T, Float>,      // [3*nEmbd, nEmbd] — combined Q, K, V projection
    val cAttnBias: Tensor<T, Float>,        // [3*nEmbd]
    val cProjWeight: Tensor<T, Float>,      // [nEmbd, nEmbd]   — output projection
    val cProjBias: Tensor<T, Float>,        // [nEmbd]
    // Pre-MLP LayerNorm
    val ln2Weight: Tensor<T, Float>,        // [nEmbd]
    val ln2Bias: Tensor<T, Float>,          // [nEmbd]
    // Feed-forward network (MLP)
    val mlpFcWeight: Tensor<T, Float>,      // [4*nEmbd, nEmbd] — up-projection
    val mlpFcBias: Tensor<T, Float>,        // [4*nEmbd]
    val mlpProjWeight: Tensor<T, Float>,    // [nEmbd, 4*nEmbd] — down-projection
    val mlpProjBias: Tensor<T, Float>       // [nEmbd]
)

/**
 * Complete GPT-2 model weights.
 *
 * The language model head (`lm_head`) is tied to the token embedding (`wte`),
 * following the weight-tying scheme from the original GPT-2 paper.
 */
public data class GPTWeights<T : DType>(
    val config: GPTConfig,
    // Embeddings
    val wte: Tensor<T, Float>,              // [vocabSize, nEmbd]  — token embeddings
    val wpe: Tensor<T, Float>,              // [blockSize, nEmbd]  — learned position embeddings
    // Transformer layers
    val layers: List<GPTLayerWeights<T>>,
    // Final LayerNorm
    val lnFWeight: Tensor<T, Float>,        // [nEmbd]
    val lnFBias: Tensor<T, Float>           // [nEmbd]
    // lm_head.weight is tied to wte — no separate tensor needed
)
