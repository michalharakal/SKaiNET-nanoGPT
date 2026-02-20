/**
 * GPT-2 weight loading from HuggingFace SafeTensors format.
 *
 * Follows the [BertWeightLoader] pattern from SKaiNET:
 * 1. Load all tensors via [ParametersLoader] into a flat map
 * 2. Map HuggingFace tensor names to [GPTWeights] structure
 * 3. Transpose Conv1D weights to standard Linear format
 *
 * HuggingFace GPT-2 uses OpenAI's Conv1D module which stores weights
 * as [inFeatures, outFeatures] — the opposite of standard nn.Linear
 * [outFeatures, inFeatures]. Weight matrices marked as Conv1D are
 * transposed during loading.
 */
package sk.ainet.nanogpt

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.t
import sk.ainet.io.ParametersLoader
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * HuggingFace GPT-2 tensor name constants.
 *
 * Names follow the `model.safetensors` layout for `gpt2`, `gpt2-medium`,
 * `gpt2-large`, and `gpt2-xl` checkpoints on HuggingFace.
 */
/**
 * HuggingFace GPT-2 tensor name resolver.
 *
 * Some checkpoints use a `transformer.` prefix (e.g. PyTorch `state_dict` exports),
 * while others omit it (e.g. SafeTensors from `openai-community/gpt2`).
 * This class auto-detects the prefix from the loaded tensor keys.
 */
private class GPT2TensorNames(tensorKeys: Set<String>) {
    private val prefix: String = if (tensorKeys.any { it.startsWith("transformer.") }) "transformer." else ""

    val tokenEmbedding get() = "${prefix}wte.weight"
    val positionEmbedding get() = "${prefix}wpe.weight"
    val lnFWeight get() = "${prefix}ln_f.weight"
    val lnFBias get() = "${prefix}ln_f.bias"

    fun ln1Weight(layer: Int) = "${prefix}h.$layer.ln_1.weight"
    fun ln1Bias(layer: Int) = "${prefix}h.$layer.ln_1.bias"
    fun cAttnWeight(layer: Int) = "${prefix}h.$layer.attn.c_attn.weight"
    fun cAttnBias(layer: Int) = "${prefix}h.$layer.attn.c_attn.bias"
    fun cProjWeight(layer: Int) = "${prefix}h.$layer.attn.c_proj.weight"
    fun cProjBias(layer: Int) = "${prefix}h.$layer.attn.c_proj.bias"
    fun ln2Weight(layer: Int) = "${prefix}h.$layer.ln_2.weight"
    fun ln2Bias(layer: Int) = "${prefix}h.$layer.ln_2.bias"
    fun mlpFcWeight(layer: Int) = "${prefix}h.$layer.mlp.c_fc.weight"
    fun mlpFcBias(layer: Int) = "${prefix}h.$layer.mlp.c_fc.bias"
    fun mlpProjWeight(layer: Int) = "${prefix}h.$layer.mlp.c_proj.weight"
    fun mlpProjBias(layer: Int) = "${prefix}h.$layer.mlp.c_proj.bias"

    companion object {
        /** Weight keys that are Conv1D format and need transposing to Linear format. */
        val CONV1D_WEIGHTS = setOf("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
    }
}

/**
 * Map a flat tensor dictionary into typed [GPTWeights].
 *
 * Handles Conv1D → Linear transposition for weight matrices.
 */
public object GPTWeightMapper {
    public fun <T : DType> map(
        tensors: Map<String, Tensor<T, Float>>,
        config: GPTConfig
    ): GPTWeights<T> {
        val names = GPT2TensorNames(tensors.keys)

        fun get(name: String): Tensor<T, Float> =
            tensors[name] ?: error("Missing tensor: '$name'. Available: ${tensors.keys.sorted()}")

        fun getConv1D(name: String): Tensor<T, Float> {
            val tensor = get(name)
            // Conv1D stores [inFeatures, outFeatures]; Linear expects [outFeatures, inFeatures]
            return if (GPT2TensorNames.CONV1D_WEIGHTS.any { name.endsWith(it) }) {
                tensor.t()
            } else {
                tensor
            }
        }

        val layers = (0 until config.nLayer).map { i ->
            GPTLayerWeights(
                ln1Weight = get(names.ln1Weight(i)),
                ln1Bias = get(names.ln1Bias(i)),
                cAttnWeight = getConv1D(names.cAttnWeight(i)),
                cAttnBias = get(names.cAttnBias(i)),
                cProjWeight = getConv1D(names.cProjWeight(i)),
                cProjBias = get(names.cProjBias(i)),
                ln2Weight = get(names.ln2Weight(i)),
                ln2Bias = get(names.ln2Bias(i)),
                mlpFcWeight = getConv1D(names.mlpFcWeight(i)),
                mlpFcBias = get(names.mlpFcBias(i)),
                mlpProjWeight = getConv1D(names.mlpProjWeight(i)),
                mlpProjBias = get(names.mlpProjBias(i))
            )
        }

        return GPTWeights(
            config = config,
            wte = get(names.tokenEmbedding),
            wpe = get(names.positionEmbedding),
            layers = layers,
            lnFWeight = get(names.lnFWeight),
            lnFBias = get(names.lnFBias)
        )
    }
}

/**
 * Load GPT-2 weights from one or more SafeTensors files.
 *
 * Usage:
 * ```kotlin
 * val loader = SafeTensorsParametersLoader(
 *     sourceProvider = { JvmRandomAccessSource.open(modelPath) }
 * )
 * val weights = loadGPTWeights(loader, ctx, FP32::class, GPTConfig.GPT2)
 * ```
 *
 * For models split across multiple shards (e.g., `model-00001-of-00002.safetensors`),
 * pass multiple loaders.
 */
public suspend fun <T : DType> loadGPTWeights(
    loaders: List<ParametersLoader>,
    ctx: ExecutionContext,
    dtype: KClass<T>,
    config: GPTConfig,
    onProgress: (String) -> Unit = {}
): GPTWeights<T> {
    val tensors = mutableMapOf<String, Tensor<T, Float>>()

    for ((idx, loader) in loaders.withIndex()) {
        onProgress("Loading shard ${idx + 1}/${loaders.size}...")
        loader.load<T, Float>(ctx, dtype) { name, tensor ->
            tensors[name] = tensor
            onProgress("  Loaded: $name ${tensor.shape}")
        }
    }

    onProgress("Mapping ${tensors.size} tensors to GPT-2 architecture...")
    return GPTWeightMapper.map(tensors, config)
}

/** Convenience overload for a single SafeTensors file. */
public suspend fun <T : DType> loadGPTWeights(
    loader: ParametersLoader,
    ctx: ExecutionContext,
    dtype: KClass<T>,
    config: GPTConfig,
    onProgress: (String) -> Unit = {}
): GPTWeights<T> = loadGPTWeights(listOf(loader), ctx, dtype, config, onProgress)
