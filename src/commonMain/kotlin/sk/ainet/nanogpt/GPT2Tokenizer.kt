/**
 * GPT-2 byte-level BPE tokenizer.
 *
 * Loads `vocab.json` and `merges.txt` from a HuggingFace GPT-2 model directory
 * and performs encoding/decoding matching the original OpenAI tiktoken behavior.
 */
package sk.ainet.nanogpt

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.int

/**
 * GPT-2 BPE tokenizer.
 *
 * Usage:
 * ```kotlin
 * val tokenizer = GPT2Tokenizer.fromDirectory("/path/to/gpt2/")
 * val ids = tokenizer.encode("Hello world")
 * val text = tokenizer.decode(ids)
 * ```
 */
public class GPT2Tokenizer private constructor(
    private val encoder: Map<String, Int>,
    private val decoder: Map<Int, String>,
    private val bpeRanks: Map<Pair<String, String>, Int>
) {
    /** The <|endoftext|> token ID (50256). */
    val eotToken: Int = encoder["<|endoftext|>"] ?: 50256

    /**
     * Encode text into a list of BPE token IDs.
     */
    public fun encode(text: String): IntArray {
        val tokens = mutableListOf<Int>()
        // Split into words (GPT-2 regex pattern: contractions, words, numbers, whitespace+char)
        val pattern = Regex("""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        for (match in pattern.findAll(text)) {
            val word = match.value
            // Convert each byte to the GPT-2 byte encoding
            val encoded = word.encodeToByteArray().map { BYTE_ENCODER[it.toInt() and 0xFF]!! }.joinToString("")
            // Apply BPE merges
            val bpeTokens = bpe(encoded)
            for (bpeToken in bpeTokens.split(" ")) {
                val id = encoder[bpeToken]
                if (id != null) tokens.add(id)
            }
        }
        return tokens.toIntArray()
    }

    /**
     * Decode token IDs back to text.
     */
    public fun decode(tokenIds: IntArray): String {
        val pieces = tokenIds.map { decoder[it] ?: "" }
        val joined = pieces.joinToString("")
        // Reverse the byte encoding
        val bytes = joined.map { BYTE_DECODER[it] ?: 0 }.toByteArray()
        return bytes.decodeToString()
    }

    /** Decode a single token ID to text. */
    public fun decode(tokenId: Int): String = decode(intArrayOf(tokenId))

    // ── BPE algorithm ─────────────────────────────────────────────────────

    private fun bpe(token: String): String {
        if (token.length <= 1) return token

        var word = token.map { it.toString() }

        while (true) {
            // Find the highest-priority (lowest-rank) merge pair
            var bestPair: Pair<String, String>? = null
            var bestRank = Int.MAX_VALUE
            for (i in 0 until word.size - 1) {
                val pair = word[i] to word[i + 1]
                val rank = bpeRanks[pair]
                if (rank != null && rank < bestRank) {
                    bestRank = rank
                    bestPair = pair
                }
            }
            if (bestPair == null) break

            // Merge the pair
            val (first, second) = bestPair
            val merged = first + second
            val newWord = mutableListOf<String>()
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 && word[i] == first && word[i + 1] == second) {
                    newWord.add(merged)
                    i += 2
                } else {
                    newWord.add(word[i])
                    i++
                }
            }
            word = newWord
            if (word.size == 1) break
        }

        return word.joinToString(" ")
    }

    public companion object {
        /**
         * Load tokenizer from raw file contents (vocab.json text and merges.txt text).
         */
        public fun fromFiles(vocabJson: String, mergesTxt: String): GPT2Tokenizer {
            // Parse vocab.json: {"token": id, ...}
            val json = Json { ignoreUnknownKeys = true }
            val vocabObj = json.decodeFromString<JsonObject>(vocabJson)
            val encoder = mutableMapOf<String, Int>()
            for ((key, value) in vocabObj) {
                encoder[key] = value.jsonPrimitive.int
            }
            val decoder = encoder.entries.associate { (k, v) -> v to k }

            // Parse merges.txt: skip header line, each line is "token1 token2"
            val lines = mergesTxt.lines()
            val bpeRanks = mutableMapOf<Pair<String, String>, Int>()
            var rank = 0
            for (line in lines) {
                if (line.startsWith("#") || line.isBlank()) continue
                val parts = line.split(" ")
                if (parts.size == 2) {
                    bpeRanks[parts[0] to parts[1]] = rank++
                }
            }

            return GPT2Tokenizer(encoder, decoder, bpeRanks)
        }

        // ── GPT-2 byte encoder / decoder ──────────────────────────────────
        // Maps byte values 0-255 to unicode characters, avoiding control chars.

        private val BYTE_ENCODER: Map<Int, Char> = buildMap {
            val bs = mutableListOf<Int>()
            // printable ASCII ranges
            for (b in '!'.code..'~'.code) bs.add(b)
            for (b in '¡'.code..'¬'.code) bs.add(b)
            for (b in '®'.code..'ÿ'.code) bs.add(b)

            val cs = bs.toMutableList()
            var n = 0
            for (b in 0..255) {
                if (b !in bs) {
                    bs.add(b)
                    cs.add(256 + n)
                    n++
                }
            }
            for (i in bs.indices) {
                put(bs[i], cs[i].toChar())
            }
        }

        private val BYTE_DECODER: Map<Char, Byte> = BYTE_ENCODER.entries.associate { (k, v) ->
            v to k.toByte()
        }
    }
}
