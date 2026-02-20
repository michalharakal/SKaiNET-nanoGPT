package sk.ainet.nanogpt.cli

import sk.ainet.io.RandomAccessSource

internal class ByteArrayRandomAccessSource(private val data: ByteArray) : RandomAccessSource {
    override val size: Long = data.size.toLong()

    override fun readAt(position: Long, length: Int): ByteArray {
        require(position >= 0) { "Position must be non-negative" }
        require(length >= 0) { "Length must be non-negative" }
        require(position + length <= size) { "Read beyond end of data" }
        return data.copyOfRange(position.toInt(), (position + length).toInt())
    }

    override fun readAt(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
        val available = minOf(length, (size - position).toInt())
        data.copyInto(buffer, offset, position.toInt(), position.toInt() + available)
        return available
    }

    override fun close() {}
}
