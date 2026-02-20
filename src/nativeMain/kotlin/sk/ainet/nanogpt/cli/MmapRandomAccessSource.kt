package sk.ainet.nanogpt.cli

import kotlinx.cinterop.CPointer
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.get
import kotlinx.cinterop.plus
import kotlinx.cinterop.readBytes
import platform.posix.MAP_PRIVATE
import platform.posix.PROT_READ
import platform.posix.close
import platform.posix.fstat
import platform.posix.mmap
import platform.posix.munmap
import platform.posix.open
import platform.posix.stat
import platform.posix.O_RDONLY
import platform.posix.MAP_FAILED
import sk.ainet.io.RandomAccessSource
import kotlinx.cinterop.alloc
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.toLong

/**
 * Memory-mapped [RandomAccessSource] using POSIX mmap.
 *
 * Maps the entire file into virtual memory, letting the OS page in data
 * on demand. No heap copy — the kernel handles caching and eviction.
 * Works on both macOS and Linux.
 */
@OptIn(ExperimentalForeignApi::class)
internal class MmapRandomAccessSource private constructor(
    private val ptr: CPointer<ByteVar>,
    override val size: Long,
    private val fd: Int
) : RandomAccessSource {

    override fun readAt(position: Long, length: Int): ByteArray {
        require(position >= 0) { "Position must be non-negative" }
        require(length >= 0) { "Length must be non-negative" }
        require(position + length <= size) { "Read beyond end of data: pos=$position len=$length size=$size" }
        return (ptr + position.toInt())!!.readBytes(length)
    }

    override fun readAt(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
        val available = minOf(length, (size - position).toInt())
        val src = readAt(position, available)
        src.copyInto(buffer, offset)
        return available
    }

    override fun close() {
        munmap(ptr, size.toULong())
        close(fd)
    }

    companion object {
        fun open(filePath: String): MmapRandomAccessSource = memScoped {
            val fd = open(filePath, O_RDONLY)
            if (fd < 0) error("Cannot open file: $filePath")

            val st = alloc<stat>()
            if (fstat(fd, st.ptr) != 0) {
                close(fd)
                error("Cannot stat file: $filePath")
            }
            val fileSize = st.st_size

            val mapped = mmap(null, fileSize.toULong(), PROT_READ, MAP_PRIVATE, fd, 0)
            if (mapped == MAP_FAILED) {
                close(fd)
                error("mmap failed for: $filePath ($fileSize bytes)")
            }

            MmapRandomAccessSource(
                ptr = mapped!!.reinterpret(),
                size = fileSize,
                fd = fd
            )
        }
    }
}
