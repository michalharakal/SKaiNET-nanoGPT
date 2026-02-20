package sk.ainet.nanogpt.cli

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext

internal actual fun createExecutionContext(): ExecutionContext = DirectCpuExecutionContext()
