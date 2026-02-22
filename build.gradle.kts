import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.kotlinSerialization)
    alias(libs.plugins.shadow)
}

kotlin {
    jvmToolchain(21)

    jvm()

    js {
        browser()
        nodejs()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
        binaries.executable()
    }

    macosArm64 {
        binaries {
            executable {
                entryPoint = "sk.ainet.nanogpt.cli.main"
                baseName = "nanogpt"
            }
        }
    }

    linuxX64 {
        binaries {
            executable {
                entryPoint = "sk.ainet.nanogpt.cli.main"
                baseName = "nanogpt"
            }
        }
    }

    linuxArm64 {
        binaries {
            executable {
                entryPoint = "sk.ainet.nanogpt.cli.main"
                baseName = "nanogpt"
            }
        }
    }

    @Suppress("OPT_IN_USAGE")
    applyDefaultHierarchyTemplate {
        common {
            group("native") {
                group("macos") {
                    withMacosArm64()
                }
                group("linux") {
                    withLinuxX64()
                    withLinuxArm64()
                }
            }
        }
    }

    sourceSets {
        commonMain.dependencies {
            implementation(kotlin("stdlib"))
            implementation(libs.kotlinx.coroutines)
            implementation(libs.kotlinx.serialization.json)

            // SKaiNET core -- tensor language, NN layers, slicing DSL, tensor DSL
            implementation(libs.skainet.lang.core)
            implementation(libs.skainet.lang.dag)

            // SKaiNET compile -- autograd tape, graph execution context
            implementation(libs.skainet.compile.core)
            implementation(libs.skainet.compile.dag)

            // SKaiNET backend -- CPU execution
            implementation(libs.skainet.backend.cpu)

            // SKaiNET I/O -- SafeTensors weight loading
            implementation(libs.skainet.io.core)
            implementation(libs.skainet.io.safetensors)
        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }

        val jvmMain by getting {
            dependencies {
                implementation(libs.kotlinx.cli)
            }
        }

        val nativeMain by getting {
            dependencies {
                implementation(libs.kotlinx.io.core)
            }
        }

        val wasmJsMain by getting {
            dependencies {
                implementation(libs.kotlinx.browser)
            }
        }
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
    maxHeapSize = "8g"
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

// Shadow jar for JVM fat-jar distribution
tasks.withType<com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar> {
    archiveBaseName.set("nanogpt")
    manifest {
        attributes("Main-Class" to "sk.ainet.nanogpt.MainKt")
    }
    mergeServiceFiles()
}
