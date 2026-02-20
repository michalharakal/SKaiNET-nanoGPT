plugins {
    alias(libs.plugins.jetbrainsKotlinJvm)
    alias(libs.plugins.kotlinSerialization)
    alias(libs.plugins.shadow)
    application
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation(libs.kotlinx.coroutines)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.kotlinx.cli)

    // SKaiNET core -- tensor language, NN layers, slicing DSL, tensor DSL
    implementation(libs.skainet.lang.core)
    implementation(libs.skainet.lang.dag)

    // SKaiNET compile -- autograd tape, graph execution context
    implementation(libs.skainet.compile.core)
    implementation(libs.skainet.compile.dag)

    // SKaiNET backend -- CPU execution with JDK 21 Vector API / SIMD
    implementation(libs.skainet.backend.cpu)

    // SKaiNET I/O -- SafeTensors weight loading
    implementation(libs.skainet.io.core)
    implementation(libs.skainet.io.safetensors)

    testImplementation(libs.kotlin.test)
}

application {
    mainClass.set("sk.ainet.nanogpt.MainKt")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
    maxHeapSize = "8g"
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

tasks.withType<com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar> {
    archiveBaseName.set("nanogpt")
    manifest {
        attributes("Main-Class" to "sk.ainet.nanogpt.MainKt")
    }
    mergeServiceFiles()
}
