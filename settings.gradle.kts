pluginManagement {
    repositories {
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
        mavenCentral()
    }
}

rootProject.name = "SKaiNET-nanoGPT"

// Composite build: resolve SKaiNET modules from the local source tree.
// This substitutes Maven Central coordinates with the local project builds.
includeBuild("../SKaiNET")
