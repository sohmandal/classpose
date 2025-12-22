plugins {
    id("maven-publish")
    id("qupath-conventions")
}

qupathExtension {
    name = "qupath-extension-classpose"
    group = "io.github.qupath"
    version = "0.1.0-SNAPSHOT"
    description = "QuPath extension to run Classpose predict-wsi.py and import results"
    automaticModule = "qupath.ext.classpose"
}

dependencies {
    implementation(libs.qupath.gui.fx)
    implementation(libs.qupath.fxtras)
    implementation(libs.extensionmanager)
    implementation("commons-io:commons-io:2.15.0")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.17.1")
}

// Javadoc config (optional)
tasks.withType<Javadoc> {
    (options as StandardJavadocDocletOptions).addBooleanOption("html5", true)
    setDestinationDir(File(project.rootDir, "docs"))
}

// Avoid duplicate handling error when packaging sources
tasks.withType<org.gradle.jvm.tasks.Jar> {
    duplicatesStrategy = DuplicatesStrategy.INCLUDE
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])
            pom {
                licenses {
                    license {
                        name = "Apache License v2.0"
                        url = "http://www.apache.org/licenses/LICENSE-2.0"
                    }
                }
            }
        }
    }
}
