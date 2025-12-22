package qupath.ext.classpose.py;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Minimal Python process runner. Executes the specified Python interpreter with arguments,
 * forwards stdout and stderr to provided consumers, and returns the exit code.
 */
public class PythonRunner {

    /**
     * Start a Python process asynchronously.
     *
     * @return the started Process (caller is responsible for waiting/cancelling)
     */
    public Process start(
            final String classposeDir,
            final File workingDir,
            final Map<String, String> env,
            final List<String> args,
            final Consumer<String> onStdout,
            final Consumer<String> onStderr
    ) throws IOException {
        if (classposeDir == null || classposeDir.isBlank())
            throw new IllegalArgumentException("Classpose directory is required");
        // Preprocess args to handle --output_type with spaces
        List<String> processedArgs = preprocessArgs(args);
        final List<String> command = new ArrayList<>();
        command.add("uv");
        command.add("run");
        command.add("--project");
        command.add(classposeDir);
        if (processedArgs != null)
            command.addAll(processedArgs);

        final ProcessBuilder pb = new ProcessBuilder(command);

        if (workingDir != null) {
            if (!workingDir.exists()) {
                if (!workingDir.mkdirs()) {
                    throw new IOException("Failed to create working directory: " + workingDir.getAbsolutePath());
                }
            } else if (!workingDir.isDirectory()) {
                throw new IOException("Working directory is not a directory: " + workingDir.getAbsolutePath());
            }
            pb.directory(workingDir);
        }
        if (env != null)
            pb.environment().putAll(env);
        String existingPath = pb.environment().getOrDefault("PATH", System.getenv("PATH"));
        String effectivePath = extendPath(existingPath);
        pb.environment().put("PATH", effectivePath);
        String uvExec = findOnPath("uv", effectivePath);
        if (uvExec != null) {
            command.set(0, uvExec);
        } 
        pb.redirectErrorStream(false);

        final Process process = pb.start();

        Thread tOut = new Thread(() -> streamLines(process.getInputStream(), onStdout), "py-stdout");
        Thread tErr = new Thread(() -> streamLines(process.getErrorStream(), onStderr), "py-stderr");
        tOut.setDaemon(true);
        tErr.setDaemon(true);
        tOut.start();
        tErr.start();

        return process;
    }

    private static List<String> preprocessArgs(List<String> args) {
        if (args == null) return null;
        List<String> result = new ArrayList<>();
        for (int i = 0; i < args.size(); i++) {
            String arg = args.get(i);
            if ("--output_type".equals(arg) && i + 1 < args.size()) {
                String value = args.get(i + 1);
                if (value.contains(" ")) {
                    // Split the value by spaces and add as separate args
                    String[] parts = value.split("\\s+");
                    result.add(arg);
                    for (String part : parts) {
                        if (!part.trim().isEmpty()) {
                            result.add(part.trim());
                        }
                    }
                    i++; // Skip the next arg since we processed it
                } else {
                    result.add(arg);
                }
            } else {
                result.add(arg);
            }
        }
        return result;
    }

    /**
     * Run a Python process.
     *
     * @param classposeDir Absolute path to the Classpose directory.
     * @param workingDir Working directory for the process (may be null).
     * @param env Additional environment variables (may be null).
     * @param args Full argument list, including the script path and its args.
     * @param onStdout Consumer for stdout lines (may be null).
     * @param onStderr Consumer for stderr lines (may be null).
     * @return Exit code of the process.
     * @throws IOException If the process cannot be started.
     * @throws InterruptedException If the process is interrupted while waiting.
     */
    public int run(
            final String classposeDir,
            final File workingDir,
            final Map<String, String> env,
            final List<String> args,
            final Consumer<String> onStdout,
            final Consumer<String> onStderr
    ) throws IOException, InterruptedException {
        Process process = start(classposeDir, workingDir, env, args, onStdout, onStderr);
        return process.waitFor();
    }

    private static void streamLines(java.io.InputStream is, Consumer<String> consumer) {
        if (consumer == null) return;
        try (BufferedReader br = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                consumer.accept(line);
            }
        } catch (IOException ignored) {
        }
    }

    /** Cancel a running process (best effort). */
    public void kill(Process p) {
        if (p == null) return;
        try {
            p.destroy();
            try { Thread.sleep(500); } catch (InterruptedException ignored) {}
            if (p.isAlive()) p.destroyForcibly();
        } catch (Throwable ignored) {}
    }

    private static String extendPath(String path) {
        String base = path == null ? "" : path;
        String home = System.getProperty("user.home");
        String[] extras = new String[] {
            "/opt/homebrew/bin",
            "/usr/local/bin",
            home == null ? null : home + "/.local/bin",
            home == null ? null : home + "/.cargo/bin"
        };
        String result = base;
        for (String e : extras) {
            if (e == null) continue;
            if (!containsPathSegment(result, e)) {
                if (!result.isEmpty()) result += File.pathSeparator;
                result += e;
            }
        }
        return result;
    }

    private static boolean containsPathSegment(String path, String segment) {
        if (path == null || segment == null) return false;
        String[] parts = path.split(java.util.regex.Pattern.quote(File.pathSeparator));
        for (String p : parts) {
            if (segment.equals(p)) return true;
        }
        return false;
    }

    private static String findOnPath(String exe, String path) {
        if (exe == null || exe.isEmpty()) return null;
        String[] parts = path == null ? new String[0] : path.split(java.util.regex.Pattern.quote(File.pathSeparator));
        for (String dir : parts) {
            if (dir == null || dir.isEmpty()) continue;
            File f = new File(dir, exe);
            if (isExecutable(f)) return f.getAbsolutePath();
        }
        return null;
    }

    private static boolean isExecutable(File f) {
        return f != null && f.exists() && f.isFile() && f.canExecute();
    }
}
