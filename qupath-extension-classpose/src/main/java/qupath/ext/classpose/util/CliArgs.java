package qupath.ext.classpose.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Small helper to build CLI argument lists safely.
 */
public class CliArgs {

    private final List<String> args = new ArrayList<>();

    public static CliArgs create() { return new CliArgs(); }

    public CliArgs module(String module) {
        if (module != null && !module.isBlank()) {
            args.add("-m");
            args.add(module);
        }
        return this;
    }

    public CliArgs flag(String flag) {
        if (flag != null && !flag.isBlank()) args.add(flag);
        return this;
    }

    public CliArgs opt(String name, String value) {
        if (name != null && !name.isBlank() && value != null && !value.isBlank()) {
            args.add(name);
            args.add(value);
        }
        return this;
    }

    public CliArgs opt(String name, int value) {
        return opt(name, Integer.toString(value));
    }

    public List<String> build() { return Collections.unmodifiableList(new ArrayList<>(args)); }
}
