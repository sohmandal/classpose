package qupath.ext.classpose.util;

import java.io.File;

/** Utility helpers for common path/basename operations. */
public final class PathsUtil {
    private PathsUtil() {}

    /** Return file basename without extension. */
    public static String basename(String path) {
        if (path == null || path.isBlank()) return path;
        String base = new File(path).getName();
        int dot = base.lastIndexOf('.');
        return dot > 0 ? base.substring(0, dot) : base;
    }

    /** Ensure directory exists; return the directory File for chaining. */
    public static File ensureDir(File dir) {
        if (dir != null) dir.mkdirs();
        return dir;
    }

    /** Build an output prefix file: &lt;outDir&gt;/&lt;basename(slidePath)&gt;. */
    public static File outPrefix(File outDir, String slidePath) {
        String base = basename(slidePath);
        return new File(outDir, base);
    }
}
