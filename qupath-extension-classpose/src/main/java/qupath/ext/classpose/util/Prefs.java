package qupath.ext.classpose.util;

import java.util.prefs.Preferences;

/**
 * Centralized preferences helper using a single node for the extension.
 * Provides typed get/put helpers and simple grouping via key prefixes.
 */
public final class Prefs {

    private static final String NODE = "qupath.ext.classpose";
    private static final Preferences PREFS = Preferences.userRoot().node(NODE);

    private Prefs() {}

    public static String k(String group, String key) {
        return group == null || group.isBlank() ? key : group + "_" + key;
    }

    public static void putString(String key, String value) {
        if (value == null) return;
        PREFS.put(key, value);
    }

    public static String getString(String key, String defaultValue) {
        return PREFS.get(key, defaultValue);
    }

    public static void putBoolean(String key, boolean value) {
        PREFS.putBoolean(key, value);
    }

    public static boolean getBoolean(String key, boolean defaultValue) {
        return PREFS.getBoolean(key, defaultValue);
    }

    public static void putInt(String key, int value) {
        PREFS.putInt(key, value);
    }

    public static int getInt(String key, int defaultValue) {
        return PREFS.getInt(key, defaultValue);
    }
}
