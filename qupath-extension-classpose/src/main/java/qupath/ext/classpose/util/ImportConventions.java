package qupath.ext.classpose.util;

import java.io.File;

/**
 * Centralizes naming for output files produced by Python entrypoints.
 */
public final class ImportConventions {
    private static final String BASE_NAME_TOKEN = "{base_name}";
    private static final String CELL_CONTOURS_ENV = "CLASSPOSE_CELL_CONTOURS_GEOJSON";
    private static final String CELL_CENTROIDS_ENV = "CLASSPOSE_CELL_CENTROIDS_GEOJSON";
    private static final String TISSUE_CONTOURS_ENV = "CLASSPOSE_TISSUE_CONTOURS_GEOJSON";
    private static final String ARTEFACT_CONTOURS_ENV = "CLASSPOSE_ARTEFACT_CONTOURS_GEOJSON";
    private static final String ROI_GEOJSON_ENV = "CLASSPOSE_ROI_GEOJSON";
    private static final String CELL_CONTOURS_DEFAULT = "{base_name}_cell_contours.geojson";
    private static final String CELL_CENTROIDS_DEFAULT = "{base_name}_cell_centroids.geojson";
    private static final String TISSUE_CONTOURS_DEFAULT = "{base_name}_tissue_contours.geojson";
    private static final String ARTEFACT_CONTOURS_DEFAULT = "{base_name}_artefact_contours.geojson";
    private static final String ROI_GEOJSON_DEFAULT = "{base_name}_roi.geojson";

    private ImportConventions() {}

    public static File cellContours(File outFolder, String slidePath) {
        return buildFile(outFolder, PathsUtil.basename(slidePath), CELL_CONTOURS_ENV, CELL_CONTOURS_DEFAULT);
    }

    public static File cellCentroids(File outFolder, String slidePath) {
        return buildFile(outFolder, PathsUtil.basename(slidePath), CELL_CENTROIDS_ENV, CELL_CENTROIDS_DEFAULT);
    }

    public static File tissueContours(File outFolder, String slidePath) {
        return buildFile(outFolder, PathsUtil.basename(slidePath), TISSUE_CONTOURS_ENV, TISSUE_CONTOURS_DEFAULT);
    }

    public static File tissueContoursFromPrefix(String outputPrefix) {
        return buildFromPrefix(outputPrefix, TISSUE_CONTOURS_ENV, TISSUE_CONTOURS_DEFAULT);
    }

    public static File artefactContours(File outFolder, String slidePath) {
        return buildFile(outFolder, PathsUtil.basename(slidePath), ARTEFACT_CONTOURS_ENV, ARTEFACT_CONTOURS_DEFAULT);
    }

    public static File artefactContoursFromPrefix(String outputPrefix) {
        return buildFromPrefix(outputPrefix, ARTEFACT_CONTOURS_ENV, ARTEFACT_CONTOURS_DEFAULT);
    }

    public static File roiGeoJSON(File outFolder, String slidePath) {
        return buildFile(outFolder, PathsUtil.basename(slidePath), ROI_GEOJSON_ENV, ROI_GEOJSON_DEFAULT);
    }

    private static File buildFromPrefix(String outputPrefix, String envKey, String defaultTemplate) {
        File prefixFile = new File(outputPrefix).getAbsoluteFile();
        return buildFile(prefixFile.getParentFile(), prefixFile.getName(), envKey, defaultTemplate);
    }

    private static File buildFile(File outFolder, String baseName, String envKey, String defaultTemplate) {
        String template = readTemplate(envKey, defaultTemplate);
        return new File(outFolder, template.replace(BASE_NAME_TOKEN, baseName));
    }

    private static String readTemplate(String envKey, String defaultTemplate) {
        String value = System.getenv(envKey);
        if (value == null || value.isBlank()) {
            return defaultTemplate;
        }
        return value.replace("\"", "");
    }
}
