package qupath.ext.classpose.util;

import java.io.File;

/**
 * Centralizes naming for output files produced by Python entrypoints.
 */
public final class ImportConventions {
    private ImportConventions() {}

    public static File cellContours(File outFolder, String slidePath) {
        return new File(outFolder, PathsUtil.basename(slidePath) + "_cell_contours.geojson");
    }

    public static File cellCentroids(File outFolder, String slidePath) {
        return new File(outFolder, PathsUtil.basename(slidePath) + "_cell_centroids.geojson");
    }

    public static File tissueContours(File outFolder, String slidePath) {
        return new File(outFolder, PathsUtil.basename(slidePath) + "_tissue_contours.geojson");
    }

    public static File artefactContours(File outFolder, String slidePath) {
        return new File(outFolder, PathsUtil.basename(slidePath) + "_artefact_contours.geojson");
    }

    public static File roiGeoJSON(File outFolder, String slidePath) {
        return new File(outFolder, PathsUtil.basename(slidePath) + "_roi.geojson");
    }
}
