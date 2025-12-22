package qupath.ext.classpose.io;

import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.projects.ProjectImageEntry;
import qupath.lib.roi.interfaces.ROI;
import qupath.lib.roi.ROIs;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.classes.PathClassFactory;
// import qupath.lib.measurements.MeasurementList;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.ArrayList;
import java.util.Iterator;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;

/**
 * Utilities to import predictions from GeoJSON files into the QuPath hierarchy.
 *
 * <p>This importer expects files following a GeoJSON FeatureCollection schema with
 * features of type {@code Polygon} (for contours) or {@code Point} (for centroids).
 * If a feature contains a {@code properties.classification} block with a {@code name}
 * (and optional {@code color: [r,g,b]}), the created annotations are assigned that
 * {@link qupath.lib.objects.classes.PathClass}.</p>
 */
public class GeoJsonImporter {

    /**
     * Import any available outputs into the provided hierarchy.
     *
     * @param hierarchy Target hierarchy (must be non-null to import).
     * @param cellContours Optional path to cell contour FeatureCollection (Polygon features).
     * @param cellCentroids Optional path to cell centroid FeatureCollection (Point features).
     * @param tissueContours Optional path to tissue contour FeatureCollection (Polygon features).
     * @param artefactContours Optional path to artefact contour FeatureCollection (Polygon features).
     * @throws Exception On I/O or parsing errors.
     */
    public static void importOutputs(
            final PathObjectHierarchy hierarchy,
            final File cellContours,
            final File cellCentroids,
            final File tissueContours,
            final File artefactContours
    ) throws Exception {
        if (hierarchy == null) return;
        if (cellContours != null && cellContours.exists())
            addAll(hierarchy, parsePolygons(cellContours));
        if (cellCentroids != null && cellCentroids.exists())
            addAll(hierarchy, parsePoints(cellCentroids));
        if (tissueContours != null && tissueContours.exists())
            addAll(hierarchy, parsePolygons(tissueContours));
        if (artefactContours != null && artefactContours.exists())
            addAll(hierarchy, parsePolygons(artefactContours));
        hierarchy.fireHierarchyChangedEvent(thisOrNull());
    }

    // Helpers stubs (to be implemented)
    /**
     * Parse Polygon features from a GeoJSON file and create annotation objects.
     *
     * @param geojson Input FeatureCollection file containing Polygon geometries.
     * @return List of created {@link PathObject} annotations.
     * @throws IOException If the file cannot be read or parsed.
     */
    private static List<PathObject> parsePolygons(File geojson) throws IOException {
        List<PathObject> out = new ArrayList<>();
        for (JsonNode feature : iterateFeatures(geojson)) {
            JsonNode geometry = feature.get("geometry");
            if (geometry == null || geometry.get("type") == null) continue;
            String type = geometry.get("type").asText("");
            if (!"Polygon".equalsIgnoreCase(type)) continue;

            ArrayNode rings = (ArrayNode) geometry.get("coordinates");
            if (rings == null || rings.size() == 0) continue;
            ArrayNode coords = (ArrayNode) rings.get(0);
            if (coords == null || coords.size() < 3) continue;

            double[] xs = new double[coords.size()];
            double[] ys = new double[coords.size()];
            for (int i = 0; i < coords.size(); i++) {
                ArrayNode p = (ArrayNode) coords.get(i);
                xs[i] = p.get(0).asDouble();
                ys[i] = p.get(1).asDouble();
            }
            ROI roi = ROIs.createPolygonROI(xs, ys, null);
            PathObject obj = createAnnotatedObject(feature, roi);
            if (obj != null) out.add(obj);
        }
        return out;
    }

    /**
     * Parse Point features from a GeoJSON file and create point annotations.
     *
     * @param geojson Input FeatureCollection file containing Point geometries.
     * @return List of created {@link PathObject} point annotations.
     * @throws IOException If the file cannot be read or parsed.
     */
    private static List<PathObject> parsePoints(File geojson) throws IOException {
        List<PathObject> out = new ArrayList<>();
        for (JsonNode feature : iterateFeatures(geojson)) {
            JsonNode geometry = feature.get("geometry");
            if (geometry == null || geometry.get("type") == null) continue;
            String type = geometry.get("type").asText("");
            if (!"Point".equalsIgnoreCase(type)) continue;

            ArrayNode pt = (ArrayNode) geometry.get("coordinates");
            if (pt == null || pt.size() < 2) continue;
            double x = pt.get(0).asDouble();
            double y = pt.get(1).asDouble();
            ROI roi = ROIs.createPointsROI(new double[]{x}, new double[]{y}, null);
            PathObject obj = createAnnotatedObject(feature, roi);
            if (obj != null) out.add(obj);
        }
        return out;
    }

    /**
     * Iterate over features in a GeoJSON FeatureCollection file.
     *
     * @param geojson Path to GeoJSON file.
     * @return Iterable of feature nodes; empty if the file does not contain a FeatureCollection.
     * @throws IOException If the file cannot be read.
     */
    private static Iterable<JsonNode> iterateFeatures(File geojson) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        JsonNode root = mapper.readTree(geojson);
        if (root == null) return List.of();
        if (!"FeatureCollection".equalsIgnoreCase(root.path("type").asText(""))) return List.of();
        ArrayNode features = (ArrayNode) root.get("features");
        if (features == null) return List.of();
        List<JsonNode> list = new ArrayList<>();
        for (JsonNode f : features) list.add(f);
        return list;
    }

    /**
     * Create a {@link PathObject} annotation for a given ROI, applying classification
     * if the feature includes a {@code properties.classification.name} (and optional color).
     */
    private static PathObject createAnnotatedObject(JsonNode feature, ROI roi) {
        if (roi == null) return null;
        PathClass pathClass = null;
        JsonNode props = feature.get("properties");
        if (props != null) {
            JsonNode cls = props.get("classification");
            if (cls != null) {
                String name = cls.path("name").asText(null);
                if (name != null && !name.isBlank()) {
                    // Optional color array [r,g,b]
                    JsonNode colorArr = cls.get("color");
                    if (colorArr != null && colorArr.isArray() && colorArr.size() >= 3) {
                        int r = colorArr.get(0).asInt(0);
                        int g = colorArr.get(1).asInt(0);
                        int b = colorArr.get(2).asInt(0);
                        int rgb = ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
                        pathClass = PathClassFactory.getPathClass(name, rgb);
                    } else {
                        pathClass = PathClassFactory.getPathClass(name);
                    }
                }
            }
        }
        PathObject obj = pathClass != null ? PathObjects.createAnnotationObject(roi, pathClass) : PathObjects.createAnnotationObject(roi);
        // Measurements skipped for compatibility; can be added if API supports it in this QuPath version
        return obj;
    }

    /**
     * Add all given objects to the hierarchy.
     */
    private static void addAll(PathObjectHierarchy hierarchy, List<PathObject> list) {
        if (list == null || list.isEmpty()) return;
        hierarchy.addObjects(list);
    }

    /** Placeholder for an optional event source object. */
    private static PathObjectHierarchy thisOrNull() { return null; }
}
