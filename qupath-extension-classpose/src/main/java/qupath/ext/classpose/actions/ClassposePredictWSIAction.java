package qupath.ext.classpose.actions;

import qupath.lib.gui.QuPathGUI;
import qupath.ext.classpose.py.PythonRunner;
import qupath.ext.classpose.io.GeoJsonImporter;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.objects.PathObject;
import qupath.lib.roi.interfaces.ROI;
import qupath.lib.objects.classes.PathClass;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import qupath.ext.classpose.ClassposeExtension;
import qupath.ext.classpose.util.AbstractClassposeAction;
import qupath.ext.classpose.util.CliArgs;
import qupath.ext.classpose.util.ImportConventions;
import qupath.ext.classpose.util.PathsUtil;
import qupath.ext.classpose.util.Prefs;

import javafx.concurrent.Task;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import javafx.scene.control.SpinnerValueFactory;
import javafx.scene.layout.HBox;
import javafx.geometry.Pos;
import javafx.geometry.HPos;
import javafx.application.Platform;
import javafx.scene.control.*;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Priority;
import javafx.scene.layout.ColumnConstraints;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import javafx.stage.DirectoryChooser;
import javafx.stage.Stage;

import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.net.URI;
import java.util.Collection;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.prefs.Preferences;

/**
 * Action to run Classpose predict-wsi.py on the current image and import results.
 *
 * <p>If you use any part of Classpose which makes use of GrandQC please follow the instructions 
 * at https://github.com/cpath-ukk/grandqc/tree/main to cite them appropriately. Similarly to 
 * Classpose, GrandQC is under a non-commercial license whose terms can be found at 
 * https://github.com/cpath-ukk/grandqc/blob/main/LICENSE.</p>
 *
 * <p>Collects parameters via a dialog, runs the Python module
 * {@code classpose.entrypoints.predict_wsi}, streams logs live, and imports
 * any generated GeoJSONs back into the current hierarchy. Most UI options are
 * cached across sessions using {@link java.util.prefs.Preferences}.</p>
 */
public class ClassposePredictWSIAction extends AbstractClassposeAction {

    public ClassposePredictWSIAction(final QuPathGUI qupath) {
        super(qupath);
    }

    /**
     * Entry point to run the action. This will later:
     * 1) Collect parameters via a dialog
     * 2) Build CLI for predict-wsi.py
     * 3) Run Python process
     * 4) Import GeoJSON outputs
     */
    public void run() {
        Stage dialog = new Stage();
        dialog.setTitle("Classpose – Predict WSI");

        GridPane grid = new GridPane();
        grid.setHgap(6);
        grid.setVgap(8);
        grid.setPadding(new Insets(12));
        // Define consistent column constraints for alignment: label, control, browse/action
        ColumnConstraints col0 = new ColumnConstraints();
        col0.setMinWidth(Region.USE_COMPUTED_SIZE);
        col0.setPrefWidth(Region.USE_COMPUTED_SIZE);
        col0.setHalignment(HPos.RIGHT);
        ColumnConstraints col1 = new ColumnConstraints();
        col1.setHgrow(Priority.ALWAYS);
        ColumnConstraints col2 = new ColumnConstraints();
        grid.getColumnConstraints().addAll(col0, col1, col2);

        int row = 0;
        TextField tfClassposeDir = new TextField();
        addRow(grid, row++, "Classpose directory", tfClassposeDir, true, () -> browseDirectory(tfClassposeDir, "Select Classpose directory"));
        // Load cached classpose directory if available
        try {
            Preferences prefs = Preferences.userRoot().node("qupath.ext.classpose");
            String cached = prefs.get("classpose_dir", null);
            if (cached != null && !cached.isBlank()) tfClassposeDir.setText(cached);
        } catch (Throwable ignored) {}

        // Model selection: preset dropdown + optional local model path
        final String LOCAL_OPTION = "local model (specify below)";
        ComboBox<String> cbModelChoice = new ComboBox<>();
        cbModelChoice.getItems().addAll(
                "conic",
                "consep",
                "glysac",
                "monusac",
                "nucls",
                "puma",
                LOCAL_OPTION
        );
        // Default selection
        cbModelChoice.getSelectionModel().selectFirst();
        // Add to grid (label + control)
        Label lblModel = new Label("Model *");
        grid.add(lblModel, 0, row);
        grid.add(cbModelChoice, 1, row++);

        final boolean[] nuclsWarningShown = new boolean[] { false };

        TextField tfLocalModel = new TextField();
        addRow(grid, row++, "Local model", tfLocalModel, false, () -> browseFile(tfLocalModel, "Select local model file"));

        // Load cached model choice & local model path if available
        try {
            Preferences prefs = Preferences.userRoot().node("qupath.ext.classpose");
            String cachedChoice = prefs.get("model_choice", null);
            if (cachedChoice != null && !cachedChoice.isBlank() && cbModelChoice.getItems().contains(cachedChoice)) {
                cbModelChoice.getSelectionModel().select(cachedChoice);
            }
            // Prefer new local_model_config; fall back to legacy model_path if present
            String cachedLocal = prefs.get("local_model_config", null);
            if ((cachedLocal == null || cachedLocal.isBlank())) {
                String legacy = prefs.get("model_config", null);
                if (legacy != null && !legacy.isBlank()) cachedLocal = legacy;
            }
            if (cachedLocal != null && !cachedLocal.isBlank()) tfLocalModel.setText(cachedLocal);
        } catch (Throwable ignored) {}

        // Enable local model field only if the local option is selected
        tfLocalModel.setDisable(!LOCAL_OPTION.equals(cbModelChoice.getSelectionModel().getSelectedItem()));
        cbModelChoice.getSelectionModel().selectedItemProperty().addListener((obs, ov, nv) -> {
            boolean useLocal = LOCAL_OPTION.equals(nv);
            tfLocalModel.setDisable(!useLocal);

            if (!nuclsWarningShown[0] && "nucls".equals(nv) && !"nucls".equals(ov)) {
                nuclsWarningShown[0] = true;
                Alert alert = new Alert(Alert.AlertType.WARNING);
                alert.setTitle("Classpose – Disclaimer");
                alert.setHeaderText("'nucls' performance may be subpar");
                alert.setContentText("The 'nucls' preset may produce lower-quality results compared to other models. Take extra caution when using this model and consider selecting another model or using a local model if results are unsatisfactory.");
                alert.showAndWait();
            }
        });


        TextField tfOut = new TextField();
        addRow(grid, row++, "Output folder", tfOut, true, () -> browseDirectory(tfOut, "Select output folder"));
        // Load cached output folder if available
        try {
            Preferences prefs = Preferences.userRoot().node("qupath.ext.classpose");
            String cached = prefs.get("output_folder", null);
            if (cached != null && !cached.isBlank()) tfOut.setText(cached);
        } catch (Throwable ignored) {}

        // Advanced
        CheckBox cbBf16 = new CheckBox("Enable bf16");
        grid.add(cbBf16, 1, row++);

        CheckBox cbTTA = new CheckBox("Enable TTA");
        grid.add(cbTTA, 1, row++);

        CheckBox cbTissue = new CheckBox("Enable tissue detection");
        cbTissue.setSelected(true);
        grid.add(cbTissue, 1, row++);

        CheckBox cbArtefacts = new CheckBox("Enable artefact filtering");
        grid.add(cbArtefacts, 1, row++);

        CheckBox cbROI = new CheckBox("Use selected annotation(s) as ROI");
        grid.add(cbROI, 1, row++);

        CheckBox cbCsv = new CheckBox("Output CSV");
        grid.add(cbCsv, 1, row++);

        CheckBox cbSpatialdata = new CheckBox("Output SpatialData");
        grid.add(cbSpatialdata, 1, row++);

        // Device selection via dropdown and optional GPU ids
        ComboBox<String> cboDevice = new ComboBox<>();
        cboDevice.getItems().addAll("CPU", "GPU", "MPS (Apple Silicon)");
        cboDevice.setEditable(false);
        cboDevice.setValue("CPU");
        // Place combo in the form
        Label lDevice = new Label("Device");
        grid.add(lDevice, 0, row);
        grid.add(cboDevice, 1, row++);

        TextField tfGpuIds = new TextField();
        tfGpuIds.setPromptText("Comma-separated GPU ids, e.g. 0 or 0,1");
        addRow(grid, row++, "GPU ids", tfGpuIds, false, null);

        // Advanced options: Batch size, Tile size (Slider + TextField), Overlap, Min tissue area
        TextField tfBatch = new TextField("8");
        TextField tfTile = new TextField("1024");
        tfTile.setPrefColumnCount(4);
        tfTile.setMaxWidth(60);
        Spinner<Integer> spOverlap = new Spinner<>();
        spOverlap.setValueFactory(new SpinnerValueFactory.IntegerSpinnerValueFactory(0, 4096, 64, 16));
        spOverlap.setEditable(true);
        Spinner<Integer> spMinArea = new Spinner<>();
        spMinArea.setValueFactory(new SpinnerValueFactory.IntegerSpinnerValueFactory(0, 100000000, 100000, 10000));
        spMinArea.setEditable(true);

        // Advanced options inside an Accordion with a TitledPane
        GridPane advGrid = new GridPane();
        advGrid.setHgap(6);
        advGrid.setVgap(6);
        ColumnConstraints a0 = new ColumnConstraints();
        a0.setMinWidth(Region.USE_COMPUTED_SIZE);
        a0.setPrefWidth(Region.USE_COMPUTED_SIZE);
        a0.setHalignment(HPos.RIGHT);
        ColumnConstraints a1 = new ColumnConstraints(); a1.setHgrow(Priority.ALWAYS);
        ColumnConstraints a2 = new ColumnConstraints();
        advGrid.getColumnConstraints().addAll(a0, a1, a2);

        int advRow = 0;
        advGrid.add(new Label("Batch size"), 0, advRow);
        advGrid.add(tfBatch, 1, advRow++);

        // Tile size row with Slider 256–2048 and editable TextField
        advGrid.add(new Label("Tile size"), 0, advRow);
        Slider slTile = new Slider(256, 2048, 1024);
        slTile.setMajorTickUnit(256);
        slTile.setMinorTickCount(3);
        slTile.setShowTickMarks(true);
        slTile.setShowTickLabels(true);
        slTile.setBlockIncrement(64);
        slTile.setSnapToTicks(true);
        slTile.valueProperty().addListener((obs, ov, nv) -> tfTile.setText(Integer.toString((int)Math.round(nv.doubleValue()))));
        tfTile.textProperty().addListener((obs, ov, nv) -> {
            try {
                int val = Integer.parseInt(nv.trim());
                if (val < 256) val = 256;
                if (val > 2048) val = 2048;
                int snapped = (int)Math.round(val / 64.0) * 64;
                if (snapped < 256) snapped = 256;
                if (snapped > 2048) snapped = 2048;
                if ((int)slTile.getValue() != snapped) slTile.setValue(snapped);
            } catch (Exception ignored) {}
        });
        HBox tileBox = new HBox(6, slTile, tfTile);
        tileBox.setAlignment(Pos.CENTER_LEFT);
        advGrid.add(tileBox, 1, advRow++);

        advGrid.add(new Label("Overlap"), 0, advRow);
        advGrid.add(spOverlap, 1, advRow++);

        advGrid.add(new Label("Min tissue area (µm²)"), 0, advRow);
        advGrid.add(spMinArea, 1, advRow++);

        TitledPane advancedPane = new TitledPane("Advanced options", advGrid);
        advancedPane.setExpanded(false);
        // Disable animation to make sizeToScene() adjustments immediate & reliable
        advancedPane.setAnimated(false);
        Accordion accordion = new Accordion(advancedPane);
        accordion.setMaxWidth(Double.MAX_VALUE);
        GridPane.setHgrow(accordion, Priority.ALWAYS);
        accordion.expandedPaneProperty().addListener((o, oldVal, newVal) -> Platform.runLater(dialog::sizeToScene));
        advancedPane.expandedProperty().addListener((o, was, isNow) -> Platform.runLater(dialog::sizeToScene));
        // Also react to content height changes while expanded
        advGrid.heightProperty().addListener((obs, ov, nv) -> {
            if (advancedPane.isExpanded()) Platform.runLater(dialog::sizeToScene);
        });

        // Place accordion above bottom bar
        grid.add(accordion, 0, row++, 3, 1);

        // Compact bottom bar with Run/Cancel only
        Button btnRun = new Button("Run");
        Button btnCancel = new Button("Cancel");
        btnRun.setDefaultButton(true);
        btnCancel.setCancelButton(true);
        btnCancel.setOnAction(e -> dialog.close());
        HBox bottomBar = new HBox(10);
        bottomBar.setAlignment(Pos.CENTER_LEFT);
        bottomBar.getChildren().addAll(btnRun, btnCancel);
        grid.add(bottomBar, 0, row, 3, 1);

        Scene scene = new Scene(grid);
        dialog.setScene(scene);
        dialog.show();
        // Pack once after showing to ensure baseline size is correct
        Platform.runLater(dialog::sizeToScene);

        // Enable/disable CSV/SpatialData based on tissue detection
        cbCsv.setDisable(!cbTissue.isSelected());
        cbSpatialdata.setDisable(!cbTissue.isSelected());
        cbTissue.selectedProperty().addListener((obs, ov, nv) -> {
            cbCsv.setDisable(!nv);
            cbSpatialdata.setDisable(!nv);
        });

        // Load cached UI state
        try {
            String G = "predict";
            cbTTA.setSelected(Prefs.getBoolean(Prefs.k(G, "tta"), cbTTA.isSelected()));
            cbBf16.setSelected(Prefs.getBoolean(Prefs.k(G, "bf16"), cbBf16.isSelected()));
            cbTissue.setSelected(Prefs.getBoolean(Prefs.k(G, "tissue"), cbTissue.isSelected()));
            cbArtefacts.setSelected(Prefs.getBoolean(Prefs.k(G, "artefacts"), cbArtefacts.isSelected()));
            cbROI.setSelected(Prefs.getBoolean(Prefs.k(G, "roi"), false));
            cbCsv.setSelected(Prefs.getBoolean("output_csv", false));
            cbSpatialdata.setSelected(Prefs.getBoolean("output_spatialdata", false));
            String batch = Prefs.getString(Prefs.k(G, "batch"), null); if (batch != null) tfBatch.setText(batch);
            String tile = Prefs.getString(Prefs.k(G, "tile"), null); if (tile != null) tfTile.setText(tile);
            int overlap = Prefs.getInt(Prefs.k(G, "overlap"), spOverlap.getValue()); spOverlap.getValueFactory().setValue(overlap);
            int minArea = Prefs.getInt("min_area", spMinArea.getValue()); spMinArea.getValueFactory().setValue(minArea);
            String gpuIds = Prefs.getString("gpu_ids", null); if (gpuIds != null) tfGpuIds.setText(gpuIds);
            String dv = Prefs.getString("device", null); if (dv != null) cboDevice.setValue(dv);
            String cp = Prefs.getString("classpose_dir", null); if (cp != null) tfClassposeDir.setText(cp);
            // Restore model selection & local model path with legacy fallback
            String cachedChoice = Prefs.getString("model_choice", null);
            if (cachedChoice != null && cbModelChoice.getItems().contains(cachedChoice)) {
                cbModelChoice.getSelectionModel().select(cachedChoice);
            }
            String localPath = Prefs.getString("local_model_path", null);
            if (localPath == null || localPath.isBlank()) {
                String legacy = Prefs.getString("model_path", null);
                if (legacy != null && !legacy.isBlank()) localPath = legacy;
            }
            if (localPath != null && !localPath.isBlank()) tfLocalModel.setText(localPath);
            // Sync enable state of local model field
            boolean useLocalInit = "local model (specify below)".equals(cbModelChoice.getSelectionModel().getSelectedItem());
            tfLocalModel.setDisable(!useLocalInit);
            String out = Prefs.getString("output_folder", null); if (out != null) tfOut.setText(out);
        } catch (Throwable ignored) {}

        btnRun.setOnAction(e -> {
            // Validate required fields
            String selectedModel = cbModelChoice.getSelectionModel().getSelectedItem();
            boolean useLocalModel = LOCAL_OPTION.equals(selectedModel);
            if (isBlank(tfClassposeDir) || isBlank(tfOut) || (useLocalModel && isBlank(tfLocalModel)) || selectedModel == null || selectedModel.isBlank()) {
                showAlert("Please fill all required fields.");
                return;
            }

            if ((cbCsv.isSelected() || cbSpatialdata.isSelected()) && !cbTissue.isSelected()) {
                showAlert("CSV and SpatialData output require tissue detection to be enabled.");
                return;
            }

            String slidePath = resolveCurrentSlidePath();
            if (slidePath == null) {
                showAlert("Could not resolve current slide path. Please open a local WSI.");
                return;
            }

            // Compute fixed model paths inside the extensions folder next to the JAR
            File modelsDir = getModelsDir();
            File tissueModelPath = new File(modelsDir, "grandqc_tissue_model.pt");
            File artefactModelPath = new File(modelsDir, "grandqc_artefact_model.pt");
            // Ensure the directory exists (download can create file later)
            modelsDir.mkdirs();

            // Resolve model argument either from dropdown or local model field
            String modelArg = useLocalModel ? tfLocalModel.getText().trim() : selectedModel.trim();
            CliArgs builder = CliArgs.create()
                    .module("classpose.entrypoints.predict_wsi")
                    .opt("--model_config", modelArg)
                    .opt("--slide_path", slidePath)
                    .opt("--tissue_detection_model_path", cbTissue.isSelected() ? tissueModelPath.getAbsolutePath() : null)
                    .opt("--artefact_detection_model_path", cbArtefacts.isSelected() ? artefactModelPath.getAbsolutePath() : null)
                    .opt("--output_folder", tfOut.getText().trim());

            List<String> outputTypes = new ArrayList<>();
            if (cbCsv.isSelected()) outputTypes.add("csv");
            if (cbSpatialdata.isSelected()) outputTypes.add("spatialdata");
            if (!outputTypes.isEmpty()) {
                builder.opt("--output_type", String.join(" ", outputTypes));
            }

            // If ROI mode enabled, export selected annotations to GeoJSON and pass to Python
            if (cbROI.isSelected()) {
                try {
                    File roiFile = ImportConventions.roiGeoJSON(new File(tfOut.getText().trim()), slidePath);
                    boolean ok = exportSelectedAnnotationsAsGeoJSON(roiFile);
                    if (!ok) {
                        showAlert("ROI mode enabled but no valid polygon annotations are selected.");
                        return;
                    }
                    builder.opt("--roi_geojson", roiFile.getAbsolutePath());
                } catch (Exception ex) {
                    showAlert("Failed to export ROI GeoJSON: " + ex.getMessage());
                    return;
                }
            }

            if (cbTTA.isSelected()) builder.flag("--tta");
            if (cbBf16.isSelected()) builder.flag("--bf16");
            // Compute Python device string
            String deviceSelection = cboDevice.getSelectionModel().getSelectedItem();
            String deviceKind = null;
            if (deviceSelection != null) {
                if (deviceSelection.startsWith("CPU")) deviceKind = "cpu";
                else if (deviceSelection.startsWith("GPU")) deviceKind = "cuda";
                else if (deviceSelection.startsWith("MPS")) deviceKind = "mps";
            }
            String deviceValue = null;
            if (deviceKind != null) {
                if ("cuda".equals(deviceKind)) {
                    String ids = tfGpuIds.getText() != null ? tfGpuIds.getText().trim() : "";
                    if (!ids.isEmpty()) deviceValue = deviceKind + ":" + ids;
                    else deviceValue = deviceKind;
                } else {
                    deviceValue = deviceKind;
                }
            }
            builder
                .opt("--device", deviceValue)
                .opt("--batch_size", tfBatch.getText().trim())
                .opt("--tile_size", tfTile.getText().trim())
                .opt("--overlap", Integer.toString(spOverlap.getValue()))
                .opt("--min_area", Integer.toString(spMinArea.getValue()));

            List<String> args = new ArrayList<>(builder.build());

            // Cache all options for next runs
            try {
                String G = "predict";
                Prefs.putString("classpose_dir", tfClassposeDir.getText().trim());
                // Persist model selection and local model path
                if (selectedModel != null) Prefs.putString("model_choice", selectedModel);
                if (tfLocalModel.getText() != null) Prefs.putString("local_model_path", tfLocalModel.getText().trim());
                // Keep legacy key updated with the resolved model argument for backward compatibility
                Prefs.putString("model_path", modelArg);
                Prefs.putString("output_folder", tfOut.getText().trim());
                Prefs.putBoolean(Prefs.k(G, "tta"), cbTTA.isSelected());
                Prefs.putBoolean(Prefs.k(G, "bf16"), cbBf16.isSelected());
                Prefs.putBoolean(Prefs.k(G, "tissue"), cbTissue.isSelected());
                Prefs.putBoolean(Prefs.k(G, "artefacts"), cbArtefacts.isSelected());
                Prefs.putBoolean(Prefs.k(G, "roi"), cbROI.isSelected());
                Prefs.putBoolean("output_csv", cbCsv.isSelected());
                Prefs.putBoolean("output_spatialdata", cbSpatialdata.isSelected());
                // Store the label directly for easier restoration
                String selectedLabel = cboDevice.getSelectionModel().getSelectedItem();
                if (selectedLabel != null) Prefs.putString("device", selectedLabel);
                if (tfGpuIds.getText() != null) Prefs.putString("gpu_ids", tfGpuIds.getText().trim());
                if (!isBlank(tfBatch)) Prefs.putString(Prefs.k(G, "batch"), tfBatch.getText().trim());
                if (!isBlank(tfTile)) Prefs.putString(Prefs.k(G, "tile"), tfTile.getText().trim());
                Prefs.putInt(Prefs.k(G, "overlap"), spOverlap.getValue());
                Prefs.putInt("min_area", spMinArea.getValue());
            } catch (Throwable ignored) {}

            // Run in background
            long startTs = System.currentTimeMillis();
            dialog.close();
            runPythonWithLogging(tfClassposeDir.getText().trim(), new File(tfOut.getText().trim()), args, startTs);
        });
    }



    /**
     * Export currently selected polygon annotations as a GeoJSON FeatureCollection (level-0 coords).
     * Returns true if at least one polygon feature was written.
     */
    private boolean exportSelectedAnnotationsAsGeoJSON(File outFile) throws IOException {
        var imgData = qupath.getImageData();
        if (imgData == null) return false;
        var hier = imgData.getHierarchy();
        if (hier == null) return false;
        var selected = hier.getSelectionModel().getSelectedObjects();
        if (selected == null || selected.isEmpty()) return false;

        ObjectMapper mapper = new ObjectMapper();
        ObjectNode fc = mapper.createObjectNode();
        fc.put("type", "FeatureCollection");
        ArrayNode features = mapper.createArrayNode();

        int count = 0;
        for (PathObject po : selected) {
            if (po == null || !po.isAnnotation()) continue;
            ROI roi = po.getROI();
            if (roi == null) continue;

            // Extract one or more subpaths from ROI shape and export each as a Polygon feature
            java.awt.Shape shape = roi.getShape();
            if (shape == null) continue;

            java.awt.geom.PathIterator it = shape.getPathIterator(null, 1.0);
            java.util.List<double[]> current = new java.util.ArrayList<>();
            double[] coords = new double[6];
            while (!it.isDone()) {
                int seg = it.currentSegment(coords);
                switch (seg) {
                    case java.awt.geom.PathIterator.SEG_MOVETO:
                        // start a new ring
                        if (!current.isEmpty()) {
                            addPolygonFeatureFromRing(mapper, features, po, current);
                            count++;
                            current = new java.util.ArrayList<>();
                        }
                        current.add(new double[]{coords[0], coords[1]});
                        break;
                    case java.awt.geom.PathIterator.SEG_LINETO:
                        current.add(new double[]{coords[0], coords[1]});
                        break;
                    case java.awt.geom.PathIterator.SEG_CLOSE:
                        // finalize current ring
                        if (!current.isEmpty()) {
                            addPolygonFeatureFromRing(mapper, features, po, current);
                            count++;
                            current = new java.util.ArrayList<>();
                        }
                        break;
                    default:
                        // For curves, approximate by line to coordinate
                        current.add(new double[]{coords[0], coords[1]});
                }
                it.next();
            }
            // Flush leftover ring if any
            if (!current.isEmpty()) {
                addPolygonFeatureFromRing(mapper, features, po, current);
                count++;
            }
        }

        if (count == 0) return false;
        fc.set("features", features);
        outFile.getParentFile().mkdirs();
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(outFile))) {
            bw.write(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(fc));
        }
        return true;
    }

    private String basenameFromPath(String path) {
        String base = new File(path).getName();
        int dot = base.lastIndexOf('.');
        return dot > 0 ? base.substring(0, dot) : base;
    }

    private void addPolygonFeatureFromRing(ObjectMapper mapper, ArrayNode features, PathObject po, java.util.List<double[]> ring) {
        if (ring == null || ring.size() < 3) return;
        // Build outer ring array and close it
        ArrayNode outer = mapper.createArrayNode();
        for (double[] p : ring) {
            ArrayNode xy = mapper.createArrayNode();
            xy.add(p[0]);
            xy.add(p[1]);
            outer.add(xy);
        }
        // close ring
        double[] p0 = ring.get(0);
        ArrayNode xy0 = mapper.createArrayNode();
        xy0.add(p0[0]); xy0.add(p0[1]);
        outer.add(xy0);

        ArrayNode coords = mapper.createArrayNode();
        coords.add(outer);

        ObjectNode geom = mapper.createObjectNode();
        geom.put("type", "Polygon");
        geom.set("coordinates", coords);

        ObjectNode props = mapper.createObjectNode();
        PathClass pc = po.getPathClass();
        if (pc != null) {
            ObjectNode cls = mapper.createObjectNode();
            cls.put("name", pc.getName());
            props.set("classification", cls);
        }

        ObjectNode feat = mapper.createObjectNode();
        feat.put("type", "Feature");
        feat.set("geometry", geom);
        feat.set("properties", props);
        features.add(feat);
    }



    private void runPythonWithLogging(String classposeDir, File workingDir, List<String> args, long startTs) {
        PythonRunner runner = new PythonRunner();

        // Create a live log window
        Stage logStage = new Stage();
        logStage.setTitle("Classpose – Inference Log");
        TextArea ta = new TextArea();
        ta.setEditable(false);
        ta.setWrapText(false);
        Button btnCancel = new Button("Cancel");
        VBox vbox = new VBox(ta, btnCancel);
        VBox.setVgrow(ta, Priority.ALWAYS);
        logStage.setScene(new Scene(vbox, 900, 500));
        logStage.show();

        // Prepare log file in output folder
        File logFile = new File(workingDir, "classpose_predict.log");
        final BufferedWriter[] writerRef = new BufferedWriter[1];
        try {
            writerRef[0] = new BufferedWriter(new FileWriter(logFile, true));
            writerRef[0].write("Command: uv run --project " + classposeDir + " " + String.join(" ", args) + "\n\n");
            writerRef[0].flush();
        } catch (IOException ioe) {
            // ignore file errors, continue with UI-only logging
        }

        Consumer<String> append = line -> {
            javafx.application.Platform.runLater(() -> {
                ta.appendText(line + "\n");
            });
            if (writerRef[0] != null) {
                try {
                    writerRef[0].write(line + "\n");
                } catch (IOException ignored) {}
            }
        };

        Consumer<String> logOut = line -> append.accept("[OUT] " + line);
        Consumer<String> logErr = line -> append.accept("[ERR] " + line);

        final Process[] procRef = new Process[1];

        Task<Void> task = new Task<>() {
            @Override
            protected Void call() throws Exception {
                procRef[0] = runner.start(classposeDir, workingDir, null, args, logOut, logErr);
                int code = procRef[0].waitFor();
                if (writerRef[0] != null) {
                    try { writerRef[0].flush(); writerRef[0].close(); } catch (IOException ignored) {}
                }
                if (code != 0) {
                    throw new RuntimeException("Python exited with code " + code + ". See log: " + logFile.getAbsolutePath());
                }
                return null;
            }
        };

        btnCancel.setOnAction(ev -> {
            try {
                runner.kill(procRef[0]);
                btnCancel.setDisable(true);
                ta.appendText("[INFO] Cancel requested by user.\n");
            } catch (Throwable ignored) {}
        });

        task.setOnFailed(e -> {
            Throwable ex = task.getException();
            showAlert("Prediction failed: " + (ex != null ? ex.getMessage() : "Unknown error"));
            // Keep the log window open for inspection
        });
        task.setOnSucceeded(e -> {
            try {
                // Close log window on success
                logStage.close();

                String slidePath = args.get(args.indexOf("--slide_path") + 1);
                String outFolder = args.get(args.indexOf("--output_folder") + 1);
                File cellContours = ImportConventions.cellContours(new File(outFolder), slidePath);
                File cellCentroids = ImportConventions.cellCentroids(new File(outFolder), slidePath);
                File tissueContours = ImportConventions.tissueContours(new File(outFolder), slidePath);
                File artefactContours = ImportConventions.artefactContours(new File(outFolder), slidePath);

                var imgData = qupath.getImageData();
                if (imgData != null) {
                    PathObjectHierarchy hier = imgData.getHierarchy();
                    // Only import files generated in this run
                    if (cellContours.exists() && cellContours.lastModified() >= startTs) {
                        GeoJsonImporter.importOutputs(hier, cellContours, null, null, null);
                    }
                    if (cellCentroids.exists() && cellCentroids.lastModified() >= startTs) {
                        GeoJsonImporter.importOutputs(hier, null, cellCentroids, null, null);
                    }
                    if (tissueContours.exists() && tissueContours.lastModified() >= startTs) {
                        GeoJsonImporter.importOutputs(hier, null, null, tissueContours, null);
                    }
                    if (artefactContours.exists() && artefactContours.lastModified() >= startTs) {
                        GeoJsonImporter.importOutputs(hier, null, null, null, artefactContours);
                    }
                }

                String additionalOutputs = "";
                File csvFile = new File(outFolder, basenameFromPath(slidePath) + "_cell_densities.csv");
                if (csvFile.exists() && csvFile.lastModified() >= startTs) {
                    additionalOutputs += "CSV: " + csvFile.getAbsolutePath() + "\n";
                }
                File zarrFile = new File(outFolder, basenameFromPath(slidePath) + "_spatialdata.zarr");
                if (zarrFile.exists() && zarrFile.lastModified() >= startTs) {
                    additionalOutputs += "SpatialData Zarr: " + zarrFile.getAbsolutePath() + "\n";
                }

                String message = "Prediction completed and results imported.";
                if (!additionalOutputs.isEmpty()) {
                    message += "\n\nAdditional outputs:\n" + additionalOutputs;
                }
                Alert ok = new Alert(Alert.AlertType.INFORMATION, message, ButtonType.OK);
                ok.initOwner(qupath.getStage());
                ok.showAndWait();
            } catch (Exception ex) {
                showAlert("Completed but failed to import results: " + ex.getMessage());
            }
        });

        Thread th = new Thread(task, "ClassposePredictWSI");
        th.setDaemon(true);
        th.start();
    }




}
