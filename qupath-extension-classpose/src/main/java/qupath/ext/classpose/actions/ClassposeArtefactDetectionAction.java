package qupath.ext.classpose.actions;

import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.stage.Stage;
import qupath.ext.classpose.io.GeoJsonImporter;
import qupath.ext.classpose.py.PythonRunner;
import qupath.ext.classpose.util.AbstractClassposeAction;
import qupath.ext.classpose.util.Prefs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;

import java.io.*;
import java.util.*;
import java.util.function.Consumer;
import java.util.prefs.Preferences;

/**
 * Action to run GrandQC artefact detection on the current image and import results.
 *
 * <p>Collects parameters via a dialog, runs {@code classpose.grandqc.wsi_artefact_detection},
 * streams logs live, and imports generated artefact GeoJSON. UI fields are cached across
 * sessions using {@link java.util.prefs.Preferences}.</p>
 */
public class ClassposeArtefactDetectionAction extends AbstractClassposeAction {

    public ClassposeArtefactDetectionAction(final QuPathGUI qupath) {
        super(qupath);
    }

    public void run() {
        Stage dialog = new Stage();
        dialog.setTitle("Classpose – Artefact detection");

        GridPane grid = new GridPane();
        grid.setHgap(8);
        grid.setVgap(8);
        grid.setPadding(new Insets(12));

        int row = 0;
        TextField tfClassposeDir = new TextField();
        addRow(grid, row++, "Classpose directory", tfClassposeDir, true, () -> browseDirectory(tfClassposeDir, "Select Classpose directory"));
        // Load cached classpose directory if available
        try {
            Preferences prefs = Preferences.userRoot().node("qupath.ext.classpose");
            String cached = prefs.get("classpose_dir", null);
            if (cached != null && !cached.isBlank()) tfClassposeDir.setText(cached);
        } catch (Throwable ignored) {}

        TextField tfOut = new TextField();
        addRow(grid, row++, "Output folder", tfOut, true, () -> browseDirectory(tfOut, "Select output folder"));

        TextField tfMppArt = new TextField("1.0");
        addRow(grid, row++, "Artefact model MPP", tfMppArt, true, null);

        // Device selection via dropdown
        ComboBox<String> cboDevice = new ComboBox<>();
        cboDevice.getItems().addAll("CPU", "GPU", "MPS (Apple Silicon)");
        cboDevice.setEditable(false);
        cboDevice.setValue("CPU");
        Label lDevice = new Label("Device");
        grid.add(lDevice, 0, row);
        grid.add(cboDevice, 1, row++);

        TextField tfMinArea = new TextField("0");
        addRow(grid, row++, "Min tissue area (µm²)", tfMinArea, false, null);

        Button btnRun = new Button("Run");
        Button btnCancel = new Button("Cancel");
        btnRun.setDefaultButton(true);
        btnCancel.setCancelButton(true);
        btnCancel.setOnAction(e -> dialog.close());
        HBox bottomBar = new HBox(10);
        bottomBar.setAlignment(Pos.CENTER_LEFT);
        bottomBar.getChildren().addAll(btnRun, btnCancel);
        grid.add(bottomBar, 0, row, 3, 1);

        dialog.setScene(new Scene(grid));
        dialog.show();

        try {
            String G = "artefact";
            String cp = Prefs.getString("classpose_dir", null); if (cp != null) tfClassposeDir.setText(cp);
            String out = Prefs.getString(Prefs.k(G, "out"), null); if (out != null) tfOut.setText(out);
            String mppA = Prefs.getString(Prefs.k(G, "mpp_model"), null); if (mppA != null) tfMppArt.setText(mppA);
            String dev = Prefs.getString("device", null);
            if (dev != null) {
                if (dev.equals("CPU") || dev.equals("GPU") || dev.startsWith("MPS")) {
                    cboDevice.setValue(dev);
                }
            }
            String minA = Prefs.getString("min_area", null); if (minA != null) tfMinArea.setText(minA);
        } catch (Throwable ignored) {}

        btnRun.setOnAction(e -> {
            if (isBlank(tfClassposeDir) || isBlank(tfOut) || isBlank(tfMppArt)) {
                showAlert("Please fill all required fields.");
                return;
            }
            String slidePath = resolveCurrentSlidePath();
            if (slidePath == null) {
                showAlert("Could not resolve current slide path. Please open a local WSI.");
                return;
            }

            // Cache python path
            try {
                String G = "artefact";
                Prefs.putString("classpose_dir", tfClassposeDir.getText().trim());
                Prefs.putString(Prefs.k(G, "out"), tfOut.getText().trim());
                Prefs.putString(Prefs.k(G, "mpp_model"), tfMppArt.getText().trim());
                String deviceKind = null;
                String selection = cboDevice.getSelectionModel().getSelectedItem();
                if (selection != null) {
                    if (selection.startsWith("CPU")) deviceKind = "cpu";
                    else if (selection.startsWith("GPU")) deviceKind = "cuda";
                    else if (selection.startsWith("MPS")) deviceKind = "mps";
                }
                if (deviceKind != null) Prefs.putString("device", deviceKind);
                if (!isBlank(tfMinArea)) Prefs.putString(Prefs.k(G, "min_area"), tfMinArea.getText().trim());
            } catch (Throwable ignored) {}

            File outDir = new File(tfOut.getText().trim());
            outDir.mkdirs();
            String base = new File(slidePath).getName();
            int dot = base.lastIndexOf('.');
            String basename = dot > 0 ? base.substring(0, dot) : base;
            File outPrefix = new File(outDir, basename);

            // Resolve shared models dir
            File modelsDir = getModelsDir();
            File tissueModelPath = new File(modelsDir, "grandqc_tissue_model.pt");
            File artefactModelPath = new File(modelsDir, "grandqc_artefact_model.pt");
            modelsDir.mkdirs();

            List<String> args = new ArrayList<>();
            // python -m classpose.grandqc.wsi_artefact_detection --slide_path ...
            args.add("-m"); args.add("classpose.grandqc.wsi_artefact_detection");
            args.add("--slide_path"); args.add(slidePath);
            args.add("--output_path"); args.add(outPrefix.getAbsolutePath());
            args.add("--mpp_model_art"); args.add(tfMppArt.getText().trim());
            args.add("--model_td_path"); args.add(tissueModelPath.getAbsolutePath());
            args.add("--model_art_path"); args.add(artefactModelPath.getAbsolutePath());
            // Compute device parameter from selection
            String deviceKind = null;
            String selection = cboDevice.getSelectionModel().getSelectedItem();
            if (selection != null) {
                if (selection.startsWith("CPU")) deviceKind = "cpu";
                else if (selection.startsWith("GPU")) deviceKind = "cuda";
                else if (selection.startsWith("MPS")) deviceKind = "mps";
            }
            if (deviceKind != null) { args.add("--device"); args.add(deviceKind); }
            if (!isBlank(tfMinArea)) { args.add("--min_area"); args.add(tfMinArea.getText().trim()); }

            long startTs = System.currentTimeMillis();
            dialog.close();
            runPythonWithLogging(tfClassposeDir.getText().trim(), outDir, args, outPrefix.getAbsolutePath(), startTs);
        });
    }


    private void runPythonWithLogging(String classposeDir, File workingDir, List<String> args, String outPrefix, long startTs) {
        PythonRunner runner = new PythonRunner();

        Stage logStage = new Stage();
        logStage.setTitle("Classpose - Artefact detection log");
        TextArea ta = new TextArea();
        ta.setEditable(false);
        Button btnCancel = new Button("Cancel");
        var vbox = new javafx.scene.layout.VBox(ta, btnCancel);
        javafx.scene.layout.VBox.setVgrow(ta, javafx.scene.layout.Priority.ALWAYS);
        logStage.setScene(new Scene(vbox, 900, 500));
        logStage.show();

        // Per-slide log file in the output folder
        String basename = new File(outPrefix).getName();
        File logFile = new File(workingDir, basename + "_artefacts.log");
        final BufferedWriter[] writerRef = new BufferedWriter[1];
        try {
            writerRef[0] = new BufferedWriter(new FileWriter(logFile, true));
            writerRef[0].write("Command: uv run --project " + classposeDir + " " + String.join(" ", args) + "\n\n");
            writerRef[0].flush();
        } catch (IOException ignored) {}

        Consumer<String> append = line -> {
            javafx.application.Platform.runLater(() -> ta.appendText(line + "\n"));
            if (writerRef[0] != null) { try { writerRef[0].write(line + "\n"); } catch (IOException ignored) {} }
        };
        Consumer<String> logOut = s -> append.accept("[OUT] " + s);
        Consumer<String> logErr = s -> append.accept("[ERR] " + s);

        final Process[] procRef = new Process[1];

        var task = new javafx.concurrent.Task<Void>() {
            @Override protected Void call() throws Exception {
                procRef[0] = runner.start(classposeDir, workingDir, null, args, logOut, logErr);
                int code = procRef[0].waitFor();
                if (writerRef[0] != null) { try { writerRef[0].flush(); writerRef[0].close(); } catch (IOException ignored) {} }
                if (code != 0) throw new RuntimeException("Python exited with code " + code + ". See log: " + logFile.getAbsolutePath());
                return null;
            }
        };

        btnCancel.setOnAction(e -> {
            try { runner.kill(procRef[0]); btnCancel.setDisable(true); ta.appendText("[INFO] Cancel requested by user.\n"); } catch (Throwable ignored) {}
        });

        task.setOnFailed(e -> {
            Throwable ex = task.getException();
            showAlert("Artefact detection failed: " + (ex != null ? ex.getMessage() : "Unknown error"));
        });
        task.setOnSucceeded(e -> {
            logStage.close();
            try {
                // Import generated GeoJSON
                File artefactGeo = new File(outPrefix + "_artefact_contours.geojson");
                var imgData = qupath.getImageData();
                if (imgData != null && artefactGeo.exists() && artefactGeo.lastModified() >= startTs) {
                    PathObjectHierarchy hier = imgData.getHierarchy();
                    GeoJsonImporter.importOutputs(hier, null, null, null, artefactGeo);
                }
                Alert ok = new Alert(Alert.AlertType.INFORMATION, "Artefact detection completed and results imported.", ButtonType.OK);
                ok.initOwner(qupath.getStage());
                ok.showAndWait();
            } catch (Exception ex) {
                showAlert("Completed but failed to import artefact results: " + ex.getMessage());
            }
        });

        Thread th = new Thread(task, "ClassposeArtefactDetection");
        th.setDaemon(true);
        th.start();
    }

}
