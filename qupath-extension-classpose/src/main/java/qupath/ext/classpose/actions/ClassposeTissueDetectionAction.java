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
 * Action to run GrandQC tissue detection on the current image and import results.
 *
 * <p>If you use any part of Classpose which makes use of GrandQC please follow the instructions 
 * at https://github.com/cpath-ukk/grandqc/tree/main to cite them appropriately. Similarly to 
 * Classpose, GrandQC is under a non-commercial license whose terms can be found at 
 * https://github.com/cpath-ukk/grandqc/blob/main/LICENSE.</p>
 *
 * <p>Collects parameters via a dialog, runs {@code classpose.grandqc.wsi_tissue_detection},
 * streams logs live, and imports generated tissue GeoJSON. UI fields are cached across
 * sessions using {@link java.util.prefs.Preferences}.</p>
 */
public class ClassposeTissueDetectionAction extends AbstractClassposeAction {

    public ClassposeTissueDetectionAction(final QuPathGUI qupath) {
        super(qupath);
    }

    public void run() {
        Stage dialog = new Stage();
        dialog.setTitle("Classpose – Tissue detection");

        GridPane grid = new GridPane();
        grid.setHgap(8);
        grid.setVgap(8);
        grid.setPadding(new Insets(12));

        int row = 0;
        TextField tfClassposeDir = new TextField();
        addRow(grid, row++, "Classpose directory", tfClassposeDir, true, () -> browseFile(tfClassposeDir, "Select Classpose directory"));
        // Load cached python path if available
        try {
            Preferences prefs = Preferences.userRoot().node("qupath.ext.classpose");
            String cached = prefs.get("classpose_dir", null);
            if (cached != null && !cached.isBlank()) tfClassposeDir.setText(cached);
        } catch (Throwable ignored) {}

        TextField tfOut = new TextField();
        addRow(grid, row++, "Output folder", tfOut, true, () -> browseDirectory(tfOut, "Select output folder"));

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
        // Load cached UI state
        try {
            String G = "tissue";
            String cp = Prefs.getString("classpose_dir", null); if (cp != null) tfClassposeDir.setText(cp);
            String out = Prefs.getString(Prefs.k(G, "out"), null); if (out != null) tfOut.setText(out);
            String dev = Prefs.getString("device", null);
            if (dev != null) {
                switch (dev.toLowerCase()) {
                    case "cpu": cboDevice.getSelectionModel().select("CPU"); break;
                    case "cuda": cboDevice.getSelectionModel().select("GPU"); break;
                    case "mps": cboDevice.getSelectionModel().select("MPS (Apple Silicon)"); break;
                }
            }
            String minA = Prefs.getString("min_area", null); if (minA != null) tfMinArea.setText(minA);
        } catch (Throwable ignored) {}

        btnRun.setOnAction(e -> {
            if (isBlank(tfClassposeDir) || isBlank(tfOut)) {
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
                String G = "tissue";
                Prefs.putString("classpose_dir", tfClassposeDir.getText().trim());
                Prefs.putString(Prefs.k(G, "out"), tfOut.getText().trim());
                String selectedLabel = cboDevice.getSelectionModel().getSelectedItem();
                if (selectedLabel != null) Prefs.putString("device", selectedLabel);
                if (!isBlank(tfMinArea)) Prefs.putString(Prefs.k(G, "min_area"), tfMinArea.getText().trim());
            } catch (Throwable ignored) {}

            // Prepare output prefix: <folder>/<basename>
            File outDir = new File(tfOut.getText().trim());
            outDir.mkdirs();
            String base = new File(slidePath).getName();
            int dot = base.lastIndexOf('.');
            String basename = dot > 0 ? base.substring(0, dot) : base;
            File outPrefix = new File(outDir, basename);

            List<String> args = new ArrayList<>();
            // Use module form: python -m classpose.grandqc.wsi_tissue_detection ...
            args.add("-m"); args.add("classpose.grandqc.wsi_tissue_detection");
            args.add("--slide_path"); args.add(slidePath);
            args.add("--output_path"); args.add(outPrefix.getAbsolutePath());

            // Resolve the shared models dir and pass the same tissue model path used by Predict WSI
            File modelsDir = getModelsDir();
            File tissueModelPath = new File(modelsDir, "grandqc_tissue_model.pt");
            modelsDir.mkdirs();
            args.add("--model_path"); args.add(tissueModelPath.getAbsolutePath());
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
        logStage.setTitle("Classpose – Tissue detection log");
        TextArea ta = new TextArea();
        ta.setEditable(false);
        Button btnCancel = new Button("Cancel");
        var vbox = new javafx.scene.layout.VBox(ta, btnCancel);
        javafx.scene.layout.VBox.setVgrow(ta, javafx.scene.layout.Priority.ALWAYS);
        logStage.setScene(new Scene(vbox, 900, 500));
        logStage.show();

        // Per-slide log file in the output folder
        String basename = new File(outPrefix).getName();
        File logFile = new File(workingDir, basename + "_tissue.log");
        final BufferedWriter[] writerRef = new BufferedWriter[1];
        try {
            writerRef[0] = new BufferedWriter(new FileWriter(logFile, true));
            writerRef[0].write("Command: uv run --project " + classposeDir + " " + String.join(" ", args) + "\n\n");
            writerRef[0].flush();
        } catch (IOException ignored) {}

        Consumer<String> append = line -> {
            javafx.application.Platform.runLater(() -> ta.appendText(line + "\n"));
            if (writerRef[0] != null) {
                try { writerRef[0].write(line + "\n"); } catch (IOException ignored) {}
            }
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
            showAlert("Tissue detection failed: " + (ex != null ? ex.getMessage() : "Unknown error"));
        });
        task.setOnSucceeded(e -> {
            logStage.close();
            try {
                // Import generated GeoJSON
                File geojson = new File(outPrefix + "_geojson.json");
                var imgData = qupath.getImageData();
                if (imgData != null && geojson.exists() && geojson.lastModified() >= startTs) {
                    PathObjectHierarchy hier = imgData.getHierarchy();
                    GeoJsonImporter.importOutputs(hier, null, null, geojson, null);
                }
                Alert ok = new Alert(Alert.AlertType.INFORMATION, "Tissue detection completed and results imported.", ButtonType.OK);
                ok.initOwner(qupath.getStage());
                ok.showAndWait();
            } catch (Exception ex) {
                showAlert("Completed but failed to import tissue results: " + ex.getMessage());
            }
        });

        Thread th = new Thread(task, "ClassposeTissueDetection");
        th.setDaemon(true);
        th.start();
    }
}
