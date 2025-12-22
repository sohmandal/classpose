package qupath.ext.classpose.util;

import javafx.geometry.HPos;
import javafx.scene.control.*;
import javafx.scene.layout.ColumnConstraints;
import javafx.scene.layout.GridPane;
import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import qupath.ext.classpose.ClassposeExtension;
import qupath.lib.gui.QuPathGUI;

import java.io.File;
import java.net.URI;
import java.util.Collection;

/**
 * Base helpers for Classpose actions.
 * Provides small UI utilities, slide path resolution, and model dir discovery.
 */
public abstract class AbstractClassposeAction {

    protected final QuPathGUI qupath;

    protected AbstractClassposeAction(final QuPathGUI qupath) {
        this.qupath = qupath;
    }

    protected void addRow(GridPane grid, int row, String label, TextField tf, boolean required, Runnable browseAction) {
        Label l = new Label(label + (required ? " *" : ""));
        grid.add(l, 0, row);
        grid.add(tf, 1, row);
        if (browseAction != null) {
            Button b = new Button("Browse");
            b.setOnAction(e -> browseAction.run());
            grid.add(b, 2, row);
        }
    }

    protected void configureFormColumns(GridPane grid) {
        ColumnConstraints col0 = new ColumnConstraints();
        col0.setHalignment(HPos.RIGHT);
        ColumnConstraints col1 = new ColumnConstraints();
        ColumnConstraints col2 = new ColumnConstraints();
        grid.getColumnConstraints().setAll(col0, col1, col2);
    }

    protected boolean isBlank(TextField tf) {
        return tf.getText() == null || tf.getText().trim().isEmpty();
    }

    protected void browseFile(TextField target, String title) {
        FileChooser chooser = new FileChooser();
        chooser.setTitle(title);
        File f = chooser.showOpenDialog(qupath.getStage());
        if (f != null) target.setText(f.getAbsolutePath());
    }

    protected void browseDirectory(TextField target, String title) {
        DirectoryChooser chooser = new DirectoryChooser();
        chooser.setTitle(title);
        File f = chooser.showDialog(qupath.getStage());
        if (f != null) target.setText(f.getAbsolutePath());
    }

    protected void showAlert(String msg) {
        Alert alert = new Alert(Alert.AlertType.WARNING, msg, ButtonType.OK);
        alert.initOwner(qupath.getStage());
        alert.showAndWait();
    }

    /** Attempt to resolve a local filesystem path for the currently opened image. */
    protected String resolveCurrentSlidePath() {
        try {
            var imgData = qupath.getImageData();
            if (imgData == null) return null;
            var server = imgData.getServer();
            if (server == null) return null;
            Collection<URI> uris = server.getURIs();
            URI firstUri = (uris == null || uris.isEmpty()) ? null : uris.iterator().next();
            if (firstUri != null) {
                File f = new File(firstUri);
                if (f.exists()) return f.getAbsolutePath();
            }
            String path = server.getPath();
            if (path != null) {
                File f = new File(path);
                if (f.exists()) return f.getAbsolutePath();
            }
        } catch (Throwable ignored) { }
        return null;
    }

    /** Resolve a fixed directory for shared model weights next to the installed JAR, with a user-home fallback. */
    protected File getModelsDir() {
        try {
            File jar = new File(ClassposeExtension.class.getProtectionDomain().getCodeSource().getLocation().toURI());
            File baseDir = jar.getParentFile();
            return new File(baseDir, "classpose-models");
        } catch (Exception e) {
            return new File(System.getProperty("user.home"), ".qupath-classpose-models");
        }
    }
}
