package qupath.ext.classpose;

import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.QuPathExtension;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuItem;
import qupath.ext.classpose.actions.ClassposePredictWSIAction;
import qupath.ext.classpose.actions.ClassposeTissueDetectionAction;
import qupath.ext.classpose.actions.ClassposeArtefactDetectionAction;

/**
 * QuPath extension entry point for Classpose integration.
 */
public class ClassposeExtension implements QuPathExtension {

    @Override
    public void installExtension(final QuPathGUI qupath) {
        // Register a minimal menu action to launch the Classpose prediction dialog/workflow
        Menu extensionsMenu = qupath.getMenu("Extensions", true);
        if (extensionsMenu != null) {
            Menu classposeMenu = null;
            // Try to find an existing Classpose menu to avoid duplicates
            for (var item : extensionsMenu.getItems()) {
                if (item instanceof Menu && ((Menu) item).getText().equals("Classpose")) {
                    classposeMenu = (Menu) item;
                    break;
                }
            }
            if (classposeMenu == null) {
                classposeMenu = new Menu("Classpose");
                extensionsMenu.getItems().add(classposeMenu);
            }

            MenuItem predictItem = new MenuItem("Predict WSI…");
            predictItem.setOnAction(e -> new ClassposePredictWSIAction(qupath).run());
            classposeMenu.getItems().add(predictItem);

            MenuItem tissueItem = new MenuItem("Tissue detection…");
            tissueItem.setOnAction(e -> new ClassposeTissueDetectionAction(qupath).run());
            classposeMenu.getItems().add(tissueItem);

            MenuItem artefactItem = new MenuItem("Artefact detection…");
            artefactItem.setOnAction(e -> new ClassposeArtefactDetectionAction(qupath).run());
            classposeMenu.getItems().add(artefactItem);
        }
    }

    @Override
    public String getName() {
        return "Classpose Extension";
    }

    @Override
    public String getDescription() {
        return "Run classpose.entrypoints.predict_wsi across the current WSI and import results.";
    }
}
