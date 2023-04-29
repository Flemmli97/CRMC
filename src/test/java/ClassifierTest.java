import io.github.flemmli97.MultiLabelClassifier;
import io.github.flemmli97.Pair;
import io.github.flemmli97.Settings;
import io.github.flemmli97.conformal.Conformal;
import io.github.flemmli97.dataset.LabelledSet;
import io.github.flemmli97.dataset.UnlabelledSet;
import org.junit.Test;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class ClassifierTest {

    @Test
    public void emotionsWithoutConformal() {
        MultiLabelClassifier.LOG = false;
        run(new Settings().withoutConformal(), "./data/emotions/emotions.arff", "./data/emotions/emotions.xml", "./data/emotions/emotions-test.arff");
    }

    @Test
    public void emotionsWithConformal() {
        MultiLabelClassifier.LOG = false;
        run(new Settings().withConfidenceP(0.2f), "./data/emotions/emotions.arff", "./data/emotions/emotions.xml", "./data/emotions/emotions-test.arff");
    }

    @Test
    public void emotionsWithConformal2() {
        MultiLabelClassifier.LOG = false;
        run(new Settings().withConfidenceP(0.2f), "./data/emotions/emotions.arff", "./data/emotions/emotions.xml", "./data/emotions/emotions-test.arff");
    }

    @Test
    public void birdsWithoutConformal() {
        MultiLabelClassifier.LOG = false;
        run(new Settings().withoutConformal(), "./data/birds/birds-train.arff", "./data/birds/birds.xml", "./data/birds/birds-test.arff");
    }

    @Test
    public void birdsWithConformal() {
        MultiLabelClassifier.LOG = false;
        run(new Settings().withConfidenceP(0.2f), "./data/birds/birds-train.arff", "./data/birds/birds.xml", "./data/birds/birds-test.arff");
    }

    @Test
    public void birdsWithConformal2() {
        MultiLabelClassifier.LOG = false;
        run(new Settings(), "./data/birds/birds-train.arff", "./data/birds/birds.xml", "./data/birds/birds-test.arff");
    }

    @Test
    public void sceneWithoutConformal() {
        MultiLabelClassifier.LOG = false;
        run(new Settings().withoutConformal(), "./data/scene/scene-train.arff", "./data/scene/scene.xml", "./data/scene/scene-test.arff");
    }

    @Test
    public void sceneWithConformal() {
        MultiLabelClassifier.LOG = false;
        run(new Settings().withConfidenceP(0.2f), "./data/scene/scene-train.arff", "./data/scene/scene.xml", "./data/scene/scene-test.arff");
    }

    @Test
    public void sceneWithConformal2() {
        MultiLabelClassifier.LOG = false;
        run(new Settings(), "./data/scene/scene-train.arff", "./data/scene/scene.xml", "./data/scene/scene-test.arff");
    }

    private static void run(Settings settings, String filePath, String xml, String testFile) {
        MultiLabelClassifier.LOG = false;
        String classifier = "RMLC";
        List<Pair<Long, Long>> timings = new ArrayList<>();
        for (int i = 0; i < 500; i++) {
            MultiLabelClassifier.runLearner(classifier, settings, new LabelledSet(Path.of(filePath), Path.of(xml)), new UnlabelledSet(Path.of(testFile)), false, false, timings);
        }
        System.out.println((!settings.conformal ? "=====NonConformal" : "=====Conformal timings ") + filePath + " p " + settings.confidenceP);
        System.out.println("Timings " + timings);
        System.out.println("Average learning time " + timings.stream().map(Pair::first).mapToLong(v -> v).average().orElse(0));
        System.out.println("Average prediction time " + timings.stream().map(Pair::second).mapToLong(v -> v).average().orElse(0));
    }
}
