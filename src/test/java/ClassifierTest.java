import io.github.flemmli97.MultiLabelClassifier;
import io.github.flemmli97.Pair;
import io.github.flemmli97.Settings;
import io.github.flemmli97.dataset.LabelledSet;
import io.github.flemmli97.dataset.UnlabelledSet;
import org.junit.Test;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class ClassifierTest {

    @Test
    public void runWithoutConformal() {
        MultiLabelClassifier.LOG = false;
        String filePath = "./data/emotions/emotions.arff";
        String xml = "./data/emotions/emotions.xml";
        String testFile = "./data/emotions/emotions-test.arff";
        String classifier = "RMLC";

        Settings settings = new Settings();
        settings.withoutConformal();
        List<Pair<Long, Long>> timings = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            MultiLabelClassifier.runLearner(classifier, settings, new LabelledSet(Path.of(filePath), Path.of(xml)), new UnlabelledSet(Path.of(testFile)), false, false, timings);
        }
        System.out.println("=====Nonconformal timings");
        System.out.println("Timings " + timings);
        System.out.println("Average learning time " + timings.stream().map(Pair::first).mapToLong(v -> v).average().orElse(0));
        System.out.println("Average prediction time " + timings.stream().map(Pair::second).mapToLong(v -> v).average().orElse(0));
    }

    @Test
    public void runWithConformal() {
        MultiLabelClassifier.LOG = false;
        String filePath = "./data/emotions/emotions.arff";
        String xml = "./data/emotions/emotions.xml";
        String testFile = "./data/emotions/emotions-test.arff";
        String classifier = "RMLC";

        Settings settings = new Settings();
        List<Pair<Long, Long>> timings = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            MultiLabelClassifier.runLearner(classifier, settings, new LabelledSet(Path.of(filePath), Path.of(xml)), new UnlabelledSet(Path.of(testFile)), false, false, timings);
        }
        System.out.println("=====Conformal timings");
        System.out.println("Timings " + timings);
        System.out.println("Average learning time " + timings.stream().map(Pair::first).mapToLong(v -> v).average().orElse(0));
        System.out.println("Average prediction time " + timings.stream().map(Pair::second).mapToLong(v -> v).average().orElse(0));
    }

    @Test
    public void runWithConformal2() {
        MultiLabelClassifier.LOG = false;
        String filePath = "./data/emotions/emotions.arff";
        String xml = "./data/emotions/emotions.xml";
        String testFile = "./data/emotions/emotions-test.arff";
        String classifier = "RMLC";

        Settings settings = new Settings();
        settings.confidenceP = 0.2;
        List<Pair<Long, Long>> timings = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            MultiLabelClassifier.runLearner(classifier, settings, new LabelledSet(Path.of(filePath), Path.of(xml)), new UnlabelledSet(Path.of(testFile)), false, false, timings);
        }
        System.out.println("=====Conformal timings");
        System.out.println("Timings " + timings);
        System.out.println("Average learning time " + timings.stream().map(Pair::first).mapToLong(v -> v).average().orElse(0));
        System.out.println("Average prediction time " + timings.stream().map(Pair::second).mapToLong(v -> v).average().orElse(0));
    }
}
