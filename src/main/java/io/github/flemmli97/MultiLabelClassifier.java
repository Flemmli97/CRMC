package io.github.flemmli97;

import io.github.flemmli97.api.learners.Learners;
import io.github.flemmli97.dataset.LabelledSet;
import io.github.flemmli97.dataset.Output;
import io.github.flemmli97.dataset.UnlabelledSet;
import io.github.flemmli97.learner.Learner;
import io.github.flemmli97.learner.RuleMultiLabelLearner;
import io.github.flemmli97.plots.PlotVisualizer;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.OneR;
import weka.core.Instance;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MultiLabelClassifier {

    public static void main(String[] args) {
        Path path = Path.of("").toAbsolutePath();

        Options options = new Options();

        Option.builder();

        //--Create learning instance

        //--Create labelled instance
        String file = "./data/20NG-F.arff";
        //LabelledSet set = new LabelledSet(Path.of(file), 10);

        String f = "./data/CAL500";
        LabelledSet set2 = new LabelledSet(Path.of(f + ".arff"), Path.of(f + ".xml"));

        String learnerID = "RLCM";
        Settings settings = new Settings();

        Learner learner = Learners.getLearner(learnerID)
                .orElseThrow(() -> new RuntimeException("No such learner " + learnerID))
                .apply(settings);

        learner.learn(set2);

        ArrayList<PlotVisualizer.PlotPair> res = new ArrayList<>();

        for(double d = 0; d < 1; d+=0.05) {
            ((RuleMultiLabelLearner)learner).setThreshold(d);
            Output o = learner.predict(set2);
            res.add(new PlotVisualizer.PlotPair(d, o));
            //for (List<String> l : o.instanceLabelsReadable)
            //    System.out.println(l);
            /*System.out.println(o.confusionMatrix.accuracy());
            System.out.println(o.confusionMatrix.precision());
            System.out.println(o.confusionMatrix.recall());
            System.out.println(o.confusionMatrix.f1());
            System.out.println(o.confusionMatrix.MCC());*/
        }
        PlotVisualizer.plotF1(res);

        BinaryRelevance r = new BinaryRelevance(new DecisionTable());
        try {
            r.build(new MultiLabelInstances("./data/CAL500.arff", "./data/CAL500.xml"));
            MultiLabelInstances mI = new MultiLabelInstances("./data/CAL500.arff", "./data/CAL500.xml");
            Map<Integer, List<Integer>> result = new HashMap<>();
            int i = 0;
            for (Instance inst : mI.getDataSet()) {
                MultiLabelOutput multiLabelOutput = r.makePrediction(inst);
                List<Integer> l = new ArrayList<>();
                for (int li = 0; li < multiLabelOutput.getBipartition().length; li++) {
                    boolean b = multiLabelOutput.getBipartition()[li];
                    if (b)
                        l.add(set2.labels[li]);
                }
                result.computeIfAbsent(i, old -> new ArrayList<>())
                        .addAll(l);
                i++;
            }
            Output out = new Output(new UnlabelledSet(mI.getDataSet()), set2.labels, result);
            System.out.println("=======");
            System.out.println(out.confusionMatrix.accuracy());
            System.out.println(out.confusionMatrix.precision());
            System.out.println(out.confusionMatrix.recall());
            System.out.println(out.confusionMatrix.f1());
            System.out.println(out.confusionMatrix.MCC());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(Option option) {

    }
}
