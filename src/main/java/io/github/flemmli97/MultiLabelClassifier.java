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
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import weka.classifiers.rules.DecisionTable;
import weka.core.Instance;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MultiLabelClassifier {

    public static void main(String[] args) {
        Options options = new Options();
        options.addOption("f", true, "File path");
        options.addOption("xml", true, "XML path");
        options.addOption("label", true, "File path");
        options.addOption("c", true, "The classifier implementation. Currently only RMLC");
        options.addOption("test", true, "The test file");
        options.addOption("plot", false, "If set runs using thresholds 0-1 and creates a plot using that");
        options.addOption("rules", false, "If set prints the rules");
        options.addOption("nonconformal", false, "Disable conformal prediction");
        options.addOption("log", false, "Logging things. WIP");
        options.addOption("t", true, "threshold");

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd = null;
        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("java -jar <jar>", options);
            System.exit(1);
        }

        String filePath = cmd.getOptionValue("-f");
        String xml = cmd.getOptionValue("-xml");
        LabelledSet set;
        if (xml == null) {
            String labelString = cmd.getOptionValue("-label");
            if (labelString == null) {
                formatter.printHelp("utility-name", options);
                throw new RuntimeException("Neither xml nor label count specified");
            }
            set = new LabelledSet(Path.of(filePath), Integer.parseInt(labelString));
        } else {
            set = new LabelledSet(Path.of(filePath), Path.of(xml));
        }
        if (xml != null)
            System.out.printf("Using dataset %s with label file %s%n", filePath, xml);
        else
            System.out.printf("Using dataset %s with label file %s%n", filePath, cmd.getOptionValue("-label"));
        String testFile = cmd.getOptionValue("-test");

        UnlabelledSet test = testFile == null ? new UnlabelledSet(set.insts) : new UnlabelledSet(Path.of(testFile));
        String classifier = cmd.getOptionValue("-c", "RMLC");

        Settings settings = new Settings();

        if (cmd.hasOption("-nonconformal"))
            settings.withoutConformal();

        if (cmd.hasOption("-log"))
            settings.log();

        if (cmd.hasOption("t")) {
            settings.withConfidenceP(Float.parseFloat(cmd.getOptionValue("t")));
        }

        runLearner(classifier, settings, set, test, cmd.hasOption("-rules"), cmd.hasOption("-plot"));
    }

    private static void runLearner(String classifier, Settings settings, LabelledSet set, UnlabelledSet test, boolean rules, boolean plot) {
        Learner learner = Learners.getLearner(classifier)
                .orElseThrow(() -> new RuntimeException("No such learner " + classifier))
                .apply(settings);
        System.out.println("Learning...");
        long time = System.nanoTime();
        learner.learn(set);
        System.out.println("Learning took " + (System.nanoTime() - time) + " ns");
        if (rules)
            System.out.println(((RuleMultiLabelLearner) learner).getRules());
        if (plot) {
            ArrayList<Pair<Double, Output>> res = new ArrayList<>();
            for (double d = 0; d < 1.5; d += 0.025) {
                ((RuleMultiLabelLearner) learner).setThreshold(d); //Currently only this impl so its fine
                Output o = learner.predict(test);
                res.add(new Pair<>(d, o));
                System.out.println("FALSE Loss: " + o.confusionMatrix.getFalsePositive());
                System.out.println("FALSE NEG: " + o.confusionMatrix.getFalseNegative());
            }
            PlotVisualizer.plotF1(res);
            PlotVisualizer.plot(res, "Hamming-Loss", "Hamming-Loss", Output::hammingLoss);
        } else {
            Output o = learner.predict(test);
            System.out.println("Micro Accuracy: " + o.confusionMatrix.accuracy());
            System.out.println("Macro Accuracy: " + o.confusionMatrix.macroAccuracy());
            System.out.println("Precision: " + o.confusionMatrix.precision());
            System.out.println("Macro Precision: " + o.confusionMatrix.macroPrecision());
            System.out.println("Recall: " + o.confusionMatrix.recall());
            System.out.println("Macro Recall: " + o.confusionMatrix.macroRecall());
            System.out.println("F1: " + o.confusionMatrix.f1());
            System.out.println("Macro F1: " + o.confusionMatrix.macroF1());
            System.out.println("MCC: " + o.confusionMatrix.MCC());
            System.out.println("Hamming Loss: " + o.hammingLoss());
            System.out.println("AMOUT: " + (o.dataSet.insts.size() * o.labels.length));
            System.out.println("FALSE Pos: " + o.confusionMatrix.getFalsePositive());
            System.out.println("FALSE NEG: " + o.confusionMatrix.getFalseNegative());
            //System.out.println("=======Labels: ");
            //for(Pair<Integer, List<String>> val : o.instanceLabelsReadable)
            //    System.out.println(val);
        }
    }

    private static void binaryRelevanceMulanTest(LabelledSet set, String file, String xml) {
        System.out.println("Running mulan binary relevance");
        long time = System.nanoTime();
        BinaryRelevance r = new BinaryRelevance(new DecisionTable());
        try {
            r.build(new MultiLabelInstances(file, xml));
            System.out.println("Learning took " + (System.nanoTime() - time) + " ns");
            MultiLabelInstances mI = new MultiLabelInstances("./data/CAL500.arff", "./data/CAL500.xml");
            Map<Integer, List<Integer>> result = new HashMap<>();
            int i = 0;
            for (Instance inst : mI.getDataSet()) {
                MultiLabelOutput multiLabelOutput = r.makePrediction(inst);
                List<Integer> l = new ArrayList<>();
                for (int li = 0; li < multiLabelOutput.getBipartition().length; li++) {
                    boolean b = multiLabelOutput.getBipartition()[li];
                    if (b)
                        l.add(set.labels[li]);
                }
                result.computeIfAbsent(i, old -> new ArrayList<>())
                        .addAll(l);
                i++;
            }
            Output out = new Output(new UnlabelledSet(mI.getDataSet()), set.labels, result, true);
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
}
