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
import weka.core.Instances;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MultiLabelClassifier {

    public static boolean LOG = true;

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
        options.addOption("nopruning", false, "Disable JRIP pruning");
        options.addOption("hold", true, "Use x-hold dataset as test");

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd = null;
        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            log(e.getMessage());
            formatter.printHelp("java -jar <jar>", options);
            System.exit(1);
        }

        String filePath = cmd.getOptionValue("f");
        String xml = cmd.getOptionValue("xml");
        LabelledSet set;
        if (xml == null) {
            String labelString = cmd.getOptionValue("label");
            if (labelString == null) {
                formatter.printHelp("utility-name", options);
                throw new RuntimeException("Neither xml nor label count specified");
            }
            set = new LabelledSet(Path.of(filePath), Integer.parseInt(labelString));
        } else {
            set = new LabelledSet(Path.of(filePath), Path.of(xml));
        }
        if (xml != null)
            log("Using dataset %s with label file %s%n", filePath, xml);
        else
            log("Using dataset %s with label file %s%n", filePath, cmd.getOptionValue("-label"));

        String testFile = cmd.getOptionValue("test");
        int hold = Math.max(1, Integer.parseInt(cmd.getOptionValue("hold", "10")));
        UnlabelledSet test;
        if (testFile != null) {
            test = new UnlabelledSet(Path.of(testFile));
        } else {
            int split = set.insts.size() - set.insts.size() / hold;
            test = new UnlabelledSet(new Instances(set.insts, split, set.insts.size() - split));
            set = new LabelledSet(new Instances(set.insts, 0, split), set.labels);
        }
        String classifier = cmd.getOptionValue("c", "RMLC");

        Settings settings = new Settings();

        if (cmd.hasOption("nonconformal"))
            settings.withoutConformal();

        if (cmd.hasOption("log"))
            settings.log();

        if (cmd.hasOption("t")) {
            settings.withConfidenceP(Float.parseFloat(cmd.getOptionValue("t")));
        }

        if (cmd.hasOption("nopruning")) {
            settings.disablePruning();
        }

        runLearner(classifier, settings, set, test, cmd.hasOption("-rules"), cmd.hasOption("-plot"), null);
    }

    public static void runLearner(String classifier, Settings settings, LabelledSet set, UnlabelledSet test, boolean rules, boolean plot, List<Pair<Long, Long>> timings) {
        Learner learner = Learners.getLearner(classifier)
                .orElseThrow(() -> new RuntimeException("No such learner " + classifier))
                .apply(settings);
        log("Learning...");
        long time = System.nanoTime();
        learner.learn(set);
        long learnTime = (System.nanoTime() - time);
        log("Learning took " + learnTime + " ns");
        if (rules)
            log(((RuleMultiLabelLearner) learner).getRules());
        time = System.nanoTime();
        if (plot) {
            ArrayList<Pair<Double, Output>> res = new ArrayList<>();
            for (double d = 0; d < 1.5; d += 0.1) {
                ((RuleMultiLabelLearner) learner).setThreshold(d); //Currently only this impl so its fine
                Output o = learner.predict(test);
                res.add(new Pair<>(d, o));
            }
            PlotVisualizer.plotF1(res);
            PlotVisualizer.plot(res, "Hamming-Loss", "Hamming-Loss", Output::hammingLoss);
        } else {
            Output o = learner.predict(test);
            log("Micro Accuracy: " + o.confusionMatrix.accuracy());
            log("Macro Accuracy: " + o.confusionMatrix.macroAccuracy());
            log("Precision: " + o.confusionMatrix.precision());
            log("Macro Precision: " + o.confusionMatrix.macroPrecision());
            log("Recall: " + o.confusionMatrix.recall());
            log("Macro Recall: " + o.confusionMatrix.macroRecall());
            log("F1: " + o.confusionMatrix.f1());
            log("Macro F1: " + o.confusionMatrix.macroF1());
            log("MCC: " + o.confusionMatrix.MCC());
            log("Hamming Loss: " + o.hammingLoss());
            log("AMOUT: " + (o.dataSet.insts.size() * o.labels.length));
            log("True Pos: " + o.confusionMatrix.getTruePositive());
            log("True NEG: " + o.confusionMatrix.getTrueNegative());
            log("FALSE Pos: " + o.confusionMatrix.getFalsePositive());
            log("FALSE NEG: " + o.confusionMatrix.getFalseNegative());
            //log("=======Labels: ");
            //for(Pair<Integer, List<String>> val : o.instanceLabelsReadable)
            //    log(val);
        }
        long predictTime = (System.nanoTime() - time);
        if (timings != null)
            timings.add(new Pair<>(learnTime, predictTime));
    }

    private static void binaryRelevanceMulanTest(LabelledSet set, String file, String xml) {
        log("Running mulan binary relevance");
        long time = System.nanoTime();
        BinaryRelevance r = new BinaryRelevance(new DecisionTable());
        try {
            r.build(new MultiLabelInstances(file, xml));
            log("Learning took " + (System.nanoTime() - time) + " ns");
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
            log("=======");
            log(out.confusionMatrix.accuracy());
            log(out.confusionMatrix.precision());
            log(out.confusionMatrix.recall());
            log(out.confusionMatrix.f1());
            log(out.confusionMatrix.MCC());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void log(Object msg, Object... obj) {
        if (LOG) {
            if (obj.length == 0)
                System.out.println(msg);
            else
                System.out.printf(msg.toString(), obj);
        }
    }
}
