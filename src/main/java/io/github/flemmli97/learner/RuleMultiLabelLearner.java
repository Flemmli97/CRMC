package io.github.flemmli97.learner;

import io.github.flemmli97.Pair;
import io.github.flemmli97.Settings;
import io.github.flemmli97.dataset.LabelledSet;
import io.github.flemmli97.dataset.Output;
import io.github.flemmli97.dataset.UnlabelledSet;
import io.github.flemmli97.measure.BaseMeasures;
import io.github.flemmli97.reflection.ReflectionUtil;
import weka.classifiers.Classifier;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.Rule;
import weka.classifiers.rules.RuleStats;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Implementation of a BinaryRelevance classifier using wekas implementation of RIPPER {@link JRip}
 */
public class RuleMultiLabelLearner implements Learner {

    private int threadCount;
    private double pConfidence;

    private boolean learned;

    private Rule[][] rules;
    private double[][] ruleQuality;
    private double[][][] distributions;
    private Classifier[] clss;
    private int[] labels;

    private final InstanceTransformer instanceTransformer = (insts, originLabels, label) -> {
        Remove attRemover = new Remove();
        int[] removeIndices = new int[originLabels.length - 1];
        int i = 0;
        for (int remove : originLabels) {
            if (remove == label)
                continue;
            removeIndices[i] = remove;
            i++;
        }
        attRemover.setAttributeIndicesArray(removeIndices);
        attRemover.setInputFormat(insts);
        return Filter.useFilter(insts, attRemover);
    };

    private final boolean applyConformal, log;

    public RuleMultiLabelLearner(Settings settings) {
        this.threadCount = settings.threadCount;
        this.pConfidence = settings.confidenceP;
        this.applyConformal = settings.conformal;
        this.log = settings.log;
    }

    public double threshold() {
        return this.pConfidence;
    }

    public void setThreshold(double val) {
        this.pConfidence = val;
    }

    @Override
    public void learn(LabelledSet set) {
        this.clss = new Classifier[set.labels.length];
        this.rules = new Rule[this.clss.length][];
        this.distributions = new double[this.clss.length][][];
        this.ruleQuality = new double[this.clss.length][];
        int ci = 0;
        this.labels = set.labels;
        Random random = new Random();
        for (int label : set.labels) {
            JRip ripper = new JRip();
            Instances clonedMain = new Instances(set.insts);
            random.setSeed(0);
            //Instances shuffled = new Instances(set.insts);
            //shuffled.randomize(random);
            Instances copy = clonedMain;
            try {
                copy.setClassIndex(label);
                copy = this.instanceTransformer.of(copy, set.labels, label);
                ripper.buildClassifier(copy);
            } catch (Exception e) {
                e.printStackTrace();
            }
            Attribute classAttribute = ReflectionUtil.getField(ripper, "m_Class");
            this.clss[ci] = ripper;
            ArrayList<double[]> dist = ReflectionUtil.getField(ripper, "m_Distributions");
            this.rules[ci] = ripper.getRuleset().toArray(Rule[]::new);
            this.distributions[ci] = dist.toArray(double[][]::new);
            if (this.applyConformal) {
                List<RuleStats> stats = ReflectionUtil.getField(ripper, "m_RulesetStats");
                List<Pair<String, String>> stat = stats.stream().map(s -> {
                    ArrayList<Pair<String, String>> val = new ArrayList<>();
                    for (int i = 0; i < s.getRulesetSize(); i++) {
                        double[] d = s.getSimpleStats(i);
                        val.add(new Pair<>(classAttribute.value((int) s.getRuleset().get(i).getConsequent()), Arrays.toString(d)));
                    }
                    return val;
                }).flatMap(Collection::stream).toList();
                if (this.log) {
                    System.out.println("STATS " + stat);
                }
                this.ruleQuality[ci] = stats.stream().map(s -> {
                    ArrayList<Double> measure = new ArrayList<>();
                    for (int i = 0; i < s.getRulesetSize(); i++) {
                        double[] d = s.getSimpleStats(i);
                        //double trueP = d[0] - d[4];
                        //double pos = d[2] / (d[2] + d[3]);
                        //measure.add(pos - Math.sqrt(1 / (d[2] + d[3])));
                        //measure.add(BaseMeasures.f1((int) d[2], (int) d[4], (int) d[5]));
                        measure.add(BaseMeasures.precision((int) d[2], (int) d[4]));
                    }
                    return measure;
                }).flatMapToDouble(l -> l.stream().mapToDouble(d -> d)).toArray();
                if (this.log)
                    System.out.println("qualities " + Arrays.toString(this.ruleQuality[ci]));
            }
            ci++;
        }
        this.learned = true;
    }

    @Override
    public Output predict(UnlabelledSet data) {
        if (!this.learned)
            throw new IllegalStateException();
        //Apply conformal prediction when evaluating labels here. Note: Use new learner class
        Map<Integer, List<Integer>> result = new HashMap<>();
        int classIndex = 0;
        for (int label : this.labels) {
            try {
                data.insts.setClassIndex(label);
                Instances instances = this.instanceTransformer.of(data.insts, this.labels, label);
                int instInd = 0;
                for (Instance inst : instances) {
                    double[] distribution;
                    for (int i = 0; i < this.rules[classIndex].length; ++i) {
                        Rule rule = this.rules[classIndex][i];
                        if (!this.applyConformal) {
                            if (rule.covers(inst)) {
                                distribution = this.distributions[classIndex][i];
                                if (distribution[0] <= distribution[1])
                                    result.computeIfAbsent(instInd, o -> new ArrayList<>()).add(label);
                                break;
                            }
                        } else {
                            double quality = 1 - this.ruleQuality[classIndex][i];
                            if (rule.covers(inst)) {
                                distribution = this.distributions[classIndex][i];
                                boolean reject = quality <= this.pConfidence;
                                if (distribution[0] <= distribution[1] && !reject)
                                    result.computeIfAbsent(instInd, o -> new ArrayList<>()).add(label);
                                break;
                            }
                        }
                    }
                    instInd++;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            classIndex++;
        }
        return new Output(data, this.labels, result);
    }

    public String getRules() {
        return Arrays.deepToString(this.clss);
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("Rule based Multilabel Learner");
        builder.append("Rules:");
        builder.append(this.getRules());
        return builder.toString();
    }

    interface InstanceTransformer {
        Instances of(Instances insts, int[] originLabels, int label) throws Exception;
    }

    record Res() {

    }
}
