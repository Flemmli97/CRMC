package io.github.flemmli97.learner;

import io.github.flemmli97.Settings;
import io.github.flemmli97.dataset.LabelledSet;
import io.github.flemmli97.dataset.Output;
import io.github.flemmli97.dataset.UnlabelledSet;
import io.github.flemmli97.reflection.ReflectionUtil;
import weka.classifiers.Classifier;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.Rule;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.ArrayList;
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
    private double[] positiveLabelProbability;
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

    public RuleMultiLabelLearner(Settings settings) {
        this.threadCount = settings.threadCount;
        this.pConfidence = settings.confidenceP;
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
        this.positiveLabelProbability = new double[this.clss.length];
        this.ruleQuality = new double[this.clss.length][];
        int ci = 0;
        this.labels = set.labels;
        Random random = new Random();
        for (int label : set.labels) {
            JRip ripper = new JRip();
            Instances clonedMain = new Instances(set.insts);
            random.setSeed(0);
            Instances shuffled = new Instances(set.insts);
            shuffled.randomize(random);
            int split = (int) (shuffled.size() * 0.5);
            Instances copy = clonedMain;
            /*new Instances(clonedMain, split);
            for(int i = 0; i < split; i++)
                copy.add(shuffled.instance(i));
            Instances test = new Instances(clonedMain, shuffled.size() - split);
            for(int i = 0; i < shuffled.size() - split; i++)
                test.add(shuffled.instance(i + split));
            test.setClassIndex(label);*/
            Instances test = clonedMain;
            int positive = 0;
            try {
                copy.setClassIndex(label);
                copy = this.instanceTransformer.of(copy, set.labels, label);
                ripper.buildClassifier(copy);
                for (Instance i : copy) {
                    if (i.value(copy.classAttribute()) == 1)
                        positive++;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            int size = copy.size();
            this.clss[ci] = ripper;
            ArrayList<double[]> dist = ReflectionUtil.getField(ripper, "m_Distributions");
            this.rules[ci] = ripper.getRuleset().toArray(Rule[]::new);
            this.distributions[ci] = dist.toArray(double[][]::new);
            this.positiveLabelProbability[ci] = positive / (double) size;
            for (Instance inst : copy) {
                double labelVal = inst.value(copy.classAttribute());
                if (labelVal == 1)
                    positive++;
            }
            List<Double> ruleQuality = new ArrayList<>();
            for (int i = 0; i < ripper.getRuleset().size(); i++) {
                int correct = 0;
                int covered = 0;
                for (int instIndex = 0; instIndex < test.size(); instIndex++) {
                    Instance inst = test.instance(instIndex);
                    double labelVal = inst.value(test.classAttribute());
                    double[] distInst = this.distributions[ci][i];
                    boolean p = distInst[0] <= distInst[1];
                    if (ripper.getRuleset().get(i).covers(inst)) {
                        covered++;
                        if ((labelVal == 0 && !p) || (labelVal == 1 && p))
                            correct++;
                    }
                }
                //The rule quality here is defined as X/C
                //where X= amount of instances correctly determined by the rule
                //C = amount of instance the rule covers
                ruleQuality.add(correct / (double) test.size());
            }
            this.ruleQuality[ci] = ruleQuality.stream().mapToDouble(Double::doubleValue).toArray();
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
        int clssI = 0;
        for (int label : this.labels) {
            try {
                data.insts.setClassIndex(label);
                Instances instances = this.instanceTransformer.of(data.insts, this.labels, label);
                int instInd = 0;
                Classifier classifier = this.clss[clssI];
                for (Instance inst : instances) {
                    double[] distribution = classifier.distributionForInstance(inst);
                    double[] distribution2;
                    for (int i = 0; i < this.rules[clssI].length; ++i) {
                        Rule rule = this.rules[clssI][i];
                        double quality = this.ruleQuality[clssI][i];
                        if (rule.covers(inst)) {
                            distribution2 = this.distributions[clssI][i];
                            boolean reject = quality <= this.pConfidence;
                            if (distribution2[0] <= distribution2[1] || (rule.hasAntds() && reject))
                                result.computeIfAbsent(instInd, o -> new ArrayList<>()).add(label);
                            break;
                        }
                    }
                    instInd++;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            clssI++;
        }
        return new Output(data, this.labels, result);
    }

    public String getRules() {
        return "";
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
