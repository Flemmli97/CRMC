package io.github.flemmli97.learner;

import io.github.flemmli97.Pair;
import io.github.flemmli97.Settings;
import io.github.flemmli97.dataset.LabelledSet;
import io.github.flemmli97.dataset.Output;
import io.github.flemmli97.dataset.UnlabelledSet;
import io.github.flemmli97.reflection.ReflectionUtil;
import weka.classifiers.Classifier;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.Rule;
import weka.classifiers.rules.RuleStats;
import weka.core.Attribute;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collector;

/**
 * Implementation of a BinaryRelevance classifier using wekas implementation of RIPPER {@link JRip}
 */
public class RuleMultiLabelLearner implements Learner {

    private int threadCount;
    private double pConfidence;

    private boolean learned;

    private Rule[][] rules;
    private double[][][] ruleData;
    private double[][][] ruleScores;

    private double[][][] distributions;
    private Classifier[] clss;
    private int[] labels;
    private Instances training;

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
        this.training = set.insts;
        this.clss = new Classifier[set.labels.length];
        this.rules = new Rule[this.clss.length][];
        this.distributions = new double[this.clss.length][][];
        this.ruleData = new double[this.clss.length][][];
        this.ruleScores = new double[this.clss.length][][];
        int ci = 0;
        this.labels = set.labels;
        Random random = new Random();
        for (int label : set.labels) {
            JRip ripper = new JRip();
            Instances main = new Instances(set.insts);
            random.setSeed(0);
            int split = main.size() - main.size() / 3;
            Instances train = new Instances(main, split);
            for (int i = 0; i < split; i++) {
                train.add(main.instance(i));
            }
            Instances test = new Instances(main, main.size() - split);
            for (int i = 0; i < main.size() - split; i++) {
                train.add(main.instance(split + i));
            }
            try {
                train.setClassIndex(label);
                train = this.instanceTransformer.of(train, set.labels, label);
                test.setClassIndex(label);
                test = this.instanceTransformer.of(test, set.labels, label);
                ripper.buildClassifier(train);
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
                if (this.log) {
                    List<Pair<String, String>> stat = stats.stream().map(s -> {
                        ArrayList<Pair<String, String>> val = new ArrayList<>();
                        for (int i = 0; i < s.getRulesetSize(); i++) {
                            double[] d = s.getSimpleStats(i);
                            val.add(new Pair<>(classAttribute.value((int) s.getRuleset().get(i).getConsequent()), Arrays.toString(d)));
                        }
                        return val;
                    }).flatMap(Collection::stream).toList();
                    System.out.println("STATS " + stat);
                }
                Instances instCopy = train;
                this.ruleData[ci] = Arrays.stream(this.rules[ci]).map(rule -> {
                    double[] ret = new double[3];
                    var attV = Double.parseDouble(classAttribute.value((int) rule.getConsequent()));
                    ret[0] = attV;
                    for (Instance inst : instCopy) {
                        if (rule.covers(inst)) {
                            ret[1]++;
                            var instV = inst.value(classAttribute);
                            if (attV == instV)
                                ret[2]++;
                        }
                    }
                    return ret;
                }).toArray(double[][]::new);
                this.ruleScores[ci] = Arrays.stream(this.ruleData[ci]).collect(Collector.of(() -> new Pair<>(new ArrayList<Double>(), new ArrayList<Double>()), (result, d) -> {
                            if (d[0] == 0) {
                                result.first().add(d[2] / d[1] - Math.sqrt(1 / (d[1])));
                            } else if (d[0] == 1) {
                                result.second().add(d[2] / d[1] - Math.sqrt(1 / (d[1])));
                            }
                        }, (d1, d2) -> d1,
                        p -> new double[][]{p.first().stream().mapToDouble(f -> f).toArray(), p.second().stream().mapToDouble(f -> f).toArray()}));
                if (this.log)
                    System.out.println("Rule-scores " + Arrays.toString(this.ruleScores[ci]));
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
        DistanceFunction f = new EuclideanDistance();
        int classIndex = 0;
        for (int label : this.labels) {
            try {
                data.insts.setClassIndex(label);
                Instances instances = this.instanceTransformer.of(data.insts, this.labels, label);
                this.training.setClassIndex(label);
                Instances train = this.instanceTransformer.of(this.training, this.labels, label);
                f.setInstances(train);

                Attribute classAttribute = ReflectionUtil.getField(this.clss[classIndex], "m_Class");
                Pair<ArrayList<Instance>, ArrayList<Instance>> mappedInsts = data.insts.stream()
                        .collect(() -> new Pair<>(new ArrayList<>(), new ArrayList<>()), (i, inst) -> {
                            if (inst.value(label) == 0)
                                i.first().add(inst);
                            else
                                i.second().add(inst);
                        }, (i, i2) -> {
                        });
                int instInd = 0;
                for (Instance inst : instances) {
                    double[] distribution;
                    if (this.applyConformal) {
                        double[] scoreP = this.scoreFor(inst, classIndex, classAttribute, 1);
                        double[] scoreN = this.scoreFor(inst, classIndex, classAttribute, 0);
                        int indC = classIndex;
                        ArrayList<Instance> match = scoreP[2] == 0 ? mappedInsts.first() : mappedInsts.second();
                        var t = match.stream().map(i->this.scoreFor(i, indC, classAttribute, scoreP[2])).toList();
                        var cou = match.stream().filter(i->scoreP[0] >= this.scoreFor(i, indC, classAttribute, scoreP[2])[0]).count();
                        double pp = (double) mappedInsts.second().stream().filter(i->scoreP[0] >= this.scoreFor(i, indC, classAttribute, scoreP[2])[0]).count() / mappedInsts.second().size();
                        double pn = (double) mappedInsts.first().stream().filter(i->scoreN[0] >= this.scoreFor(i, indC, classAttribute, scoreN[2])[0]).count() / mappedInsts.first().size();
                        if (pp >= this.pConfidence * pn) {
                            result.computeIfAbsent(instInd, o -> new ArrayList<>()).add(label);
                        }
                    } else {
                        for (int i = 0; i < this.rules[classIndex].length; ++i) {
                            Rule rule = this.rules[classIndex][i];
                            if (rule.covers(inst)) {
                                distribution = this.distributions[classIndex][i];
                                if (distribution[1] >= distribution[0])
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
        return new Output(data, this.labels, result, true);
    }

    private double[] scoreFor(Instance instance, int classIndex, Attribute classAtt, double match) {
        double scoreC = 0;
        double ruleID = 0;
        double val = match;
        int offsetP = 0;
        int offsetN = 0;
        for (int i = 0; i < this.rules[classIndex].length; ++i) {
            Rule rule = this.rules[classIndex][i];
            double[] ruleData = this.ruleData[classIndex][i];
            double score;
            if(ruleData[0] == 0) {
                score = this.ruleScores[classIndex][0][i - offsetN];
                offsetP++;
            } else {
                score = this.ruleScores[classIndex][1][i - offsetP];
                offsetN++;
            }
            double att = Double.parseDouble(classAtt.value((int) rule.getConsequent()));
            if (rule.covers(instance) && (match == -1 || att == match)){// && score > (1-this.pConfidence)) {
                scoreC = score;
                ruleID = i;
                val = Double.parseDouble(classAtt.value((int) rule.getConsequent()));
                                /*distribution = this.distributions[classIndex][i];
                                if (distribution[1] >= distribution[0]) {
                                    result.computeIfAbsent(instInd, o -> new ArrayList<>()).add(label);
                                    p++;
                                }
                                break;*/
                break;
            }
        }
        return new double[] {scoreC, ruleID, val};
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
}
