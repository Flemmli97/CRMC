package io.github.flemmli97.learner;

import io.github.flemmli97.MultiLabelClassifier;
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
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Implementation of a BinaryRelevance classifier using wekas implementation of RIPPER {@link JRip}
 */
public class RuleMultiLabelLearner implements Learner {

    private final int threadCount;
    private double threshold;

    private boolean learned;

    private int[] labels;
    private Classifier[] classifiers;
    private double[][][] distributions;

    private Rule[][] rules;
    private RuleStats[][] ruleStats;
    private double[][][] instanceScores;

    private Instances[] training;

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

    private final boolean applyConformal, log, pruning;

    public RuleMultiLabelLearner(Settings settings) {
        this.threadCount = settings.threadCount;
        this.threshold = settings.confidenceP;
        this.applyConformal = settings.conformal;
        this.pruning = settings.pruning;
        this.log = settings.log;
    }

    public double threshold() {
        return this.threshold;
    }

    public void setThreshold(double val) {
        this.threshold = val;
    }

    @Override
    public void learn(LabelledSet set) {
        this.labels = set.labels;
        this.classifiers = new Classifier[this.labels.length];
        this.distributions = new double[this.classifiers.length][][];
        this.rules = new Rule[this.classifiers.length][];
        this.ruleStats = new RuleStats[this.classifiers.length][];
        this.instanceScores = new double[this.classifiers.length][][];
        this.training = new Instances[set.labels.length];
        for (int labelIndex = 0; labelIndex < this.labels.length; labelIndex++) {
            int label = this.labels[labelIndex];
            //Init classifier
            JRip ripper = new JRip();
            ripper.setUsePruning(this.pruning);
            Instances train = new Instances(set.insts);
            try {
                //Remove all labels except the current relevant one
                train.setClassIndex(label);
                train = this.instanceTransformer.of(train, set.labels, label);
                // JRIP rule learning
                ripper.buildClassifier(train);
            } catch (Exception e) {
                e.printStackTrace();
            }
            this.classifiers[labelIndex] = ripper;
            ArrayList<double[]> dist = ReflectionUtil.getField(ripper, "m_Distributions");
            this.distributions[labelIndex] = dist.toArray(double[][]::new);
            this.rules[labelIndex] = ripper.getRuleset().toArray(Rule[]::new);
            this.training[labelIndex] = train;
            //CP setup
            if (this.applyConformal) {
                List<RuleStats> stats = ReflectionUtil.getField(ripper, "m_RulesetStats");
                this.ruleStats[labelIndex] = stats.toArray(new RuleStats[0]);
                //Calculate conformity scores of instances
                List<Double> pos = new ArrayList<>();
                List<Double> neg = new ArrayList<>();
                for (Instance trainingInstance : train) {
                    double[] conf = this.conformity(train, trainingInstance, stats, labelIndex);
                    if (trainingInstance.classValue() == 0)
                        neg.add(conf[0]);
                    if (trainingInstance.classValue() == 1)
                        pos.add(conf[1]);
                }
                this.instanceScores[labelIndex] = new double[][]{neg.stream().mapToDouble(d -> d).toArray(), pos.stream().mapToDouble(d -> d).toArray()};
                //This is what JRIP uses
                /*List<double[][]> ripperDefaultDistribution = stats.stream().map(s->{
                    double[][] d = new double[s.getRulesetSize()][];
                    for(int i = 0; i < s.getRulesetSize(); i++) {
                        var rec = BaseMeasures.f1((int) s.getSimpleStats(i)[2], (int) s.getSimpleStats(i)[4], (int) s.getSimpleStats(i)[5]);
                        d[i] = new double[] {1-rec, rec};
                    }
                    return d;
                }).toList();*/
                /*Instances trainCpy = train;
                this.ruleData[labelIndex] = Arrays.stream(this.rules[labelIndex]).map(rule -> {
                    double[] ret = new double[4];
                    var attV = Double.parseDouble(classAttribute.value((int) rule.getConsequent()));
                    ret[0] = attV;
                    for (Instance inst : trainCpy) {
                        var instV = inst.value(classAttribute);
                        boolean covers = rule.covers(inst);
                        boolean matchesRule = attV == instV;
                        if (rule.covers(inst)) {
                            ret[1]++;
                            if (matchesRule) {
                                ret[2]++;
                                ret[3]++;
                            }
                        } else if (!matchesRule) {
                            ret[3]++;
                        }
                    }
                    return ret;
                }).toArray(double[][]::new);

                var ruzl = this.rules[labelIndex];
                double[][] rd = this.ruleData[labelIndex];
                double[][] distI = this.distributions[labelIndex];
                var conform = test.stream().map(i -> {
                    var instV = i.value(classAttribute);
                    int covers = 0;
                    int correct = 0;
                    double scoreP = Double.MIN_VALUE;
                    double scoreN = Double.MIN_VALUE;
                    for (int ri = 0; ri < ruzl.length; ri++) {
                        Rule rule = ruzl[ri];
                        double[] distIsub = distI[ri];
                        var attV = Double.parseDouble(classAttribute.value((int) rule.getConsequent()));
                        if (rule.covers(i)) {
                            double[] ruleData = rd[ri];
                            double correctV = ruleData[2];
                            if (instV == attV)
                                correct--;
                            double temp = correctV / (ruleData[1] - 1) - Math.sqrt(1f / (ruleData[1] - 1));
                            if (attV == 0 && temp > scoreN)
                                scoreN = temp;
                            if (attV == 1 && temp > scoreP)
                                scoreP = temp;
                            covers++;
                            if (instV == attV)
                                correct++;
                        }
                    }
                    return new double[]{scoreN, scoreP};//(double)correct / covers - Math.sqrt(1f/covers);
                }).toArray(double[][]::new);


                this.ruleScores2[labelIndex] = conform;/*Arrays.stream(this.ruleData[ci]).collect(Collector.of(() -> new Pair<>(new ArrayList<Double>(), new ArrayList<Double>()), (result, d) -> {
                            double measure = d[2] / d[1] - Math.sqrt(1 / (d[1]));
                            //double measure = d[3] / trainCpy.size();
                            if (d[0] == 0) {
                                result.first().add(measure);
                            } else if (d[0] == 1) {
                                result.second().add(measure);
                            }
                        }, (d1, d2) -> d1,
                        p -> new double[][]{p.first().stream().mapToDouble(f -> f).toArray(), p.second().stream().mapToDouble(f -> f).toArray()}));
                */
                if (this.log)
                    MultiLabelClassifier.log("Rule-scores " + Arrays.toString(this.ruleStats[labelIndex]));
            }
        }
        this.learned = true;
    }

    private double dot(double[] arr, double[] arr2, int ignore) {
        double sum = 0;
        for (int i = 0; i < arr.length; i++) {
            if (i == ignore)
                continue;
            sum += arr[i] * arr2[i];
        }
        return sum;
    }

    private double[] sub(double[] arr, double[] arr2) {
        double[] sub = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            sub[i] = arr[i] - arr2[i];
        }
        return sub;
    }

    private double[] conformity(Instances instances, Instance instance, List<RuleStats> stats, int labelIndex) {
        List<Pair<Double, Instance>> sorted = new ArrayList<>(instances).stream().map(i -> {
            double[] s = this.sub(instance.toDoubleArray(), i.toDoubleArray());
            return new Pair<>(Math.sqrt(this.dot(s, s, i.classIndex())), i);
        }).sorted(Comparator.comparing(Pair::first)).toList();
        int pos = 0;
        int neg = 0;
        List<Double> mP = new ArrayList<>();
        List<Double> mN = new ArrayList<>();
        for (Pair<Double, Instance> ii : sorted) {
            if (ii.second() == instance)
                continue;
            for (RuleStats stat : stats) {
                for (int iR = 0; iR < stat.getRulesetSize(); iR++) {
                    Rule rule = stat.getRuleset().get(iR);
                    if (rule.covers(ii.second())) {
                        double labelVal = ii.second().classValue();
                        if (labelVal == 1)
                            pos++;
                        else
                            neg++;
                        int n = pos + neg;
                        mP.add((double) pos / n - Math.sqrt(1f / n));
                        mN.add((double) neg / n - Math.sqrt(1f / n));
                    }
                }
            }
        }
        /*for (RuleStats stat : stats) {
            for (int iR = 0; iR < stat.getRulesetSize(); iR++) {
                Rule rule = stat.getRuleset().get(iR);
                double ruleOutput = Double.parseDouble(label.value((int) rule.getConsequent()));
                for(Instance ii : l) {
                    if(ii == instance)
                        continue;
                    //if (rule.covers(ii)) {
                        if(ruleOutput ==1)
                            pos++;
                        else
                            neg++;
                        int n = pos + neg;
                        mP.add((double) pos / n - Math.sqrt(1f / n));
                        mN.add((double) neg / n - Math.sqrt(1f / n));
                    //}
                }
            }
        }


        for (RuleStats stat : stats) {
            for (int iR = 0; iR < stat.getRulesetSize(); iR++) {
                Rule rule = stat.getRuleset().get(iR);
                double ruleOutput = Double.parseDouble(label.value((int) rule.getConsequent()));
                boolean match = ruleOutput == instLabel;
                if (rule.covers(instance)) {
                    double score = (stat.getSimpleStats(iR)[2] + (match ? -1 : 0)) / (stat.getSimpleStats(iR)[0] - 1) - Math.sqrt(1f / (stat.getSimpleStats(iR)[0] - 1));

                    if (ruleOutput == 1 && score > maxP)
                        maxP = score;
                    if (ruleOutput == 0 && score > maxN)
                        maxN = score;
                }
            }
        }
        /*for (int i = 0; i < rules.length; i++) {
            Rule rule = rules[i];
            double instLabel = instance.value(instance.classIndex());
            if(instLabel != Double.parseDouble(label.value((int) rule.getConsequent())))
                continue;
            double ruleQuality = this.ruleScores[labelIndex][i];
            if (rule.covers(instance)) {
                if (instLabel == 1 && ruleQuality > maxP)
                    maxP = ruleQuality;
                if (instLabel == 0 && ruleQuality > maxN)
                    maxN = ruleQuality;
            }
        }*/
        return new double[]{mN.stream().mapToDouble(d -> d).limit(mN.size() / 2).max().orElse(0),
                mP.stream().mapToDouble(d -> d).limit(mP.size() / 2).max().orElse(0)};
    }

    private double[] plausibility(Instances instances, Instance instance, List<RuleStats> stats, int labelIndex, Attribute label) {
        double[] conform = this.conformity(instances, instance, stats, labelIndex);
        double[][] instsScores = this.instanceScores[labelIndex];
        double cP = Arrays.stream(instsScores[1]).filter(d -> conform[1] >= d).count();
        double cN = Arrays.stream(instsScores[0]).filter(d -> conform[0] >= d).count();
        double pPos = cP / (double) instsScores[1].length;
        double pNeg = cN / (double) instsScores[0].length;
        return new double[]{pNeg, pPos};
    }

    @Override
    public Output predict(UnlabelledSet data) {
        if (!this.learned)
            throw new IllegalStateException("Model not learned");
        Map<Integer, List<Integer>> result = new HashMap<>();
        for (int labelIndex = 0; labelIndex < this.labels.length; labelIndex++) {
            int label = this.labels[labelIndex];
            data.insts.setClassIndex(label);
            Instances instances = data.insts;
            try {
                instances = this.instanceTransformer.of(data.insts, this.labels, label);
            } catch (Exception e) {
                e.printStackTrace();
            }
            Attribute classAttribute = ReflectionUtil.getField(this.classifiers[labelIndex], "m_Class");
            List<Double> pos = new ArrayList<>();
            List<Double> neg = new ArrayList<>();
            for (int instInd = 0; instInd < instances.size(); instInd++) {
                Instance inst = instances.get(instInd);
                double[] distribution;
                if (this.applyConformal) {
                    double[] scoreN = this.plausibility(this.training[labelIndex], inst, List.of(this.ruleStats[labelIndex]), labelIndex, classAttribute);
                    pos.add(scoreN[1]);
                    neg.add(scoreN[0]);
                    if (scoreN[1] >= this.threshold * scoreN[0]) {
                        result.computeIfAbsent(instInd, o -> new ArrayList<>()).add(label);
                    }
                } else {
                    for (int i = 0; i < this.rules[labelIndex].length; ++i) {
                        Rule rule = this.rules[labelIndex][i];
                        if (rule.covers(inst)) {
                            distribution = this.distributions[labelIndex][i];
                            if (distribution[1] >= distribution[0])
                                result.computeIfAbsent(instInd, o -> new ArrayList<>()).add(label);
                            break;
                        }
                    }
                }
            }
        }
        return new Output(data, this.labels, result, true);
    }

    public String getRules() {
        return Arrays.deepToString(this.classifiers);
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
