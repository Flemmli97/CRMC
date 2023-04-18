package io.github.flemmli97.dataset;

import com.google.common.collect.ImmutableList;
import io.github.flemmli97.Pair;
import io.github.flemmli97.measure.ConfusionMatrixMultiLabel;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Output {

    public final UnlabelledSet dataSet;

    public final int[] labels;

    public final Map<Integer, List<Integer>> result;

    public final List<Pair<Integer, List<String>>> instanceLabelsReadable;

    public final ConfusionMatrixMultiLabel confusionMatrix;

    private int hammingCounter;

    public List<String> trueP;

    private final boolean verify;

    public Output(UnlabelledSet dataSet, int[] labels, Map<Integer, List<Integer>> result) {
        this(dataSet, labels, result, false);
    }

    public Output(UnlabelledSet dataSet, int[] labels, Map<Integer, List<Integer>> result, boolean verify) {
        this.verify = verify;
        this.dataSet = dataSet;
        this.labels = labels;
        this.result = result;
        this.confusionMatrix = new ConfusionMatrixMultiLabel(labels);
        ImmutableList.Builder<Pair<Integer, List<String>>> builder = ImmutableList.builder();
        for (int i = 0; i < dataSet.insts.size(); i++) {
            Instance inst = dataSet.insts.get(i);
            builder.add(new Pair<>(i, this.process(inst, this.confusionMatrix, result.getOrDefault(i, List.of()))));
        }
        this.instanceLabelsReadable = builder.build();
        this.trueP = this.trueP(dataSet.insts);
    }

    public double hammingLoss() {
        return this.hammingCounter / (double) (this.labels.length * this.dataSet.insts.size());
    }

    private List<String> trueP(Instances insts) {
        List<String> trueP = new ArrayList<>();
        int instId = 0;
        for (Instance inst : insts) {
            String s = "";
            for (int i : this.labels) {
                Attribute att = inst.attribute(i);
                double value = inst.value(att.index());
                if (value == 1) {
                    if (att.isString() || att.isNominal()) {
                        s = att.name() + "=" + att.value((int) value);
                    } else if (att.isRelationValued()) {
                        s = att.name() + "=" + att.relation((int) value).relationName();
                    } else if (att.isDate()) {
                        s = att.name() + "=" + att.formatDate(value);
                    } else {
                        s = att.name() + "=" + Utils.doubleToString(value, 2);
                    }
                }
            }
            if (!s.isEmpty())
                trueP.add(instId + ": " + s);
            instId++;
        }
        return trueP;
    }

    private List<String> process(Instance inst, ConfusionMatrixMultiLabel matrix, List<Integer> labels) {
        List<String> formatted = new ArrayList<>();
        for (int i = 0; i < this.labels.length; i++) {
            int label = this.labels[i];
            Attribute att = inst.attribute(label);
            double value = inst.value(att.index());
            double prediction = labels.contains(label) ? 1 : 0;
            this.hammingCounter += (int) value ^ (int) prediction;
            String eq;
            if (prediction == 1) {
                if (value == 1) {
                    matrix.increaseTruePositive(i);
                    eq = "=";
                } else {
                    matrix.increaseFalsePositive(i);
                    eq = "!=";
                }
            } else {
                if (value != 1) {
                    matrix.increaseTrueNegative(i);
                    eq = "=";
                } else {
                    matrix.increaseFalseNegative(i);
                    eq = "!=";
                }
            }
            if (att.isString() || att.isNominal()) {
                formatted.add(att.name() + eq + att.value((int) value));
            } else if (att.isRelationValued()) {
                formatted.add(att.name() + eq + att.relation((int) value).relationName());
            } else if (att.isDate()) {
                formatted.add(att.name() + eq + att.formatDate(value));
            } else {
                formatted.add(att.name() + eq + Utils.doubleToString(value, 2));
            }
        }
        return formatted;
    }
}
