package io.github.flemmli97.dataset;

import io.github.flemmli97.measure.ConfusionMatrix;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Output {

    public final UnlabelledSet dataSet;

    public final int[] labels;

    public final Map<Integer, List<Integer>> result;

    public final List<List<String>> instanceLabelsReadable;

    public final ConfusionMatrix confusionMatrix;

    public Output(UnlabelledSet dataSet, int[] labels, Map<Integer, List<Integer>> result) {
        this.dataSet = dataSet;
        this.labels = labels;
        this.result = result;
        this.confusionMatrix = new ConfusionMatrix();
        this.instanceLabelsReadable = result.entrySet().stream().sorted(Map.Entry.comparingByKey()).map(e -> {
            Instance inst = dataSet.insts.get(e.getKey());
            return this.process(inst, this.confusionMatrix, e.getValue());
        }).toList();
    }

    private List<String> process(Instance inst, ConfusionMatrix matrix, List<Integer> labels) {
        List<String> formatted = new ArrayList<>();
        for (int i : this.labels) {
            Attribute att = inst.attribute(i);
            double value = inst.value(att.index());
            String eq;
            if (labels.contains(i)) {
                if (value == 1) {
                    matrix.increaseTruePositive();
                    eq = "=";
                } else {
                    matrix.increaseFalsePositive();
                    eq = "!=";
                }
            } else {
                if (value != 1) {
                    matrix.increaseTrueNegative();
                    eq = "=";
                } else {
                    matrix.increaseFalseNegative();
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
