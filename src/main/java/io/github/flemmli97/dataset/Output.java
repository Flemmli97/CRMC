package io.github.flemmli97.dataset;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Output {

    public final int dataID;

    public final int[] labels;

    public final Instance dataInstance;

    public Output(int dataID, int[] labels, Instance dataInstance) {
        this.dataID = dataID;
        this.labels = labels;
        Arrays.sort(this.labels);
        this.dataInstance = dataInstance;
    }

    public String labelsSimple() {
        StringBuilder builder = new StringBuilder();
        builder.append("{");
        int i = 0;
        for (int label : this.labels) {
            if (i != 0)
                builder.append(",");
            builder.append(label).append(" ").append(1);
            i++;
        }
        builder.append("}");
        return builder.toString();
    }

    public List<String> formattedLabels() {
        List<String> formatted = new ArrayList<>();
        for (int i : this.labels) {
            Attribute att = this.dataInstance.attribute(i);
            double value = this.dataInstance.value(att.index());
            if (att.isString() || att.isNominal()) {
                formatted.add(att.name() + "=" + att.value((int) value));
            } else if (att.isRelationValued()) {
                formatted.add(att.name() + "=" + att.relation((int) value).relationName());
            } else if (att.isDate()) {
                formatted.add(att.name() + "=" + att.formatDate(value));
            } else {
                formatted.add(att.name() + "=" + Utils.doubleToString(value, 2));
            }
        }
        return formatted;
    }
}
