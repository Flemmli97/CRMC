package io.github.flemmli97.learner;

import io.github.flemmli97.Settings;
import io.github.flemmli97.dataset.Data;
import io.github.flemmli97.dataset.LabelledSet;
import io.github.flemmli97.dataset.Output;

public class RuleMultiLabelLearner implements Learner {

    private boolean learned;

    public RuleMultiLabelLearner(Settings settings) {

    }

    @Override
    public void learn(LabelledSet set) {

        //IMPL
        int[] rules;
        //while(not all covered) do:
        {
            for (int l : set.labels) {

            }
        }
        this.learned = true;
    }

    @Override
    public Output predict(Data data) {
        if (!this.learned)
            throw new IllegalStateException();
        return null;
    }

    public String getRules() {
        return "";
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("Rule based Multilabel Learner");
        builder.append("Rules:");
        builder.append(getRules());
        return builder.toString();
    }
}
