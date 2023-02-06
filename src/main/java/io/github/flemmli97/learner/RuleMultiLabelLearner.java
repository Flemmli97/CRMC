package io.github.flemmli97.learner;

import io.github.flemmli97.Settings;
import io.github.flemmli97.dataset.LabelledSet;
import io.github.flemmli97.dataset.Output;
import io.github.flemmli97.dataset.UnlabelledSet;
import io.github.flemmli97.rules.RuleSet;
import weka.core.Attribute;

import java.util.ArrayList;
import java.util.List;

public class RuleMultiLabelLearner implements Learner {

    private boolean learned;
    private final List<RuleSet> ruleSets = new ArrayList<>();

    public RuleMultiLabelLearner(Settings settings) {

    }

    @Override
    public void learn(LabelledSet set) {
        //IMPL
        while (!this.covered(set, this.ruleSets)) {
            for (int label : set.labels) {
                RuleSet ruleSet = new RuleSet();
                this.tryLearnRule(set, label, ruleSet);
                this.ruleSets.add(ruleSet);
            }
        }
        this.learned = true;
    }

    private void tryLearnRule(LabelledSet set, int label, RuleSet rules) {
        int[] positiveInstances = set.labelData[label];

        while (true) { //not covered or stop condition
            for (int instanceIndex = 0; instanceIndex < positiveInstances.length; instanceIndex++) {
                double[] features = set.dataFeatures[instanceIndex]; //Get features of instance
                //Select feature
                // If rule in RuleSet clash with instance refine/adapt rule. Else select random feature and add to rule
                //Create new rule from feature or refine if present
                //rules.ruleMap.compute()
            }
            break; //remove
        }


        Attribute att = set.insts.attribute(label);

        for (int instanceIndex = 0; instanceIndex < set.dataFeatures.length; instanceIndex++) {
            double[] features = set.dataFeatures[instanceIndex]; //Get features of instance
            boolean positive = false;
            for (int labelIndex : set.labelData[instanceIndex]) {
                if (labelIndex == att.index()) {
                    positive = true; //This data instance has this label assigned
                    break;
                }
            }

            for (double val : features) { //Select feature
                //If RuleSet does not cover this instance:
                // If rule in RuleSet clash with instance refine/adapt rule. Else select random feature and add to rule
                //Create new rule from feature or refine if present
                //rules.ruleMap.compute()
            }
        }

        /*int[] sets = set.dataFeatures.labelData[att.index()]; //Get instances that are positive for the label
        for(int instance : sets) {
            double[] features = set.dataFeatures[instance]; //Get features of instance

            for(double val : features) { //Select feature
                //Create new rule from feature or refine if present
                //rules.ruleMap.compute()


            }
        }


        for(int instance : sets) {
            //Check if rule applies to all (or at least near all) instances
        }


        /*while (true) { //Stop condition
            for (int dataIndex = 0; dataIndex < set.dataLabels.length; dataIndex++) {
                int[] labels = set.dataLabels[dataIndex];
                for (int labelIndex = 0; labelIndex < labels.length; labelIndex++) {
                    int label = labels[labelIndex];
                    if (label == att.index()) { //Dataset contains matching label
                        //Attributes for this data instance
                        Set<Attribute> v = set.dataFeaturess[dataIndex];
                        //Rule
                        //Add to rules
                        break;
                    }
                }
            }
        }*/
    }

    private boolean covered(LabelledSet set, List<RuleSet> rule) {
        return false;
    }

    @Override
    public Output predict(UnlabelledSet data) {
        if (!this.learned)
            throw new IllegalStateException();
        //Apply conformal prediction when evaluating labels here. Note: Use new learner class
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
