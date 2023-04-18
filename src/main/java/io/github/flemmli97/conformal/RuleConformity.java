package io.github.flemmli97.conformal;

import io.github.flemmli97.learner.Learner;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Unused
 */
public class RuleConformity implements Conformal {
    @Override
    public double score(int label, Learner learner, Classifier classifier, Instances instances, Instance instance) {
        return 0;
    }
}
