package io.github.flemmli97.learner;

import io.github.flemmli97.dataset.Data;
import io.github.flemmli97.dataset.LabelledSet;
import io.github.flemmli97.dataset.Output;

public interface Learner {

    /**
     * Fit this learner to the given labelled set
     *
     * @param set
     */
    void learn(LabelledSet set);

    /**
     * Predict the labels of the given data using this learner
     *
     * @param data
     * @return
     */
    Output predict(Data data);

}
