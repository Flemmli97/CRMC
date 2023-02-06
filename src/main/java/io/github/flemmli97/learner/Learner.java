package io.github.flemmli97.learner;

import io.github.flemmli97.dataset.LabelledSet;
import io.github.flemmli97.dataset.Output;
import io.github.flemmli97.dataset.UnlabelledSet;

public interface Learner {

    /**
     * Fit this learner to the given labelled set
     *
     * @param set
     */
    void learn(LabelledSet set);

    /**
     * Predict the labels of the given data using this learner
     * TODO: turn to list like. as atm its single output
     *
     * @param data
     * @return
     */
    Output predict(UnlabelledSet data);

}
