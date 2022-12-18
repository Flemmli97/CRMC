package io.github.flemmli97.learner;

import io.github.flemmli97.dataset.Data;
import io.github.flemmli97.dataset.LabelledSet;
import io.github.flemmli97.dataset.Output;

public interface Learner {

    void learn(LabelledSet set);

    Output predict(Data data);

}
