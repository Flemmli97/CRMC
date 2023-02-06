package io.github.flemmli97.api.learners;

import io.github.flemmli97.Settings;
import io.github.flemmli97.learner.Learner;
import io.github.flemmli97.learner.RuleMultiLabelLearner;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;

public class Learners {

    private static final Map<String, Function<Settings, Learner>> TYPES = new HashMap<>();

    /**
     * Register a new learner with an id here
     */
    public static void registerNewLearner(String id, Function<Settings, Learner> factory) {
        TYPES.put(id, factory);
    }

    public static Optional<Function<Settings, Learner>> getLearner(String id) {
        return Optional.ofNullable(TYPES.get(id));
    }

    static {
        registerNewLearner("RLCM", RuleMultiLabelLearner::new);
    }
}
