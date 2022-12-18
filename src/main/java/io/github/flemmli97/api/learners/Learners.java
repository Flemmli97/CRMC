package io.github.flemmli97.api.learners;

import io.github.flemmli97.Settings;
import io.github.flemmli97.learner.Learner;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

public class Learners {

    private static Map<String, Function<Settings, Learner>> types = new HashMap<>();

    public static void registerNewLearner(String id, Function<Settings, Learner> factory) {
        types.put(id, factory);
    }

    public static Function<Settings, Learner> getLearner(String id) {
        return types.get(id);
    }
}
