package io.github.flemmli97;

public class Settings {

    int maxRules;

    boolean multiThreaded;

    public Settings withMaxRules(int maxRules) {
        this.maxRules = maxRules;
        return this;
    }

    public Settings useMultithreading() {
        this.multiThreaded = true;
        return this;
    }
}
