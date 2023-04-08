package io.github.flemmli97;

/**
 * Settings holder for various learner
 */
public class Settings {

    public int maxRules;

    public int threadCount = 1;
    public double confidenceP = 0.6;

    public Settings withMaxRules(int maxRules) {
        this.maxRules = maxRules;
        return this;
    }

    public Settings threadCount(int threadCount) {
        this.threadCount = threadCount;
        return this;
    }

    public Settings withConfidenceP(float confidenceP) {
        this.confidenceP = confidenceP;
        return this;
    }
}
