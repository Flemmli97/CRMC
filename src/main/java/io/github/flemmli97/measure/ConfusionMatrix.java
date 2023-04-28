package io.github.flemmli97.measure;

public class ConfusionMatrix {

    private int truePositive, trueNegative, falsePositive, falseNegative;

    public void increaseTruePositive() {
        this.truePositive++;
    }

    public void increaseTrueNegative() {
        this.trueNegative++;
    }

    public void increaseFalsePositive() {
        this.falsePositive++;
    }

    public void increaseFalseNegative() {
        this.falseNegative++;
    }

    public int getTruePositive() {
        return this.truePositive;
    }

    public int getTrueNegative() {
        return this.trueNegative;
    }

    public int getFalsePositive() {
        return this.falsePositive;
    }

    public int getFalseNegative() {
        return this.falseNegative;
    }

    public double accuracy() {
        return BaseMeasures.accuracy(this.truePositive + this.trueNegative, this.truePositive + this.trueNegative + this.falsePositive + this.falseNegative);
    }

    public double precision() {
        return BaseMeasures.precision(this.truePositive, this.falsePositive);
    }

    public double recall() {
        return BaseMeasures.recall(this.truePositive, this.falseNegative);
    }

    public double f1() {
        return BaseMeasures.f1(this.truePositive, this.falsePositive, this.falseNegative);
    }

    public double MCC() {
        return BaseMeasures.MCC(this.truePositive, this.trueNegative, this.falsePositive, this.falseNegative);
    }
}
