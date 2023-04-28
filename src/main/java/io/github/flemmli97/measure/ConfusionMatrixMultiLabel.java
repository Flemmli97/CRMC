package io.github.flemmli97.measure;

import java.util.stream.Stream;

public class ConfusionMatrixMultiLabel {

    private final int[] labels;

    private int truePositive, trueNegative, falsePositive, falseNegative;
    private final ConfusionMatrix[] labelMatrices;

    public ConfusionMatrixMultiLabel(int[] labels) {
        this.labels = labels;
        this.labelMatrices = Stream.generate(ConfusionMatrix::new).limit(labels.length).toArray(ConfusionMatrix[]::new);
    }

    public void increaseTruePositive(int labelIndex) {
        this.truePositive++;
        this.labelMatrices[labelIndex].increaseTruePositive();
    }

    public void increaseTrueNegative(int labelIndex) {
        this.trueNegative++;
        this.labelMatrices[labelIndex].increaseTrueNegative();
    }

    public void increaseFalsePositive(int labelIndex) {
        this.falsePositive++;
        this.labelMatrices[labelIndex].increaseFalsePositive();
    }

    public void increaseFalseNegative(int labelIndex) {
        this.falseNegative++;
        this.labelMatrices[labelIndex].increaseFalseNegative();
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

    public double macroAccuracy() {
        double d = 0;
        for (ConfusionMatrix labelMatrix : this.labelMatrices)
            d += labelMatrix.accuracy();
        return d / this.labels.length;
    }

    public double precision() {
        return BaseMeasures.precision(this.truePositive, this.falsePositive);
    }

    public double macroPrecision() {
        double d = 0;
        for (ConfusionMatrix labelMatrix : this.labelMatrices)
            d += labelMatrix.precision();
        return d / this.labels.length;
    }

    public double recall() {
        return BaseMeasures.recall(this.truePositive, this.falseNegative);
    }

    public double macroRecall() {
        double d = 0;
        for (ConfusionMatrix labelMatrix : this.labelMatrices)
            d += labelMatrix.recall();
        return d / this.labels.length;
    }

    public double f1() {
        return BaseMeasures.f1(this.truePositive, this.falsePositive, this.falseNegative);
    }

    public double macroF1() {
        double d = 0;
        for (ConfusionMatrix labelMatrix : this.labelMatrices)
            d += labelMatrix.f1();
        return d / this.labels.length;
    }

    public double MCC() {
        return BaseMeasures.MCC(this.truePositive, this.trueNegative, this.falsePositive, this.falseNegative);
    }
}
