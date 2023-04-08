package io.github.flemmli97.measure;

public class BaseMeasures {

    public static double accuracy(int correct, int total) {
        return correct / (double) total;
    }

    public static double precision(int trueP, int falseP) {
        return trueP / (double) (trueP + falseP);
    }

    public static double recall(int trueP, int falseN) {
        return trueP / (double) (trueP + falseN);
    }

    public static double f1(int trueP, int falseP, int falseN) {
        return 2 * precision(trueP, falseP) * recall(trueP, falseN) / (precision(trueP, falseP) + recall(trueP, falseN));
    }

    public static double MCC(int trueP, int trueN, int falseP, int falseN) {
        double d = (trueP * trueN) - (falseP * falseN);
        double d1 = trueP + falseP;
        double d2 = trueP + falseN;
        double d3 = trueN + falseP;
        double d4 = trueN + falseN;
        double d5 = d1 * d2 * d3 * d4;
        return d / Math.sqrt(d5);
    }
}
