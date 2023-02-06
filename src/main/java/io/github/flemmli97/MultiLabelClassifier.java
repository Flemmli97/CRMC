package io.github.flemmli97;

import io.github.flemmli97.dataset.LabelledSet;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

import java.nio.file.Path;
import java.util.Arrays;

public class MultiLabelClassifier {

    public static void main(String[] args) {
        Path path = Path.of("").toAbsolutePath();

        Options options = new Options();

        Option.builder();

        //--Create learning instance

        //--Create labelled instance
        String file = "./data/20NG-F.arff";
        //LabelledSet set = new LabelledSet(Path.of(file), 10);

        String f = "./data/CAL500";
        LabelledSet set2 = new LabelledSet(Path.of(f + ".arff"), Path.of(f + ".xml"));

        String learnerID = "RLCM";
        Settings settings = new Settings();

        //Learner learner = Learners.getLearner(learnerID)
        //        .orElseThrow(() -> new RuntimeException("No such learner " + learnerID))
        //        .apply(settings);

        //learner.learn(set2);

        //System.out.println(set2.insts.size());
        //System.out.println(set2.dataLabels.length);

        System.out.println(set2.labels());

        System.out.println(Arrays.toString(set2.labels));
        System.out.println(Arrays.deepToString(set2.dataLabels));
        System.out.println(Arrays.deepToString(set2.labelData));

        for (int labelIndex = 0; labelIndex < set2.labelData.length; labelIndex++) {
            int[] data = set2.labelData[labelIndex];
            for (int dataIndex : data) {
                boolean contains = false;
                for (int label : set2.dataLabels[dataIndex]) {
                    if (label == set2.labels[labelIndex])
                        contains = true;
                }
                if (!contains)
                    System.out.println("WRONG for " + labelIndex);
            }
        }
        //System.out.println(Arrays.deepToString(set2.dataFeatures));
        //System.out.println(Arrays.toString(set2.labels));

        //System.out.println(learner);
        //System.out.println("INST============\n" +set2.insts);
        //System.out.println(Arrays.toString(set2.labels));
        //System.out.println(Arrays.deepToString(set2.dataLabels));
        //-- train learning instance from labelled instance

        //--Create unlabelled instance

        //--Tests: predict for unlabelled instances

    }

    public static void main(Option option) {

    }
}
