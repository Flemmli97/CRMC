package io.github.flemmli97;

import io.github.flemmli97.api.learners.Learners;
import io.github.flemmli97.dataset.LabelledSet;
import io.github.flemmli97.learner.Learner;
import org.apache.commons.cli.Option;

import java.nio.file.Path;

public class Classifier {

    public static void main(String[] args) {
        Path path = Path.of("").toAbsolutePath();

        Option.builder();

        //--Create learning instance

        //--Create labelled instance
        String file = "./data/20NG-F.arff";
        //LabelledSet set = new LabelledSet(Path.of(file), 10);

        String f = "./data/CAL500";
        LabelledSet set2 = new LabelledSet(Path.of(f + ".arff"), Path.of(f + ".xml"));

        String learnerID = "RLCM";
        Settings settings = new Settings();

        Learner learner = Learners.getLearner(learnerID)
                .orElseThrow(() -> new RuntimeException("No such learner " + learnerID))
                .apply(settings);

        learner.learn(set2);

        System.out.println(learner);
        //System.out.println("INST============\n" +set2.insts);
        //System.out.println(Arrays.toString(set2.labels));
        //System.out.println(Arrays.deepToString(set2.dataLabels));
        //-- train learning instance from labelled instance

        //--Create unlabelled instance

        //--Tests: predict for unlabelled instances

    }
}
