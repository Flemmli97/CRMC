package io.github.flemmli97.dataset;

import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class UnlabelledSet {

    public final Instances insts;

    public UnlabelledSet(Path path) {
        try (BufferedReader reader = Files.newBufferedReader(path)) {
            //Read arff data. The instance is sparse
            ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);
            this.insts = arff.getData();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }

    public UnlabelledSet(Instances instances) {
        this.insts = instances;
    }
}
