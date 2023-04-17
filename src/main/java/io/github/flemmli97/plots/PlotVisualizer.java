package io.github.flemmli97.plots;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonConfig;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import io.github.flemmli97.Pair;
import io.github.flemmli97.dataset.Output;

import java.io.IOException;
import java.util.ArrayList;
import java.util.stream.Collectors;

public class PlotVisualizer {

    public static void plotF1(ArrayList<Pair<Double, Output>> outputs) {
        Plot plot = Plot.create(PythonConfig.pythonBinPathConfig("/usr/bin/python3"));
        plot.title("F-Measure");
        plot.xlabel("threshold");
        plot.ylabel("F-Measure");
        plot.plot().add(
                        outputs.stream().map(Pair::first).collect(Collectors.toList()),
                        outputs.stream().map(p -> {
                            double d = p.second().confusionMatrix.f1();
                            if (Double.isNaN(d))
                                return 0;
                            return d;
                        }).collect(Collectors.toList())
                )
                .color("blue")
                .linewidth(1)
                .build();
        try {
            plot.show();
        } catch (IOException | PythonExecutionException e) {
            e.printStackTrace();
        }
    }
}
