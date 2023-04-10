package io.github.flemmli97.plots;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonConfig;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import io.github.flemmli97.dataset.Output;
import io.github.flemmli97.learner.RuleMultiLabelLearner;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

public class PlotVisualizer {

    public static void plotF1(List<PlotPair> outputs) {
        Plot plot = Plot.create(PythonConfig.pythonBinPathConfig("/usr/bin/python3"));
        plot.title("F-Measure");
        plot.xlabel("threshold");
        plot.ylabel("F-Measure");
        plot.plot().add(
                    outputs.stream().map(PlotPair::threshold).collect(Collectors.toList()),
                    outputs.stream().map(p->p.output.confusionMatrix.f1()).collect(Collectors.toList())
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

    public record PlotPair(double threshold, Output output) {

    }
}
