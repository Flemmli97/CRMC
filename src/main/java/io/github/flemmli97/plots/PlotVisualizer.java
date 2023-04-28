package io.github.flemmli97.plots;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonConfig;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import io.github.flemmli97.Pair;
import io.github.flemmli97.dataset.Output;

import java.io.IOException;
import java.util.ArrayList;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

public class PlotVisualizer {

    public static void plot(String title, String xLabel, String yLabel, Consumer<Plot> cons) {
        Plot plot = Plot.create(PythonConfig.pythonBinPathConfig("/usr/bin/python3"));
        plot.title(title);
        cons.accept(plot);
        new Thread(() -> {
            try {
                plot.show();
            } catch (IOException | PythonExecutionException e) {
                e.printStackTrace();
            }
        }).start();
    }

    public static void plot(ArrayList<Pair<Double, Output>> outputs, String title, String measureLabel, Function<Output, Double> measure) {
        Plot plot = Plot.create(PythonConfig.pythonBinPathConfig("/usr/bin/python3"));
        plot.title(title);
        plot.xlabel("threshold");
        plot.ylabel(measureLabel);
        plot.plot().add(
                        outputs.stream().map(Pair::first).collect(Collectors.toList()),
                        outputs.stream().map(p -> {
                            double d = measure.apply(p.second());
                            if (Double.isNaN(d))
                                return 0;
                            return d;
                        }).collect(Collectors.toList())
                )
                .color("blue")
                .linewidth(1)
                .build();
        new Thread(() -> {
            try {
                plot.show();
            } catch (IOException | PythonExecutionException e) {
                e.printStackTrace();
            }
        }).start();
    }

    public static void plotF1(ArrayList<Pair<Double, Output>> outputs) {
        plot(outputs, "F1-Measure", "F1-Measure", o -> o.confusionMatrix.f1());
    }
}
