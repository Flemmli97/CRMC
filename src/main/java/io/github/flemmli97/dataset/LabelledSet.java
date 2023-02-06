package io.github.flemmli97.dataset;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;
import weka.core.Attribute;
import weka.core.Instance;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class LabelledSet extends UnlabelledSet {

    /**
     * Index of all labels
     */
    public int[] labels;

    /**
     * Array of datasets mapped to the matching labels
     */
    public int[][] dataLabels;

    /**
     * Array of labels matched to their data
     */
    public int[][] labelData;

    /**
     * Map of features to data instances
     */
    public double[][] dataFeatures;

    //public Map<Feature,List of pair:data instance, feature val>

    public Set<Attribute>[] dataFeaturess;

    /**
     * Creates a new labelled dataset from a given path
     *
     * @param path     The path to the arff file
     * @param labelNum The amount of attributes that serves as labels going from the first one
     */
    public LabelledSet(Path path, int labelNum) {
        super(path);
        this.labels = new int[labelNum];
        if (labelNum > this.insts.numAttributes())
            throw new IllegalStateException();
        int i = 0;
        var it = this.insts.enumerateAttributes();
        while (i < labelNum) {
            this.labels[i] = it.nextElement().index();
            i++;
        }
        this.setupLabels();
    }

    /**
     * Creates a new labelled dataset from a given path and a xml file using the mulan data format
     *
     * @param path    The path to the arff file
     * @param xmlFile The xml file defining the labels for the arff file
     */
    public LabelledSet(Path path, Path xmlFile) {
        super(path);
        Document xml;
        try {
            xml = parseXML(xmlFile);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
        //TODO label dependency?
        NodeList labelsNode = xml.getDocumentElement().getElementsByTagName("label");
        this.labels = new int[labelsNode.getLength()];
        for (int i = 0; i < labelsNode.getLength(); i++) {
            Node node = labelsNode.item(i);
            if (node.getNodeType() == Node.ELEMENT_NODE) {
                Element e = (Element) node;
                this.labels[i] = this.insts.attribute(e.getAttribute("name")).index();
            }
        }
        this.setupLabels();
    }

    public static Document parseXML(Path path) throws IOException, ParserConfigurationException, SAXException {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        try (InputStream input = Files.newInputStream(path)) {
            return factory.newDocumentBuilder().parse(input);
        }
    }

    private void setupLabels() {
        this.dataLabels = new int[this.insts.numInstances()][];
        this.dataFeatures = new double[this.insts.numInstances()][];
        Map<Integer, List<Integer>> labelData = new HashMap<>();
        for (int dataIndex = 0; dataIndex < this.insts.size(); dataIndex++) {
            Instance inst = this.insts.get(dataIndex);
            List<Integer> l = new ArrayList<>();
            List<Double> f = new ArrayList<>();
            boolean b = false;
            for (int labelIndex = 0; labelIndex < this.labels.length; labelIndex++) {
                int label = this.labels[labelIndex];
                Attribute att = inst.attribute(label);
                for (int vals = 0; vals < inst.numValues(); vals++) {
                    boolean has = (att.isString() || att.isRelationValued() || inst.value(label) != 0);
                    int attI = inst.index(vals);
                    if (attI == label && has) {
                        l.add(label);
                        labelData.computeIfAbsent(labelIndex, key -> new ArrayList<>())
                                .add(dataIndex);
                    }
                    if (!b)
                        f.add(inst.value(attI));
                }
                b = true;
            }
            this.dataLabels[dataIndex] = l.stream().mapToInt(Integer::intValue).toArray();
            this.dataFeatures[dataIndex] = f.stream().mapToDouble(Double::doubleValue).toArray();
        }
        this.labelData = labelData.entrySet().stream()
                .sorted(Comparator.comparingInt(Map.Entry::getKey))
                .map(e -> e.getValue().stream().mapToInt(Integer::intValue).toArray())
                .toArray(int[][]::new);
    }

    /**
     * @return Get the labels associated with this dataset in readable form
     */
    public String labels() {
        StringBuilder builder = new StringBuilder();
        builder.append("Labels:[");
        boolean start = true;
        for (int label : this.labels) {
            Attribute att = this.insts.attribute(label);
            if (!start) {
                builder.append(", ");
            }
            start = false;
            builder.append(att.name());
        }
        builder.append("]");
        return builder.toString();
    }
}
