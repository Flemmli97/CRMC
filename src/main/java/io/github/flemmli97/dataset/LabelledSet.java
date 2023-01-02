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
import java.util.List;

public class LabelledSet extends UnlabelledSet {

    /**
     * Index of all labels
     */
    public int[] labels;

    /**
     * Array of datasets mapped to the matching labels
     */
    public int[][] dataLabels;

    public double[][] dataFeatures;

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
     * Creates a new labelled dataset from a given path and an xml file using the mulan data format
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

    private void setupLabels() {
        this.dataLabels = new int[this.insts.numInstances()][];
        this.dataFeatures = new double[this.insts.numInstances()][];
        for (int i = 0; i < this.insts.size(); i++) {
            Instance inst = this.insts.get(i);
            List<Integer> l = new ArrayList<>();
            List<Double> f = new ArrayList<>();
            boolean b = false;
            for (int label : this.labels) {
                Attribute att = inst.attribute(label);
                for (int vals = 0; vals < inst.numValues(); vals++) {
                    boolean isNull = (att.isString() || att.isRelationValued() || inst.value(label) != 0);
                    int attI = inst.index(vals);
                    if (attI == label && isNull)
                        l.add(label);
                    if (!b)
                        f.add(inst.value(attI));
                }
                b = true;
            }
            this.dataLabels[i] = l.stream().mapToInt(Integer::intValue).toArray();
            this.dataFeatures[i] = f.stream().mapToDouble(Double::doubleValue).toArray();
            i++;
        }
    }

    public static Document parseXML(Path path) throws IOException, ParserConfigurationException, SAXException {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        try (InputStream input = Files.newInputStream(path)) {
            return factory.newDocumentBuilder().parse(input);
        }
    }
}
