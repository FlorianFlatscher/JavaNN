package deepLearning.neuralNetwork.data;

import deepLearning.neuralNetwork.data.label.DataGenerator;
import deepLearning.neuralNetwork.data.label.Label;

import java.util.ArrayList;
import java.util.Collections;

public class Dataset {
    private ArrayList<Label> labels;

    public Dataset(ArrayList<Label> labels) {
        this.labels = new ArrayList<>(labels);
    }

    public Dataset(DataGenerator dataGenerator) {
        labels = new ArrayList<>(dataGenerator.generateTrainingLabel());
    }

    public void shuffle() {
        Collections.shuffle(labels);
    }

    public ArrayList<Label> getLabels() {
        return labels;
    }

    public Label getTrainingLabel(int index) {
        return labels.get(index);
    }

    public int size() {
        return labels.size();
    }

    @Override
    public String toString() {
        return "Dataset{" +
                "labels=" + labels.toString() +
                '}';
    }
}
