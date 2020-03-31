package deepLearning.neuralNetwork.data.label;

import java.util.Arrays;

public class Label {
    private double[] input;
    private double[] target;

    public Label(double[] input, double[] target) {
        this.input = input;
        this.target = target;
    }

    public double[] getInput() {
        return input;
    }

    public void setInput(double[] input) {
        this.input = input;
    }

    public double[] getTarget() {
        return target;
    }

    public void setTarget(double[] target) {
        this.target = target;
    }

    @Override
    public String toString() {
        return "TrainingLabel{" +
                "input=" + Arrays.toString(input) +
                ", target=" + Arrays.toString(target) +
                '}';
    }
}
