package deepLearning.neuralNetwork.data.label;

import java.util.ArrayList;

public class AndGenerator implements DataGenerator {
    private int length;

    public AndGenerator(int length) {
        this.length = length;
    }

    @Override
    public ArrayList<Label> generateTrainingLabel() {
        ArrayList<Label> labels = new ArrayList<>();

//        labels.add(new Label(new double[]{0, 0}, new double[]{0}));
//        labels.add(new Label(new double[]{0, 1}, new double[]{1}));
//        labels.add(new Label(new double[]{1, 0}, new double[]{1}));
//        labels.add(new Label(new double[]{1, 1}, new double[]{0}));

        for (int i = 0; i < length; i++) {
            double a = Math.random() * 2 - 1;
            double b = Math.random() * 2 - 1;

            boolean signA = a > 0;
            boolean signB = b > 0;

            labels.add(new Label(new double[]{a,b}, new double[]{signA & signB ? 1 : -1}));
        }

        return labels;
    }
}
