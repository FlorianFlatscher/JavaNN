package deepLearning.neuralNetwork.data.label;

import java.util.ArrayList;

public interface DataGenerator {
    ArrayList<Label> generateTrainingLabel();
}
