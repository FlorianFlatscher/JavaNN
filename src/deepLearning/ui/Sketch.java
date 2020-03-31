package deepLearning.ui;

import deepLearning.neuralNetwork.NeuralNetwork;

import deepLearning.neuralNetwork.activationFunctions.Tanh;
import deepLearning.neuralNetwork.data.Dataset;

import deepLearning.neuralNetwork.data.label.XorGenerator;
import processing.core.PApplet;


public class Sketch extends PApplet {
    NeuralNetwork neuralNetwork;
    Dataset dataset;

    @Override
    public void settings() {
        size(500, 500);
    }

    @Override
    public void setup() {
        frameRate(2000);
        neuralNetwork = new NeuralNetwork(new int[]{2, 4, 2, 1}, new Tanh());
        dataset = new Dataset(new XorGenerator(100000));
    }

    @Override
    public void draw() {
        System.out.println(frameRate);
        for (int i = 0; i < 100; i++) {
            neuralNetwork.trainBatch(dataset, 1000, 0.06);
        }

//        System.out.println(Arrays.toString(neuralNetwork.predict(new double[]{1, 1})));
//        System.out.println(Arrays.toString(neuralNetwork.predict(new double[]{-1, 1})));
//        System.out.println(Arrays.toString(neuralNetwork.predict(new double[]{1, -1})));
//        System.out.println(Arrays.toString(neuralNetwork.predict(new double[]{-1, -1})));


        for (int x = 0; x < width; x += 5) {
            for (int y = 0; y < height; y += 5) {
                double[] input = new double[]{map(x, 0, width, -1, 1), map(y, 0, height, -1, 1)};
                double[] outputs = neuralNetwork.predict(input);
                fill(0, 255, (float) (128 + outputs[0] * 128));
                noStroke();
                rect(x, y, 5, 5);
            }
        }
    }

}
