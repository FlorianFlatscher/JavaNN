package deepLearning.neuralNetwork.layers;

import deepLearning.neuralNetwork.activationFunctions.ActivationFunction;
import processing.core.PApplet;

import java.util.Random;

public class Layer {
    private double[][] weights;
    private double[] bias;

    private ActivationFunction activationFunction;

    public Layer(int in, int out, ActivationFunction activationFunction) {
        weights = new double[out][in];
        Random rand = new Random();
        for (double[] o : weights) {
            for (int i = 0; i < o.length; i++) {
                o[i] = rand.nextDouble() * 2 - 1;
            }
        }
        bias = new double[out];
        for (int i = 0; i < bias.length; i++) {
            bias[i] = rand.nextDouble() * 2 - 1;
        }

        this.activationFunction = activationFunction;
    }

    public double[] feed(double[] inputs) {
        if (inputs.length != weights[0].length) {
            throw new Error("Invalid size of inputs!");
        }

        double[] outputs = new double[weights.length];

        for (int out = 0; out < weights.length; out++) {
            for (int in = 0; in < weights[out].length; in++) {
                outputs[out] += inputs[in] * weights[out][in];
            }
        }

        for (int i = 0; i < outputs.length; i++) {
            outputs[i] += bias[i];
        }

        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = activationFunction.activate(outputs[i]);
        }

        return outputs;
    }

    public double[] calcErrors(double[] errorOfNextLayer) {
        double[] newErrors = new double[this.weights[0].length];
        for (int l = 0; l < this.weights[0].length; l++) {
            for (int i = 0; i < this.weights.length; i++) {
                newErrors[l] += errorOfNextLayer[i] * this.weights[i][l];
            }
        }
        return newErrors;
    }

    public void adjustWeights(double[][] deltas) {
        for (int i = 0; i < deltas.length; i++) {
            for (int l = 0; l < deltas[i].length; l++) {
                double a = weights[l][i] + deltas[i][l];
                if (!Double.isInfinite(a)) {
                    weights[l][i] = a;
                }
            }
        }
    }

    public void adjustBiases(double[] deltas) {
        if (deltas.length != bias.length)
            throw new IllegalArgumentException("Deltas must be the same size as connections");
        for (int i = 0; i < bias.length; i++) {
            bias[i] += deltas[i];
        }
    }

    public void show(PApplet s, int xStart, int yStart, int w, int h) {
        for (int i = 0; i < weights[0].length; i++) {
            s.strokeWeight(1);
            s.stroke(0);
            s.fill(255);
            s.ellipse(xStart, PApplet.map(i +0.5f, 0, weights[0].length, yStart, yStart + h), 20, 20);
        }
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                s.strokeWeight((float) Math.abs(weights[i][j]) * 5);
                s.beginShape();
                s.vertex(xStart + w, PApplet.map(i + 0.5f, 0, weights.length, yStart, yStart + h));
                s.vertex(xStart, PApplet.map(j + 0.5f, 0, weights[i].length, yStart, yStart + h));
                s.endShape();
            }
        }
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }
}
