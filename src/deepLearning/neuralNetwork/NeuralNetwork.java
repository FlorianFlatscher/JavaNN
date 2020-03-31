package deepLearning.neuralNetwork;

import deepLearning.neuralNetwork.activationFunctions.ActivationFunction;
import deepLearning.neuralNetwork.data.Adjustment;
import deepLearning.neuralNetwork.data.Dataset;
import deepLearning.neuralNetwork.data.label.Label;
import deepLearning.neuralNetwork.layers.Layer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class NeuralNetwork {
    private Layer[] layers;

    public NeuralNetwork(int[] layers, ActivationFunction activationFunction) {
        this(layers, activationFunction, activationFunction);
    }

    public NeuralNetwork(int[] layers, ActivationFunction activationFunction, ActivationFunction activationFunctionLastLayer) {
        this.layers = new Layer[layers.length - 1];
        for (int i = 0; i < this.layers.length - 1; i++) {
            this.layers[i] = new Layer(layers[i], layers[i + 1], activationFunction);
        }
        this.layers[this.layers.length - 1] = new Layer(layers[this.layers.length - 1], layers[this.layers.length], activationFunctionLastLayer);
    }

    public double[] predict(double[] inputs) {
        double[] i = inputs.clone();
        for (Layer l : layers) {
            i = l.feed(i);
        }
        return i;
    }

    public void trainSGD(double[] inputs, double[] targets, double learningRate) {
        applyAdjustment(calculateAdjustment(inputs, targets, learningRate));
    }

    public void trainBatch(Dataset d, double learningRate) {
        List<Future<Adjustment>> adjustments = new ArrayList<>();

        ExecutorService executor = Executors.newFixedThreadPool(4);

        for (int i = 0; i < d.size(); i++) {
            Label label = d.getTrainingLabel(i);
            adjustments.add(executor.submit(() -> calculateAdjustment(label.getInput(), label.getTarget(), learningRate)));
        }

        Adjustment finalAdjustment = null;
        try {
            finalAdjustment = Adjustment.avarage(adjustments);
        } catch (ExecutionException | InterruptedException e) {
            e.printStackTrace();
        }
        applyAdjustment(finalAdjustment);
        executor.shutdown();
    }

    public void trainBatch(Dataset d, int batchSize, double learningRate) {
        List<Future<Adjustment>> adjustments = new ArrayList<>();
        Random rand = new Random();

        ExecutorService executor = Executors.newFixedThreadPool(4);



        for (int i = 0; i < batchSize; i++) {
            Label label = d.getTrainingLabel(rand.nextInt(d.size()));
            adjustments.add(executor.submit(() -> calculateAdjustment(label.getInput(), label.getTarget(), learningRate)));
        }

        Adjustment finalAdjustment = null;
        try {
            finalAdjustment = Adjustment.avarage(adjustments);
        } catch (ExecutionException | InterruptedException e) {
            e.printStackTrace();
        }
        applyAdjustment(finalAdjustment);
        executor.shutdown();
    }

//    public void trainBatch(double[][] inputs, double[][] targets, double learningRate) {
//        Adjustment[] adjustments = new Adjustment[inputs.length];
//        for (int i = 0; i < inputs.length; i++) {
//            adjustments[i] = calculateAdjustment(inputs[i].clone(), targets[i].clone(), learningRate);
//        }
//
//        Adjustment finalAdjustment = Adjustment.avarage(adjustments);
//        applyAdjustment(finalAdjustment);
//    }

//    public void trainMiniBatch(Dataset d, int batchSize, double learningRate) {
//        for (int i = 0; i < d.size(); i += batchSize) {
//            ArrayList<TrainingLabel> labels = d.getLabels();
//            trainBatch(new Dataset((ArrayList<TrainingLabel>) labels.subList(i, i + batchSize)), learningRate);
//        }
//    }
//
//    public void trainMiniBatch(double[][] inputs, double[][] targets, int batchSize, double learningRate) {
//        for (int i = 0; i < inputs.length; i += batchSize) {
//            trainBatch(Arrays.copyOfRange(inputs, i, i + batchSize), Arrays.copyOfRange(targets, i, i + batchSize), learningRate);
//        }
//    }



    private double[][] matrixMultiplication(double[][] array1, double[][] array2) {
        double[][] array = new double[array1.length][array2[0].length];

        for (int row = 0; row < array.length; row++) {
            for (int colomn = 0; colomn < array[0].length; colomn++) {
                double sum = 0;
                for (int count = 0; count < array1[0].length; count++) {
                    sum += array1[row][count] * array2[count][colomn];
                }
                array[row][colomn] = sum;
            }
        }

        return array;
    }

    private double[][] transpose(double[][] a) {
        double[][] b = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int l = 0; l < a[0].length; l++) {
                b[l][i] = a[i][l];
            }
        }
        return b;
    }

    private void applyAdjustment(Adjustment adjustment) {
        for (int layer = 0; layer < layers.length; layer++) {
            layers[layer].adjustWeights(adjustment.getAdjustments()[layer]);
            layers[layer].adjustBiases(adjustment.getBiasAdjustments()[layer]);
        }
    }

    private Adjustment calculateAdjustment(double[] input, double[] target, double learningRate) {
        double[][] outputs = new double[this.layers.length + 1][];

        outputs[0] = input.clone();
        for (int l = 1; l < outputs.length; l++) {
            outputs[l] = layers[l - 1].feed(outputs[l - 1]).clone();
        }

        double[][][] deltas = new double[this.layers.length][][];
        double[][] biasDeltas = new double[this.layers.length][];

        double[][] gradients = calculateGradients(outputs.clone(), target.clone());

        for (int layer = gradients.length-1; layer >= 0; layer--) {

            for (int i = 0; i < gradients[layer].length; i++) {
                gradients[layer][i] *= learningRate;
            }

            deltas[layer] = matrixMultiplication(transpose(new double[][]{outputs[layer].clone()}), new double[][]{gradients[layer].clone()});
            biasDeltas[layer] = gradients[layer].clone();
        }

        return new Adjustment(deltas.clone(), biasDeltas.clone());
    }

    private double[][] calculateGradients(double[][] outputs, double[] target) {
        double[][] errors = new double[this.layers.length][];
        double[][] gradients = new double[this.layers.length][];

        errors[errors.length - 1] = target.clone();
        for (int i = 0; i < target.length; i++) {
            errors[errors.length - 1][i] -= outputs[outputs.length - 1][i];
            if (errors[errors.length - 1][i] > 0) {
                errors[errors.length - 1][i] *= errors[errors.length - 1][i];
            } else {
                errors[errors.length - 1][i] *= -errors[errors.length - 1][i];
            }
//            errors[errors.length-1][i] -= outputs[outputs.length-1][i];
         }

        gradients[gradients.length-1] = new double[target.length];

        for (int i = 0; i < target.length; i++) {
            gradients[gradients.length-1][i] = layers[layers.length - 1].getActivationFunction().derivative(outputs[outputs.length - 1][i]);
            gradients[gradients.length-1][i] *= errors[gradients.length-1][i];
        }

        for (int layer = this.layers.length - 2; layer >= 0; layer--) {
            errors[layer] = layers[layer+1].calcErrors(errors[layer+1]);
            gradients[layer] = new double[errors[layer].length];
            for (int i = 0; i < gradients[layer].length; i++) {
                gradients[layer][i] = layers[layer].getActivationFunction().derivative(outputs[layer+1][i]);
                gradients[layer][i] *= errors[layer][i];
            }
        }

        return gradients;
    }
}

