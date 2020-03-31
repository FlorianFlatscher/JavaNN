package deepLearning.neuralNetwork.activationFunctions;

public class Tanh implements ActivationFunction {

    @Override
    public double activate(double a) {
        return Math.tanh(a);
    }

    @Override
    public double derivative(double a) {
        return 1 - a * a;
    }
}
