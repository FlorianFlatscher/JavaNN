package deepLearning.neuralNetwork.activationFunctions;

public interface ActivationFunction {
    public double activate(double a);
    public double derivative(double a);
}
