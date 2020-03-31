package deepLearning.neuralNetwork.activationFunctions;

public class Sigmoid implements ActivationFunction {
    @Override
    public double activate(double a) {
        return 1/(1+Math.pow(Math.E, -a));
    }

    @Override
    public double derivative(double a) {
        return a * (1 - a);
    }


}
