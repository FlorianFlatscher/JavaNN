package deepLearning.neuralNetwork.activationFunctions;

public class LinearActivation implements ActivationFunction {
    @Override
    public double activate(double a) {
        return a;
    }

    @Override
    public double derivative(double a) {
        return 1;
    }


}
