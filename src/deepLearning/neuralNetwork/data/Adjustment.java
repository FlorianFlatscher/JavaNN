package deepLearning.neuralNetwork.data;

import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

public class Adjustment {
    private double[][][] adjustments;
    private double[][] biasAdjustments;

    public Adjustment(double[][][] adjustments, double[][] biasAdjustments) {
        this.adjustments = adjustments;
        this.biasAdjustments = biasAdjustments;
    }

    public double[][][] getAdjustments() {
        return adjustments;
    }

    public double[][] getBiasAdjustments() {
        return biasAdjustments;
    }

    public static Adjustment avarage(List<Future<Adjustment>> adjustments) throws ExecutionException, InterruptedException {
        double[][][] ad = adjustments.get(0).get().getAdjustments().clone();
        double[][] biasAd = adjustments.get(0).get().getBiasAdjustments().clone();

        for (int i = 1; i < adjustments.size(); i++) {
            for (int j = 0; j < ad.length; j++) {
                ad[j] = addMatrix(ad[j].clone(), adjustments.get(i).get().getAdjustments()[j].clone());
            }

            biasAd = addMatrix(biasAd.clone(), adjustments.get(i).get().getBiasAdjustments().clone());
        }

        for (int i = 0; i < biasAd.length; i++) {
            for (int j = 0; j < biasAd[i].length; j++) {
                biasAd[i][j] /= adjustments.size();
            }
        }

        for (int i = 0; i < ad.length; i++) {
            for (int j = 0; j < ad[i].length; j++) {
                for (int k = 0; k < ad[i][j].length; k++) {
                    ad[i][j][k] /= adjustments.size();
                }
            }
        }


        return new Adjustment(ad, biasAd);
    }

    private static double[][] addMatrix(double[][] array1, double[][] array2) {
        double[][] sum = new double[array1.length][];
        for (int i = 0; i < array1.length; i++) {
            sum[i] = new double[array1[i].length];
            for (int j = 0; j < sum[i].length; j++) {
                sum[i][j] = array1[i][j] + array2[i][j];
            }
        }
        return sum;
    }
}
