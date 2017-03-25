package model;

import org.ejml.simple.SimpleMatrix;

import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class SingleLayerPerceptron {

    private static final double nyu = 1;
    private static final double epsilon = 0.1;

    private static double[][] Xarr = {
            {1},
            {1}
    };
    private static double[][] Warr = {
            {0, 0},
            {0, 0}
    };
    private static double[][] trainInput = {
            {0, 0, 1, 1},
            {0, 1, 0, 1}
    };
    private static double[][] trainOutput = {
            {0, 0},
            {0, 1},
            {0, 1},
            {1, 1}
    };

    public static void main(String[] args) {
        SimpleMatrix W = new SimpleMatrix(Warr);
        SimpleMatrix INPUTS = new SimpleMatrix(trainInput);
        SimpleMatrix OUTPUTS = new SimpleMatrix(trainOutput);

        System.out.println(INPUTS);
        System.out.println(OUTPUTS);
        System.out.println(W);

        SimpleMatrix ResW = train(1000, INPUTS, OUTPUTS, W);
        System.out.println(ResW);

        for (byte b = 0; b < 4; b++) {
            System.out.println("---- " + ((b >> 1) & 1) + "  " + (b & 1) + " ----");
            System.out.println(process(new SimpleMatrix(new double[][]{{(b >> 1) & 1}, {b & 1}}), ResW));
        }

        String res = "";
        for (double i = -1; i <= 2; i += 0.4) {
            for (double j = -1; j <= 2; j += 0.4) {
                res += process(new SimpleMatrix(new double[][]{{i}, {j}}), ResW).get(0) + " ";
//                res += String.format("%.2f ", i) +
//                        String.format("%.2f ", j) +
//                        process(new SimpleMatrix(new double[][]{{i}, {j}}), ResW).get(0) + "\n";
            }
            res += "\n";
        }
        printToFile(res);
    }

    private static SimpleMatrix train(int maxIterationCount, SimpleMatrix INPUTS, SimpleMatrix OUTPUTS, SimpleMatrix W) {
        int samplesWithoutCorrect = 0;
        SimpleMatrix WRes = W;
        for (int i = 0; i < maxIterationCount; i++) {
            for (int j = 0; j < INPUTS.numCols(); j++) {
                SimpleMatrix X = new SimpleMatrix(INPUTS.extractVector(false, j));
                SimpleMatrix T = new SimpleMatrix(OUTPUTS.extractVector(true, j));
                SimpleMatrix OUT = process(X, WRes);
                int countOfCorrect = 0;
                while (!OUT.isIdentical(T, epsilon)) {
                    WRes = correctWeight(WRes, T, OUT, X, nyu);
                    OUT = process(X, WRes);
                    countOfCorrect++;
                    System.out.println(WRes);
                }
                samplesWithoutCorrect = countOfCorrect == 0 ? samplesWithoutCorrect + 1 : 0;
                if (samplesWithoutCorrect >= INPUTS.numCols()) maxIterationCount = 0;
            }
        }
        return WRes;
    }

    private static SimpleMatrix correctWeight(SimpleMatrix W, SimpleMatrix T, SimpleMatrix OUT, SimpleMatrix X, double n) {
        SimpleMatrix WNew = W.plus(X.mult(T.minus(OUT).scale(n)));
        return WNew;
    }

    private static SimpleMatrix process(SimpleMatrix X, SimpleMatrix W) {
        return process(multDot(X, W));
    }

    private static SimpleMatrix process(SimpleMatrix NET) {
        SimpleMatrix OUT = NET.copy();
        int index;
        for (int i = 0; i < OUT.numCols(); i++) {
            index = OUT.getIndex(0, i);
            OUT.set(index, (1 / (1 + Math.exp(-OUT.get(index)))) >= 0.8 ? 1 : 0);
        }
        return OUT;
    }

    private static SimpleMatrix multDot(SimpleMatrix X, SimpleMatrix W) {
        double[][] NET = new double[1][W.numCols()];
        for (int i = 0; i < W.numCols(); i++) {
            NET[0][i] = X.dot(W.extractVector(false, i));
        }
        return new SimpleMatrix(NET);
    }

    private static void printToFile(String res) {
        try (PrintWriter out = new PrintWriter("output.txt")) {
            out.println(res);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

}
