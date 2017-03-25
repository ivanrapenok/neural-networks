package model.multylayered;

import org.ejml.simple.SimpleMatrix;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MultyLayeredPerceptron {

    private static final double nyu = 0.1;
    private static final double epsilon = 0.2;

    private static SimpleMatrix INPUTS = new SimpleMatrix(new double[][]{
            {0, 0, 1, 1},
            {0, 1, 0, 1}
    });
    private static SimpleMatrix OUTPUTS = new SimpleMatrix(new double[][]{
            {1},
            {0},
            {0},
            {1}
    });

    private static List<Layer> perceptron = new ArrayList<>();

    public static void main(String[] args) {
        perceptron.add(new Layer(2, 3));
        perceptron.add(new Layer(3, 1));

        trainPerceptron(1000, INPUTS, OUTPUTS);

        System.out.println(perceptron.get(0).W);
        System.out.println(perceptron.get(1).W);
        for (int i = 0; i < 4; i++) {
            System.out.println(processPerceptron(INPUTS.extractVector(false, i)));
        }

        String res = "";
        for (double i = -1; i <= 2; i += 0.4) {
            for (double j = -1; j <= 2; j += 0.4) {
                res += processPerceptron(new SimpleMatrix(new double[][]{{i}, {j}})).get(0) + " ";
            }
            res += "\n";
        }
        printToFile(res, "output-multy.txt");
    }

    private static void trainPerceptron(int maxIterationCount, SimpleMatrix INPUTS, SimpleMatrix OUTPUTS) {
        int samplesWithoutCorrect = 0;
        for (int i = 0; i < maxIterationCount; i++) {
            for (int j = 0; j < INPUTS.numCols(); j++) {
                SimpleMatrix X = new SimpleMatrix(INPUTS.extractVector(false, j));
                SimpleMatrix T = new SimpleMatrix(OUTPUTS.extractVector(true, j));
                SimpleMatrix OUT = processPerceptron(X);
                int countOfCorrect = 0;
                while (!OUT.isIdentical(T, epsilon)) {
                    correctPerceptron(X, T);
                    OUT = processPerceptron(X);
                    countOfCorrect++;
                }
                samplesWithoutCorrect = countOfCorrect == 0 ? samplesWithoutCorrect + 1 : 0;
                if (samplesWithoutCorrect >= INPUTS.numCols()) {
                    System.out.println("Well played!");
                    maxIterationCount = 0;
                    break;
                }
            }
        }
    }

    private static SimpleMatrix processPerceptron(SimpleMatrix X) {
        SimpleMatrix XI = X.copy();
        SimpleMatrix W;
        boolean isDiscrete;
        for (int i = 0; i < perceptron.size(); i++) {
            W = perceptron.get(i).W;
            isDiscrete = false;
            XI = process(XI, W, isDiscrete);
            perceptron.get(i).OUT = XI.copy();
        }
        return XI;
    }

    private static void correctPerceptron(SimpleMatrix INPUT, SimpleMatrix OUTPUT) {
        int lastLayerId = perceptron.size() - 1;
        perceptron.get(lastLayerId).S = calcError(perceptron.get(lastLayerId).OUT, OUTPUT);
        for (int i = perceptron.size() - 2; i >= 0; i--) {
            Layer CUR = perceptron.get(i);
            Layer NEXT = perceptron.get(i + 1);
            perceptron.get(i).S = calcErrorHidden(NEXT.S, NEXT.W, CUR.OUT);
        }
        for (int i = 1; i < perceptron.size(); i++) {
            Layer CUR = perceptron.get(i);
            Layer PREV = perceptron.get(i - 1);
            perceptron.get(i).W = correctWeight(CUR.W, CUR.S, PREV.OUT);
        }
        perceptron.get(0).W = correctWeight(perceptron.get(0).W, perceptron.get(0).S, INPUT.transpose());
    }

    private static SimpleMatrix correctWeight(SimpleMatrix W, SimpleMatrix S, SimpleMatrix X) {
        return W.plus((X.transpose()).mult(S.scale(nyu)));
    }

    private static SimpleMatrix calcError(SimpleMatrix OUT, SimpleMatrix T) {
        return (T.minus(OUT)).elementMult(calcDerivative(OUT));
    }

    private static SimpleMatrix calcErrorHidden(SimpleMatrix SE, SimpleMatrix WE, SimpleMatrix OUT) {
        SimpleMatrix SUM = multDot(SE, WE.transpose());
        return calcDerivative(OUT).elementMult(SUM);
    }

    private static SimpleMatrix calcDerivative(SimpleMatrix OUT) {
        SimpleMatrix identity = SimpleMatrix.random(1, OUT.numCols(), 1, 1, new Random());
        return OUT.elementMult(OUT.scale(-1).plus(identity));
    }

    private static SimpleMatrix process(SimpleMatrix X, SimpleMatrix W, boolean isDiscrete) {
        return process(multDot(X, W), isDiscrete);
    }

    private static SimpleMatrix process(SimpleMatrix NET, boolean isDiscrete) {
        SimpleMatrix OUT = NET.copy();
        int index;
        for (int i = 0; i < OUT.numCols(); i++) {
            index = OUT.getIndex(0, i);
            OUT.set(index, 1 / (1 + Math.exp(-OUT.get(index))));
            if (isDiscrete) OUT.set(OUT.get(index) >= 0.3 ? 1 : 0);
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

    private static void printToFile(String res, String outputFile) {
        try (PrintWriter out = new PrintWriter(outputFile)) {
            out.println(res);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

}