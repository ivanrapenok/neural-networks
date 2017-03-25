package model.multylayered;

import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MultyLayeredPerceptron {

    private static final double nyu = 0.9;
    private static final double epsilon = 0.1;

    private static List<Layer> perceptron = new ArrayList<>();

    public static void main(String[] args) {
        perceptron.add(new Layer(2, 3));
        perceptron.add(new Layer(3, 2));


        SimpleMatrix TT = new SimpleMatrix(new double[][]{{2, 3, 5}});

        SimpleMatrix OO = new SimpleMatrix(new double[][]{{7, 1, 4}});
        SimpleMatrix SS = new SimpleMatrix(new double[][]{{2, 3}});
        SimpleMatrix WW = new SimpleMatrix(new double[][]{{1, 2}, {3, 1}, {5, 4}});

        System.out.println(WW);
        System.out.println(SS);

        System.out.println(calcError(OO, TT));
        System.out.println(calcErrorHidden(SS, WW, OO));

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
            if (isDiscrete) OUT.set(OUT.get(index) >= 0.8 ? 1 : 0);
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

}