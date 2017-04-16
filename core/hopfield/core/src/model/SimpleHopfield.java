package model;

import org.ejml.simple.SimpleMatrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.sqrt;
import static java.util.Arrays.asList;

public class SimpleHopfield {

    private static SimpleMatrix X = new SimpleMatrix(new double[][]{{
            1, 1, -1, 1, -1,
            1, 1, -1, 1, -1,
            1, -1, 1, 1, 1,
            1, 1, 1, -1, -1,
            -1, -1, 1, -1, 1
    }});
    //  private static SimpleMatrix X = new SimpleMatrix(new double[][]{{
    //          1, 1, 1, -1, 1,
    //          1, 1, -1, -1, 1,
    //          1, 1, 1, -1, 1,
    //          1, 1, 1, -1, 1,
    //          1, 1, 1, -1, 1
    //  }});

    private static final int N = 100;

    private static SimpleMatrix X1;
    private static SimpleMatrix X2;
    private static SimpleMatrix X3;

    private static List<SimpleMatrix> alphabet;
    private static List<SimpleMatrix> inputs;

    private static void initAlphabet() {
        X1 = new SimpleMatrix(new double[][]{{
                1, 1, 1, -1, 1,
                1, 1, -1, -1, 1,
                1, 1, 1, -1, 1,
                1, 1, 1, -1, 1,
                1, 1, 1, -1, 1
        }});
        X2 = new SimpleMatrix(new double[][]{{
                1, -1, -1, 1, 1,
                -1, 1, 1, -1, 1,
                -1, 1, 1, -1, 1,
                -1, 1, 1, -1, 1,
                1, -1, -1, 1, 1
        }});
        X3 = new SimpleMatrix(new double[][]{{
                1, 1, -1, -1, 1,
                1, -1, 1, 1, -1,
                1, 1, 1, -1, 1,
                1, 1, -1, 1, 1,
                1, -1, -1, -1, -1
        }});
        alphabet = new ArrayList<>(asList(X1, X2, X3));
    }

    public static void main(String[] args) {
        //initAlphabet();
        alphabet = initAlphabetFromFile("core/hopfield/core/files/alphabet.txt", '0');
        inputs = initAlphabetFromFile("core/hopfield/core/files/inputs.txt", '0');
        if (alphabet == null || inputs == null) {
            System.out.println("error while process file");
            return;
        }
        //int loopCount = 10;
        int loopCount = inputs.size();

        SimpleMatrix W = new SimpleMatrix(new double[N][N]);
        for (SimpleMatrix X : alphabet) {
            W = W.plus(X.transpose().mult(X));
        }
        W = W.divide(N);
        for (int i = 0; i < N; i++) {
            W.set(i, i, 0);
        }

        //----------------------
        //printW(W);
        List<SimpleMatrix> iterations;
        int maxIt = -1;
        //----------------------

        SimpleMatrix RES = new SimpleMatrix(new double[1][N]);
        for (int j = 0; j < loopCount; j++) {
            //------------------
            //X = fun(SimpleMatrix.random(1, N, -1, 1, new Random()));
            X = inputs.get(j);
            iterations = new ArrayList<>(asList(X.copy()));
            //Energy
            //System.out.println(-X.elementMult(X).elementSum() -(1/2) * fun((X.transpose()).mult(X)).elementMult(W).elementSum());
            System.out.println(-fun((X.transpose()).mult(X)).elementMult(W).elementSum());
            //------------------

            RES = fun(W.mult(X.transpose()));
            SimpleMatrix prevRES;
            for (int i = 0; i < 100; i++) {
                prevRES = RES.copy();
                //---------------
                //Energy
                //System.out.println(-RES.elementMult(X.transpose()).elementSum() -(1/2) * fun(RES.mult(RES.transpose())).elementMult(W).elementSum());
                System.out.println(-fun(RES.mult(RES.transpose())).elementMult(W).elementSum());
                //printGather(asList(prevRES, RES));
                iterations.add(RES.copy());
                //---------------
                RES = fun(W.mult(RES));

                if (prevRES.isIdentical(RES, 0) || (prevRES.isIdentical(RES.scale(-1), 0))) {
                    //-----------------------
                    if (true || i > maxIt) {
                        maxIt = i;
                        System.out.println(maxIt);
                        printGather(iterations);
                    }
                    //------------------------
                    break;
                }
            }
        }

        System.out.println();
        printGather(alphabet);
    }

    private static SimpleMatrix fun(SimpleMatrix M) {
        SimpleMatrix RES = M.copy();
        for (int i = 0; i < RES.getNumElements(); i++)
            RES.set(i, RES.get(i) == 0 ? 0 : RES.get(i) > 0 ? 1 : -1);
        return RES;
    }

    private static void print(SimpleMatrix M) {
        for (int i = 0; i < M.getNumElements(); i++) {
            System.out.print(M.get(i) != 0 ? (M.get(i) > 0 ? "\u2591" : "\u2588") : 0);
            if ((i + 1) % (int) sqrt(M.getNumElements()) == 0) System.out.print("\n");
        }
    }

    private static void printW(SimpleMatrix M) {
        for (int i = 0; i < M.getNumElements(); i++) {
            System.out.print((M.get(i) >= 0) ? "  " : " ");
            System.out.printf("%.2f",  M.get(i));
            if ((i + 1) % (int) sqrt(M.getNumElements()) == 0) System.out.print("\n");
        }
    }

    private static void printGather(List<SimpleMatrix> L) {
        int matrixNum;
        int position;
        int w = (int) sqrt(N);
        int l = L.size();
        for (int i = 0; i < N * l; i++) {
            matrixNum = (i % (w * l)) / w;
            position = (i / (w * l)) * w + i % w;
            System.out.print(L.get(matrixNum).get(position) > 0 ? "\u2591" : "\u2588");
            if ((i + 1) % w == 0) System.out.print("  ");
            if ((i + 1) % (w * l) == 0) System.out.print("\n");
        }
    }

    private static List<SimpleMatrix> initAlphabetFromFile(String file, char mnsOne) {
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            List<SimpleMatrix> res = new ArrayList<>();
            int it = 0;
            SimpleMatrix M = new SimpleMatrix(new double[1][N]);
            String sCurrentLine;

            while ((sCurrentLine = br.readLine()) != null) {
                for (char c: sCurrentLine.toCharArray())
                    M.set(it++, c == mnsOne ? 1 : -1);
                if (it >= N) {
                    res.add(M.copy());
                    it = 0;
                }
            }
            return res;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}
