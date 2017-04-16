package model;

import org.ejml.simple.SimpleMatrix;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import static java.lang.Math.sqrt;
import static java.lang.Math.tanh;

public class CommercialTraveler {

    private static final int N = 5; // number of cities
    private static final double[][] MAP = new double[][]{ // coordinates of the cities
            {0.025, 0.125},
            {0.150, 0.750},
            {0.125, 0.225},
            {0.325, 0.550},
            {0.500, 0.150}
    };

    // constants
    private static final double betta = 0.03;
    private static final double sigma = 1;
    //private static final double A = (new Random()).nextDouble() * betta;   // one unit per line
    //private static final double B = (new Random()).nextDouble();   // one unit in the column
    //private static final double C = 1 - (new Random()).nextDouble() * betta;;  // total n
    //private static final double D = (new Random()).nextDouble() * betta + 1d/N;   // min path coast
    private static final double A = 100;   // one unit per line
    private static final double B = 100;   // one unit in the column
    private static final double C = 90;  // total n
    private static final double D = 70;   // min path coast

    private static final double u0 = 50; // function param

    private static SimpleMatrix W = new SimpleMatrix(new double[N * N][N * N]);  // relations weight

    public static void main(String[] args) {
        System.out.println("u0 is " + u0);
        System.out.println("A is " + A);
        System.out.println("B is " + B);
        System.out.println("C is " + C);
        System.out.println("D is " + D);

        setWeight();
        //printW(W);

        int LOOP = 20;
        Map<Integer, Double> result = new HashMap<>();
        double minWay = 0;
        for (int l = 0; l < LOOP; l++) {
            SimpleMatrix X = SimpleMatrix.random(N * N, 1, 0, 0.01, new Random());
            //printW(X);

            SimpleMatrix RES;
            SimpleMatrix PREV;

            if (false) {
                RES = fun(W.mult(X));
                for (int i = 0; i < 100; i++) {
                    RES = fun(W.mult(RES));
                    printW(RES);
                }
            } else {
                Random rand = new Random();
                int randEl = rand.nextInt(N * N);
                RES = multOneFun(X, randEl);
                for (int i = 0; i < N * N * 15; i++) {
                    PREV = RES.copy();
                    randEl = rand.nextInt(N * N);
                    //randEl = i % 25;
                    RES = multOneFun(RES, randEl);
                    System.out.printf("SUM: %.2f,     1: %.2f,     2: %.2f \n", energyDist(RES) + energyRule(RES), energyRule(RES), energyDist(RES));
                    //if (PREV.isIdentical(RES, 0.0001)) break;
                    //log
                    //System.out.println(randEl);
                    //printW(RES);
                    //System.out.printf("e = %.4f \n", -0.5 * W.elementMult(RES.mult(RES.transpose())).elementSum());
                }
            }
            //RES = new SimpleMatrix(new double[][]{
            //               {0, 0, 0, 0, 1,
            //                0, 0, 1, 0, 0,
            //                0, 0, 0, 1, 0,
            //                1, 0, 0, 0, 0,
            //                0, 1, 0, 0, 0}
            //});
            //System.out.printf("SUM: %.2f,     1: %.2f,     2: %.2f \n", energyDist(RES) + energyRule(RES), energyRule(RES), energyDist(RES));
            //System.out.printf("e = %.4f \n", -0.5 * W.elementMult((RES.transpose()).mult(RES)).elementSum());
            System.out.println("RES");
            printW(RES);
            System.out.println();
            minWay = getExpectedPath();
            System.out.println();
            result.put(l, getResultPath(RES) / minWay - 1);
        }
        System.out.println();
        result.forEach((k, v) -> System.out.printf("%d    %.2f \n", k, v));
        System.out.printf("%.2f", result.values().stream().mapToDouble(Double::doubleValue).sum() / result.size());
    }

    private static void setWeight() {
        double value;
        for (int x = 0; x < N; x++) {
            for (int i = 0; i < N; i++) {
                for (int y = 0; y < N; y++) {
                    for (int j = 0; j < N; j++) {
                        value = -A * sig(x, y) * (1 - sig(i, j))
                                - B * sig(i, j) * (1 - sig(x, y))
                                - C * (1 - sig(x, y) * sig(i, j))
                                - D * distance(x, y) * (sig(j, i + 1) + sig(j, i - 1))
                        ;
                        W.set(x * N + i, y * N + j, value);
                    }
                }
            }
        }
    }

    private static SimpleMatrix multOneFun(SimpleMatrix IN, int element) {
        SimpleMatrix res = IN.copy();
        //SimpleMatrix one = res.extractVector(true, element).mult(W.extractVector(true, element));
        //res.set(element, fun(one.get(0)));

        int x = element / N;
        int i = element - x * N;

        double u = 0;

        for (int j = 0; j < N; j++)
            u += (i != j) ? -A * IN.get(x * N + j) : 0;

        for (int y = 0; y < N; y++)
            u += (x != y) ? -B * IN.get(y * N + i) : 0;

        for (int y = 0; y < N; y++)
            for (int j = 0; j < N; j++)
                u += -C * IN.get(y * N + j);

        u += C * (N + sigma);

        for (int y = 0; y < N; y++)
            //u += (y != x) ? -D * (IN.get(y * N + ((i == N - 1) ? -1 : i) + 1) + IN.get(y * N + ((i == 0) ? N : i) - 1)) * distance(x, y) : 0;
            u += (y != x) ? -D * (IN.get(y * N + ((i == N - 1) ? -1 : i) + 1) + IN.get(y * N + ((i == 0) ? N : i) - 1)) * distance(x, y) : 0;

        res.set(element, fun(u));
        return res;
    }

    private static double energyDist(SimpleMatrix IN) {
        double e = 0;
        for (int x = 0; x < N; x++)
            for (int y = 0; y < N; y++)
                for (int i = 0; i < N; i++)
                    e += (y == x) ? 0 : 0.5 * D * IN.get(x * N + i) * (IN.get(y * N + ((i == N - 1) ? -1 : i) + 1) + IN.get(y * N + ((i == 0) ? N : i) - 1)) * distance(x, y);
        return e;
    }

    private static double energyRule(SimpleMatrix IN) {
        double e = 0;
        double tmp = 0;
        for (int x = 0; x < N; x++)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++)
                    e += (i == j) ? 0 : 0.5 * A * IN.get(x * N + i) * IN.get(x * N + j);
                for (int y = 0; y < N; y++)
                    e += (x == y) ? 0 : 0.5 * B * IN.get(x * N + i) * IN.get(y * N + i);
                tmp += IN.get(x * N + i);
            }

        e += 0.5 * C * (tmp - (N + sigma)) * (tmp - (N + sigma));
        return e;
    }

    private static SimpleMatrix fun(SimpleMatrix IN) {
        SimpleMatrix res = IN.copy();
        for (int i = 0; i < IN.getNumElements(); i++) {
            System.out.print(res.get(i) + " ");
            if ((i + 1) % N == 0) System.out.println();
            res.set(i, fun(res.get(i)));
        }
        return res;
    }

    private static double fun(double arg) {
        return 0.5 * (1 + tanh((arg) * u0));
    }

    private static int sig(int i, int j) {
        if (j == N) j = 0;
        if (j == -1) j = N - 1;
        return i == j ? 1 : 0;
    }

    private static double distance(int i, int j) {
        return sqrt((MAP[i][0] - MAP[j][0]) * (MAP[i][0] - MAP[j][0]) +
                (MAP[i][1] - MAP[j][1]) * (MAP[i][1] - MAP[j][1]));
    }

    private static void printW(SimpleMatrix M) {
        for (int i = 0; i < M.getNumElements(); i++) {
            System.out.print((M.get(i) > 0) ? "  " : " ");
            System.out.print((M.get(i) > -10 && M.get(i) < 10) ? " " : "");
            System.out.print((M.get(i) > -100 && M.get(i) < 100) ? " " : "");
            System.out.print((M.get(i) > -1000 && M.get(i) < 1000) ? " " : "");
            System.out.print((M.get(i) > -10000 && M.get(i) < 10000) ? " " : "");

            System.out.printf("%.2f", M.get(i));
            //System.out.print(M.get(i) > -5 ? "\u2591" : "\u2588");
            if ((i + 1) % (int) sqrt(M.getNumElements()) == 0) System.out.print("\n");
        }
    }

    private static double getExpectedPath() {
        double sum = 0;
        int count = 0;

        double min = Double.MAX_VALUE;
        double way = 0;
        String path = "";
        for (int a = 0; a < N; a++)
            for (int b = 0; b < N; b++)
                if (a != b)
                    for (int c = 0; c < N; c++)
                        if (c != a && c != b)
                            for (int d = 0; d < N; d++)
                                if (d != a && d != b && d != c)
                                    for (int e = 0; e < N; e++)
                                        if (e != a && e != b && e != c && e != d) {
                                            way = distance(a, b) + distance(b, c) + distance(c, d) + distance(d, e);
                                            min = Math.min(min, way);
                                            if (min == way) path = a + " " + b + " " + c + " " + d + " " + e;

                                            count++;
                                            sum += way;
                                        }

        System.out.printf("target-> %s  min distance is %.3f    avg: %.3f", path, min, sum / count);
        return min;
    }

    private static double getResultPath(SimpleMatrix IN) {
        double way = 0;
        String path = "";
        int prev = -1;

        for (int i = 0; i < N; i++)
            for (int x = 0; x < N; x++)
                if (IN.get(x * N + i) > 0.9) {
                    path += x + " ";
                    if (prev >= 0) way += distance(prev, x);
                    prev = x;
                }
        System.out.printf("result-> %s min distance is %.3f", path, way);
        return way;
    }
}
