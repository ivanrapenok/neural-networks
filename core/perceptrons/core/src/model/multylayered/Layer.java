package model.multylayered;

import org.ejml.simple.SimpleMatrix;

import java.util.Random;

public class Layer {
    private int inputs;
    private int neurons;

    public SimpleMatrix W;
    public SimpleMatrix S;
    public SimpleMatrix OUT;

    public Layer(int inputs, int neurons) {
        this.inputs = inputs;
        this.neurons = neurons;
        W = SimpleMatrix.random(inputs, neurons, -0.5, 0.5, new Random());
    }

}

