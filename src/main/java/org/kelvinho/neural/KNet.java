package org.kelvinho.neural;

import org.kelvinho.matrix.Matrix;

import javax.annotation.Nonnull;
import java.util.function.Supplier;

@SuppressWarnings({"unused", "WeakerAccess"})
public class KNet {
    private Matrix[] synapses;
    private Parameters parameters;
    private Matrix[] lastDeltaW = null; // we don't use this as a local variable because we might want to apply some momentum term

    public static class Parameters {
        private int[] dimensions;
        private Supplier<Float> randomGenerator = RandomGenerator::random;
        private int numberOfSynapses;
        private float learningConstant = 0.5f;
        private boolean useMomentum = false;
        private float momentum = 0.5f;

        public Parameters(@Nonnull int... dimensions) {
            if (dimensions.length < 2) {
                throw new IllegalArgumentException("Dimension length must be greater than or equal to 2, because that constitutes a matrix");
            }
            this.dimensions = dimensions;
            numberOfSynapses = dimensions.length - 1;
        }

        public void setRandomGenerator(@Nonnull Supplier<Float> randomGenerator) {
            this.randomGenerator = randomGenerator;
        }

        public void setLearningConstant(double learningConstant) {
            this.learningConstant = (float) learningConstant;
        }

        public void enableMomentum() {
            useMomentum = true;
        }

        public void setMomentum(double momentum) {
            this.momentum = (float) momentum;
        }
    }

    public KNet(@Nonnull Parameters parameters) {
        this.parameters = parameters;
        initializeSynapses();
    }

    private void initializeSynapses() {
        synapses = new Matrix[parameters.numberOfSynapses];
        for (int i = 0; i < synapses.length; i++) {
            synapses[i] = new Matrix(parameters.dimensions[i], parameters.dimensions[i + 1], (x, y) -> parameters.randomGenerator.get());
        }
    }

    @SuppressWarnings("ForLoopReplaceableByForEach")
    public Matrix feed(@Nonnull Matrix X) {
        Matrix answer = X;
        for (int i = 0; i < synapses.length; i++) {
            answer = answer.dot(synapses[i]).sigmoid();
        }
        return answer;
    }

    public void backProp(@Nonnull Matrix X, @Nonnull Matrix Y) {
        // feed forward and get the result matrices
        Matrix[] resultMatrices = new Matrix[synapses.length + 1]; // X, after syn 1, after syn 2, ...
        resultMatrices[0] = X;
        for (int i = 0; i < synapses.length; i++) {
            resultMatrices[i + 1] = resultMatrices[i].dot(synapses[i]).sigmoid();
        }

        // calculate delta matrices
        Matrix[] deltaMatrices = new Matrix[synapses.length];
        deltaMatrices[synapses.length - 1] = Y.minus(resultMatrices[synapses.length]).mul(resultMatrices[synapses.length].sigmoidDerivative());
        for (int i = synapses.length - 2; i >= 0; i--) {
            deltaMatrices[i] = deltaMatrices[i + 1].dot(synapses[i + 1].T()).mul(resultMatrices[i + 1].sigmoidDerivative());
        }

        // calculate delta of synapses
        Matrix[] deltaW = new Matrix[synapses.length];
        for (int i = 0; i < synapses.length; i++) {
            deltaW[i] = resultMatrices[i].T().dot(deltaMatrices[i]).mul(parameters.learningConstant);
        }

        // learn
        if (parameters.useMomentum && lastDeltaW != null) {
            for (int i = 0; i < synapses.length; i++) {
                synapses[i] = synapses[i].add(deltaW[i]).add(lastDeltaW[i].mul(parameters.momentum));
            }
        } else {
            for (int i = 0; i < synapses.length; i++) {
                synapses[i] = synapses[i].add(deltaW[i]);
            }
        }
        lastDeltaW = deltaW;
    }
}
