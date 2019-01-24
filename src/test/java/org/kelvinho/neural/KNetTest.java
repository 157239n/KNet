package org.kelvinho.neural;

import org.junit.Test;
import org.kelvinho.matrix.Matrix;

public class KNetTest {
    @Test
    public void XORProblem() {
        KNet.Parameters parameters = new KNet.Parameters(3, 4, 1);
        parameters.setLearningConstant(1);
        KNet net = new KNet(parameters);

        Matrix X = new Matrix(new float[][]{
                new float[]{0, 0, 1},
                new float[]{0, 1, 1},
                new float[]{1, 0, 1},
                new float[]{1, 1, 1},
        });
        Matrix Y = new Matrix(new float[][]{new float[]{0, 1, 1, 0}}).transpose();

        int times = 0;
        while(Error.quadratic(net.feed(X), Y) > 0.01) {
            net.backProp(X, Y);
            times ++;
        }
        System.out.println(times);
    }

    @Test
    public void sameValueProblem() {
        KNet.Parameters parameters = new KNet.Parameters(2, 3, 1);
        parameters.setLearningConstant(0.1);
        KNet net = new KNet(parameters);

        Matrix X = new Matrix(new float[][]{
                new float[]{0, 1},
                new float[]{1, 1}
        });
        Matrix Y = new Matrix(new float[][]{new float[]{0, 1}}).transpose();

        for (int i = 0; i < 20000; i++) {
            net.backProp(X, Y);
            if (i % 3000 == 0) {
                Matrix resultMatrix = net.feed(X);
                resultMatrix.print();
                System.out.println(Error.quadratic(resultMatrix, Y));
                System.out.println();
            }
        }
        Y.print();
    }
}