package org.kelvinho.neural;

import org.kelvinho.matrix.Matrix;

import javax.annotation.Nonnull;

@SuppressWarnings({"unused", "WeakerAccess"})
public class Error {
    public static float quadratic(@Nonnull Matrix answer, @Nonnull Matrix correctAnswer) {
        return answer.minus(correctAnswer).sq().mul(0.5).sum();
    }

    public static float absolute(@Nonnull Matrix answer, @Nonnull Matrix correctAnswer) {
        return answer.minus(correctAnswer).abs().sum();
    }

    public static float crossEntropy(@Nonnull Matrix answer, @Nonnull Matrix correctAnswer) {
        throw new AssertionError("TODO");// TODO
    }
}
