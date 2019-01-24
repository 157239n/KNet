package org.kelvinho.neural;

import java.util.Random;

@SuppressWarnings({"WeakerAccess", "unused"})
public class RandomGenerator {
    public static float random(float lowerBound, float upperBound) {
        Random random = new Random();
        return random.nextFloat() * (upperBound - lowerBound) + lowerBound;
    }

    public static float random(float bounds) {
        return random(-Math.abs(bounds), Math.abs(bounds));
    }

    public static float random() {
        return random(1);
    }
}
