package com.example.ai;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public interface ActivationFunction {

    void apply(float[] input, float[] output);

    // Vector Species for Float
    VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    ActivationFunction RELU = (input, output) -> {
        int i = 0;
        int upperBound = SPECIES.loopBound(input.length);

        FloatVector zeros = FloatVector.zero(SPECIES);

        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, input, i);
            // ReLU: max(0, x)
            FloatVector vr = va.max(zeros);
            vr.intoArray(output, i);
        }

        // Scalar loop for tail
        for (; i < input.length; i++) {
            output[i] = Math.max(0.0f, input[i]);
        }
    };

    ActivationFunction SIGMOID = (input, output) -> {
        // Scalar loop for Sigmoid since exp() is not in Vector API yet
        for (int i = 0; i < input.length; i++) {
            float x = input[i];
            output[i] = (float) (1.0f / (1.0f + Math.exp(-x)));
        }
    };
}
