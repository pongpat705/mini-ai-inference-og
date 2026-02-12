package com.example.ai.benchmark;

import com.example.ai.ActivationFunction;
import com.example.ai.LinearLayer;
import org.openjdk.jmh.annotations.*;
import java.util.Random;
import java.util.concurrent.TimeUnit;

@State(Scope.Benchmark)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Fork(value = 1, jvmArgs = {"--add-modules", "jdk.incubator.vector", "--enable-preview"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class VectorBenchmark {

    @Param({"1024"})
    public int inputDim;

    @Param({"128"})
    public int outputDim;

    private LinearLayer linearLayer;
    private float[] input;

    // For baseline
    private float[] weightsArray;
    private float[] biasArray;

    @Setup(Level.Trial)
    public void setup() {
        // Initialize optimized layer
        linearLayer = new LinearLayer(inputDim, outputDim, ActivationFunction.RELU);

        Random rand = new Random();
        input = new float[inputDim];
        for (int i = 0; i < inputDim; i++) input[i] = rand.nextFloat();

        // Setup baseline arrays (simulating standard Java implementation)
        weightsArray = new float[inputDim * outputDim];
        biasArray = new float[outputDim];
        for (int i = 0; i < weightsArray.length; i++) weightsArray[i] = rand.nextFloat();
        for (int i = 0; i < biasArray.length; i++) biasArray[i] = rand.nextFloat();
    }

    @Benchmark
    public float[] vectorApiOffHeap() {
        return linearLayer.forward(input);
    }

    @Benchmark
    public float[] scalarLoopOnHeap() {
        float[] output = new float[outputDim];

        // Matrix Multiplication Loop
        for (int j = 0; j < outputDim; j++) {
            float sum = 0.0f;
            int rowOffset = j * inputDim;
            for (int i = 0; i < inputDim; i++) {
                sum += input[i] * weightsArray[rowOffset + i];
            }
            sum += biasArray[j];

            // ReLU Activation
            output[j] = Math.max(0.0f, sum);
        }
        return output;
    }
}
