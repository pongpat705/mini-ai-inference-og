package com.example.ai.verticles;

import com.example.ai.ActivationFunction;
import com.example.ai.LinearLayer;
import io.vertx.core.AbstractVerticle;
import io.vertx.core.Promise;
import io.vertx.core.buffer.Buffer;
import java.util.concurrent.atomic.AtomicInteger;

public class InferenceVerticle extends AbstractVerticle {

    public static final String ADDRESS = "ai.inference";
    private LinearLayer linearLayer;
    private final int inputDim;
    private final int outputDim;

    // Backpressure
    private final AtomicInteger inflight = new AtomicInteger(0);
    private static final int MAX_INFLIGHT = 100; // Limit pending inference tasks per instance

    public InferenceVerticle(int inputDim, int outputDim) {
        this.inputDim = inputDim;
        this.outputDim = outputDim;
    }

    @Override
    public void start(Promise<Void> startPromise) {
        // Initialize AI Core
        // Using ReLU and creating the layer
        // In a real scenario, weights would be loaded from disk or shared memory
        linearLayer = new LinearLayer(inputDim, outputDim, ActivationFunction.RELU);

        vertx.eventBus().consumer(ADDRESS, message -> {
            // Backpressure: Flow Control
            if (inflight.get() >= MAX_INFLIGHT) {
                message.fail(503, "System Overloaded: Too many pending inference requests.");
                return;
            }

            Buffer body = (Buffer) message.body();
            inflight.incrementAndGet();

            // Offload heavy calculation to worker thread to avoid blocking Event Loop
            vertx.executeBlocking(() -> {
                // Parse input
                float[] input = new float[inputDim];
                // Read from buffer (assuming size match or handle safely)
                int len = Math.min(inputDim, body.length() / 4);
                for (int i = 0; i < len; i++) {
                    input[i] = body.getFloat(i * 4);
                }

                // Inference (Vector API Optimized)
                return linearLayer.forward(input);
            }, false).onComplete(res -> {
                inflight.decrementAndGet();

                if (res.succeeded()) {
                    float[] output = res.result();
                    // Reply with Buffer
                    Buffer outBuf = Buffer.buffer(output.length * 4);
                    for (float f : output) {
                        outBuf.appendFloat(f);
                    }
                    message.reply(outBuf);
                } else {
                    message.fail(500, res.cause() != null ? res.cause().getMessage() : "Inference Failed");
                }
            });
        });

        startPromise.complete();
    }
}
