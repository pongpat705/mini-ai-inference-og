package com.example.ai.verticles;

import io.vertx.core.AbstractVerticle;
import io.vertx.core.Promise;
import io.vertx.core.eventbus.DeliveryOptions;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

public class PreProcessingVerticle extends AbstractVerticle {

    public static final String ADDRESS_IN = "ai.preprocess";
    public static final String ADDRESS_OUT = "ai.inference";

    private final Map<String, Integer> vocabulary = new HashMap<>();
    private final int vectorSize;

    public PreProcessingVerticle(int vectorSize) {
        this.vectorSize = vectorSize;
    }

    @Override
    public void start(Promise<Void> startPromise) {
        // Initialize simple vocabulary
        String[] words = {"hello", "world", "ai", "java", "vertx", "vector", "api", "inference", "high", "performance"};
        for (int i = 0; i < words.length && i < vectorSize; i++) {
            vocabulary.put(words[i], i);
        }

        vertx.eventBus().consumer(ADDRESS_IN, message -> {
            Object body = message.body();
            String text = body != null ? body.toString() : "";

            float[] vector = textToVector(text);

            // Send to Inference Verticle using Buffer for efficiency
            io.vertx.core.buffer.Buffer buffer = io.vertx.core.buffer.Buffer.buffer(vector.length * 4);
            for (float f : vector) {
                buffer.appendFloat(f);
            }

            vertx.eventBus().request(ADDRESS_OUT, buffer, new DeliveryOptions().setSendTimeout(5000))
                .onSuccess(reply -> message.reply(reply.body()))
                .onFailure(err -> message.fail(500, err != null ? err.getMessage() : "Unknown error"));
        });

        startPromise.complete();
    }

    private float[] textToVector(String text) {
        float[] vector = new float[vectorSize];
        if (text == null || text.isEmpty()) return vector;

        // Simple bag-of-words
        Stream.of(text.toLowerCase().split("\\s+"))
              .forEach(word -> {
                  Integer index = vocabulary.get(word);
                  if (index != null && index < vectorSize) {
                      vector[index] += 1.0f;
                  }
              });
        return vector;
    }
}
