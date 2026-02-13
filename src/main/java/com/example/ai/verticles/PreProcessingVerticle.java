package com.example.ai.verticles;

import io.vertx.core.AbstractVerticle;
import io.vertx.core.Promise;
import io.vertx.core.eventbus.DeliveryOptions;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class PreProcessingVerticle extends AbstractVerticle {

    public static final String ADDRESS_IN = "ai.preprocess";
    public static final String ADDRESS_OUT = "ai.inference";
    public static final String ADDRESS_CONTROL = "ai.embedding.control";

    // Store Dense Embeddings
    private final Map<String, float[]> embeddings = new HashMap<>();
    private final int embeddingDim; // e.g. 64

    public PreProcessingVerticle(int embeddingDim) {
        this.embeddingDim = embeddingDim;
    }

    @Override
    public void start(Promise<Void> startPromise) {
        // Initialize vocabulary with random dense vectors
        String[] words = {"hello", "world", "ai", "java", "vertx", "vector", "api", "inference", "high", "performance", "love", "like"};
        Random rand = new Random();

        for (String word : words) {
            float[] vector = new float[embeddingDim];
            for (int i = 0; i < embeddingDim; i++) {
                vector[i] = rand.nextFloat() - 0.5f;
            }
            embeddings.put(word, vector);
        }

        // 1. Text Preprocessing Handler
        vertx.eventBus().consumer(ADDRESS_IN, message -> {
            Object body = message.body();
            String text = body != null ? body.toString() : "";

            float[] vector = textToDenseVector(text);

            // Send to Inference Verticle using Buffer
            io.vertx.core.buffer.Buffer buffer = io.vertx.core.buffer.Buffer.buffer(vector.length * 4);
            for (float f : vector) {
                buffer.appendFloat(f);
            }

            vertx.eventBus().request(ADDRESS_OUT, buffer, new DeliveryOptions().setSendTimeout(5000))
                .onSuccess(reply -> message.reply(reply.body()))
                .onFailure(err -> message.fail(500, err != null ? err.getMessage() : "Unknown error"));
        });

        // 2. Training/Control Handler
        vertx.eventBus().consumer(ADDRESS_CONTROL, message -> {
            try {
                JsonObject command = (JsonObject) message.body();
                String type = command.getString("type");

                if ("update".equalsIgnoreCase(type)) {
                    // Update specific word vector
                    String word = command.getString("word");
                    if (word != null && command.containsKey("vector")) {
                         // Parse vector from JSON array
                         JsonArray jsonArray = command.getJsonArray("vector");
                         if (jsonArray.size() == embeddingDim) {
                             float[] newVec = new float[embeddingDim];
                             for(int i=0; i<embeddingDim; i++) {
                                 newVec[i] = jsonArray.getFloat(i);
                             }
                             embeddings.put(word, newVec);
                             message.reply(new JsonObject().put("status", "updated").put("word", word));
                         } else {
                             message.fail(400, "Vector dimension mismatch: expected " + embeddingDim + ", got " + jsonArray.size());
                         }
                    } else {
                        message.fail(400, "Missing 'word' or 'vector' field");
                    }
                } else if ("pull".equalsIgnoreCase(type)) {
                    // Pull two words closer
                    String w1 = command.getString("word1");
                    String w2 = command.getString("word2");
                    Float rate = command.getFloat("rate");
                    float r = (rate != null) ? rate : 0.1f;

                    if (embeddings.containsKey(w1) && embeddings.containsKey(w2)) {
                        float[] v1 = embeddings.get(w1);
                        float[] v2 = embeddings.get(w2);
                        pullVectors(v1, v2, r);
                        message.reply(new JsonObject().put("status", "pulled").put("word1", w1).put("word2", w2));
                    } else {
                        message.fail(404, "Words not found: " + w1 + ", " + w2);
                    }
                } else {
                    message.fail(400, "Unknown command type: " + type);
                }
            } catch (Exception e) {
                e.printStackTrace();
                message.fail(500, e.getMessage());
            }
        });

        startPromise.complete();
    }

    private float[] textToDenseVector(String text) {
        float[] sumVector = new float[embeddingDim];
        if (text == null || text.isEmpty()) return sumVector;

        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;

        String[] tokens = text.toLowerCase().split("\\s+");
        for (String word : tokens) {
            float[] vec = embeddings.get(word);
            if (vec != null) {
                // Vectorized Add: sumVector += vec
                int i = 0;
                int upperBound = species.loopBound(embeddingDim);
                for (; i < upperBound; i += species.length()) {
                    var vSum = FloatVector.fromArray(species, sumVector, i);
                    var vWord = FloatVector.fromArray(species, vec, i);
                    vSum.add(vWord).intoArray(sumVector, i);
                }
                for (; i < embeddingDim; i++) {
                    sumVector[i] += vec[i];
                }
            }
        }
        return sumVector;
    }

    private void pullVectors(float[] v1, float[] v2, float rate) {
        // v1 = v1 + rate * (v2 - v1)
        // v2 = v2 + rate * (v1 - v2)

        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        int i = 0;
        int upperBound = species.loopBound(embeddingDim);

        for (; i < upperBound; i += species.length()) {
            var vv1 = FloatVector.fromArray(species, v1, i);
            var vv2 = FloatVector.fromArray(species, v2, i);

            var diff1 = vv2.sub(vv1).mul(rate);
            var diff2 = vv1.sub(vv2).mul(rate);

            vv1.add(diff1).intoArray(v1, i);
            vv2.add(diff2).intoArray(v2, i);
        }
        for (; i < embeddingDim; i++) {
            float d1 = (v2[i] - v1[i]) * rate;
            float d2 = (v1[i] - v2[i]) * rate;
            v1[i] += d1;
            v2[i] += d2;
        }
    }
}
