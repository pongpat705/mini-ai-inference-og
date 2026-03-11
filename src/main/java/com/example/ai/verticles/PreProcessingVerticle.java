package com.example.ai.verticles;

import io.vertx.core.AbstractVerticle;
import io.vertx.core.Promise;
import io.vertx.core.eventbus.DeliveryOptions;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
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
        String[] words = {"hello", "world", "ai", "java", "vertx", "vector", "api", "inference", "high", "performance", "love", "like", "apple", "banana", "orange", "laptop", "code", "cloud"};
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
                    // Pull multiple words closer
                    JsonArray wordsArray = command.getJsonArray("words");
                    Float rate = command.getFloat("rate");
                    float r = (rate != null) ? rate : 0.1f;

                    if (wordsArray != null && wordsArray.size() >= 2) {
                        java.util.List<float[]> vecs = new java.util.ArrayList<>();
                        java.util.List<String> foundWords = new java.util.ArrayList<>();
                        for (int i = 0; i < wordsArray.size(); i++) {
                            String w = wordsArray.getString(i);
                            if (embeddings.containsKey(w)) {
                                vecs.add(embeddings.get(w));
                                foundWords.add(w);
                            }
                        }

                        if (vecs.size() >= 2) {
                            // Run pull multiple times for faster convergence during training
                            for (int iter = 0; iter < 5; iter++) {
                                pullMultipleVectors(vecs, r);
                            }
                            message.reply(new JsonObject().put("status", "pulled").put("words", new JsonArray(foundWords)));
                        } else {
                            message.fail(404, "Not enough words found in vocabulary: " + wordsArray.toString());
                        }
                    } else {
                        // Compatibility for old word1, word2 format
                        String w1 = command.getString("word1");
                        String w2 = command.getString("word2");
                        if (w1 != null && w2 != null && embeddings.containsKey(w1) && embeddings.containsKey(w2)) {
                            pullVectors(embeddings.get(w1), embeddings.get(w2), r);
                            message.reply(new JsonObject().put("status", "pulled").put("word1", w1).put("word2", w2));
                        } else {
                            message.fail(400, "Missing 'words' array or 'word1'/'word2' fields");
                        }
                    }
                } else if ("batch".equalsIgnoreCase(type)) {
                    // Process text data for batch training
                    String text = command.getString("data");
                    if (text != null) {
                        processBatchTraining(text);
                        message.reply(new JsonObject().put("status", "batch_processed"));
                    } else {
                        message.fail(400, "Missing 'data' field for batch training");
                    }
                } else if ("similarity".equalsIgnoreCase(type)) {
                    // Calculate Cosine Similarity between two words
                    String w1 = command.getString("word1");
                    String w2 = command.getString("word2");

                    if (embeddings.containsKey(w1) && embeddings.containsKey(w2)) {
                        float[] v1 = embeddings.get(w1);
                        float[] v2 = embeddings.get(w2);
                        float sim = cosineSimilarity(v1, v2);
                        message.reply(new JsonObject()
                            .put("word1", w1)
                            .put("word2", w2)
                            .put("similarity", sim));
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

    private void pullMultipleVectors(java.util.List<float[]> vecs, float rate) {
        // Calculate Centroid
        float[] centroid = new float[embeddingDim];
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        int upperBound = species.loopBound(embeddingDim);

        for (float[] v : vecs) {
            int i = 0;
            for (; i < upperBound; i += species.length()) {
                var vCentroid = FloatVector.fromArray(species, centroid, i);
                var vVec = FloatVector.fromArray(species, v, i);
                vCentroid.add(vVec).intoArray(centroid, i);
            }
            for (; i < embeddingDim; i++) {
                centroid[i] += v[i];
            }
        }

        int numVecs = vecs.size();
        for (int i = 0; i < embeddingDim; i++) {
            centroid[i] /= numVecs;
        }

        // Pull each vector towards the centroid
        for (float[] v : vecs) {
            int j = 0;
            for (; j < upperBound; j += species.length()) {
                var vv = FloatVector.fromArray(species, v, j);
                var vCentroid = FloatVector.fromArray(species, centroid, j);
                var diff = vCentroid.sub(vv).mul(rate);
                vv.add(diff).intoArray(v, j);
            }
            for (; j < embeddingDim; j++) {
                v[j] += (centroid[j] - v[j]) * rate;
            }
        }
    }

    private void processBatchTraining(String data) {
        String[] lines = data.split("\\r?\\n");
        for (String line : lines) {
            line = line.trim();
            if (line.isEmpty()) continue;

            String[] parts = line.split("\\s+");
            if (parts.length < 3) continue;

            String type = parts[0];
            try {
                float rate = Float.parseFloat(parts[1]);
                java.util.List<float[]> vecs = new java.util.ArrayList<>();
                for (int i = 2; i < parts.length; i++) {
                    String word = parts[i].toLowerCase();
                    if (embeddings.containsKey(word)) {
                        vecs.add(embeddings.get(word));
                    }
                }

                if (vecs.size() >= 2) {
                    if ("pull".equalsIgnoreCase(type)) {
                        pullMultipleVectors(vecs, rate);
                    }
                }
            } catch (NumberFormatException e) {
                // Ignore invalid lines
            }
        }
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

    private float cosineSimilarity(float[] v1, float[] v2) {
        float dotProduct = 0.0f;
        float normA = 0.0f;
        float normB = 0.0f;

        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        int i = 0;
        int upperBound = species.loopBound(embeddingDim);

        for (; i < upperBound; i += species.length()) {
            var vv1 = FloatVector.fromArray(species, v1, i);
            var vv2 = FloatVector.fromArray(species, v2, i);

            dotProduct += vv1.mul(vv2).reduceLanes(VectorOperators.ADD);
            normA += vv1.mul(vv1).reduceLanes(VectorOperators.ADD);
            normB += vv2.mul(vv2).reduceLanes(VectorOperators.ADD);
        }

        for (; i < embeddingDim; i++) {
            dotProduct += v1[i] * v2[i];
            normA += v1[i] * v1[i];
            normB += v2[i] * v2[i];
        }

        if (normA == 0 || normB == 0) return 0.0f;
        return (float) (dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)));
    }
}
