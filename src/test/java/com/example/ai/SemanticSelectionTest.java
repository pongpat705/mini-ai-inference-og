package com.example.ai;

import io.vertx.core.Vertx;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.client.WebClient;
import io.vertx.ext.web.client.WebClientOptions;

import java.util.concurrent.CountDownLatch;

/**
 * Demonstrates "Semantic Selection" (Classification).
 * We train 2 groups: FRUITS (apple, banana) and TECH (code, cloud).
 * We then infer an input and "select" the group it's most similar to.
 */
public class SemanticSelectionTest {

    public static void main(String[] args) throws InterruptedException {
        Vertx vertx = Vertx.vertx();
        CountDownLatch latch = new CountDownLatch(1);

        vertx.deployVerticle(new MainVerticle())
            .onSuccess(id -> {
                System.out.println("[DEBUG_LOG] Server started for Semantic Selection.");
                runTest(vertx, latch);
            })
            .onFailure(err -> {
                err.printStackTrace();
                System.exit(1);
            });

        latch.await();
        vertx.close();
    }

    private static void runTest(Vertx vertx, CountDownLatch latch) {
        WebClient client = WebClient.create(vertx, new WebClientOptions().setDefaultHost("localhost").setDefaultPort(8080));

        // 1. Training - Pull groups together
        // Increased iterations for much better separation
        String fruitsBatch = "pull 0.8 apple banana orange\n".repeat(100);
        String techBatch = "pull 0.8 code laptop cloud\n".repeat(100);
        String combinedBatch = fruitsBatch + techBatch;

        System.out.println("[DEBUG_LOG] 1. Training 2 groups (FRUIT vs TECH)...");
        client.post("/train").sendBuffer(io.vertx.core.buffer.Buffer.buffer(combinedBatch))
            .compose(res -> {
                System.out.println("[DEBUG_LOG] 2. Training complete.");

                // Now test classification for 'apple' and 'banana'
                return client.post("/infer").sendBuffer(io.vertx.core.buffer.Buffer.buffer("apple"))
                    .compose(resApple -> {
                        float[] appleVec = parseVector(resApple.bodyAsJsonArray());
                        return client.post("/infer").sendBuffer(io.vertx.core.buffer.Buffer.buffer("banana"))
                            .compose(resBanana -> {
                                float[] bananaVec = parseVector(resBanana.bodyAsJsonArray());
                                return client.post("/infer").sendBuffer(io.vertx.core.buffer.Buffer.buffer("code"))
                                    .map(resCode -> {
                                        float[] codeVec = parseVector(resCode.bodyAsJsonArray());
                                        return new float[][]{appleVec, bananaVec, codeVec};
                                    });
                            });
                    });
            })
            .onSuccess(vectors -> {
                float[] appleVec = vectors[0];
                float[] bananaVec = vectors[1];
                float[] codeVec = vectors[2];

                // Calculate Cosine Similarity on the inference output vectors
                double fruitSim = cosineSimilarity(appleVec, bananaVec);
                double crossSim = cosineSimilarity(appleVec, codeVec);

                System.out.println("[DEBUG_LOG] --- CLASSIFICATION RESULTS ---");
                System.out.println("[DEBUG_LOG] Input: 'apple'");
                System.out.println("[DEBUG_LOG] Similarity with 'banana' (Same Category): " + String.format("%.4f", fruitSim));
                System.out.println("[DEBUG_LOG] Similarity with 'code'   (Other Category): " + String.format("%.4f", crossSim));

                if (fruitSim > crossSim + 0.05) {
                    System.out.println("[DEBUG_LOG] SELECTION SUCCESS: 'apple' correctly identified as more similar to Fruits than Tech!");
                } else {
                    System.out.println("[DEBUG_LOG] SELECTION FAILED: Similarity did not distinguish groups clearly.");
                }

                latch.countDown();
            })
            .onFailure(err -> {
                err.printStackTrace();
                latch.countDown();
            });
    }

    private static float[] parseVector(JsonArray arr) {
        float[] v = new float[arr.size()];
        for (int i = 0; i < arr.size(); i++) {
            v[i] = arr.getFloat(i);
        }
        return v;
    }

    private static double cosineSimilarity(float[] v1, float[] v2) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < v1.length; i++) {
            dotProduct += v1[i] * v2[i];
            normA += Math.pow(v1[i], 2);
            normB += Math.pow(v2[i], 2);
        }
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}
