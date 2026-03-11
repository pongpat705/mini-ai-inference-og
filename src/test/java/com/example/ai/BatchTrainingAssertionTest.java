package com.example.ai;

import io.vertx.core.Vertx;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.client.WebClient;
import io.vertx.ext.web.client.WebClientOptions;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;

/**
 * Batch Training Assertion Test
 * This test loads training_data.txt, measures similarity before and after batch training,
 * and asserts that the similarity has increased for the training cases.
 */
public class BatchTrainingAssertionTest {

    public static void main(String[] args) throws InterruptedException {
        Vertx vertx = Vertx.vertx();
        CountDownLatch latch = new CountDownLatch(1);

        vertx.deployVerticle(new MainVerticle())
            .onSuccess(id -> {
                System.out.println("[DEBUG_LOG] Server started for Batch Training Test.");
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
        String filePath = "training_data.txt";

        try {
            List<String> lines = Files.readAllLines(Paths.get(filePath));
            if (lines.isEmpty()) {
                System.err.println("[DEBUG_LOG] training_data.txt is empty!");
                latch.countDown();
                return;
            }

            // Let's pick a few cases to verify
            String testLine = lines.get(0); // e.g., "pull 0.1 love like affection adoration"
            String[] parts = testLine.split("\\s+");
            String word1 = parts[2];
            String word2 = parts[3];

            System.out.println("[DEBUG_LOG] Verifying case: " + word1 + " <-> " + word2);

            // 1. Get initial similarity
            getSimilarity(client, word1, word2).compose(initialSim -> {
                System.out.println("[DEBUG_LOG] Initial Similarity: " + initialSim);

                // 2. Perform Batch Training
                Buffer batchData = Buffer.buffer(String.join("\n", lines));
                System.out.println("[DEBUG_LOG] Sending batch training data (" + lines.size() + " cases)...");
                return client.post("/train")
                        .putHeader("Content-Type", "text/plain")
                        .sendBuffer(batchData)
                        .compose(res -> {
                            System.out.println("[DEBUG_LOG] Batch training response: " + res.bodyAsString());
                            // 3. Get final similarity
                            return getSimilarity(client, word1, word2);
                        })
                        .onSuccess(finalSim -> {
                            System.out.println("[DEBUG_LOG] Final Similarity: " + finalSim);
                            if (finalSim > initialSim) {
                                System.out.println("[DEBUG_LOG] ASSERTION PASSED: Similarity increased from " + initialSim + " to " + finalSim);
                            } else {
                                System.err.println("[DEBUG_LOG] ASSERTION FAILED: Similarity did not increase! (Initial: " + initialSim + ", Final: " + finalSim + ")");
                            }
                            latch.countDown();
                        });
            }).onFailure(err -> {
                err.printStackTrace();
                latch.countDown();
            });

        } catch (IOException e) {
            e.printStackTrace();
            latch.countDown();
        }
    }

    private static io.vertx.core.Future<Double> getSimilarity(WebClient client, String w1, String w2) {
        JsonObject simPayload = new JsonObject()
                .put("type", "similarity")
                .put("word1", w1)
                .put("word2", w2);

        return client.post("/train").sendJsonObject(simPayload)
                .map(res -> res.bodyAsJsonObject().getDouble("similarity"));
    }
}
