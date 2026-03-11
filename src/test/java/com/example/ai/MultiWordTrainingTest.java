package com.example.ai;

import io.vertx.core.Vertx;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.client.WebClient;
import io.vertx.ext.web.client.WebClientOptions;

import java.util.concurrent.CountDownLatch;

/**
 * Test to verify multi-word training and batch file training.
 */
public class MultiWordTrainingTest {

    public static void main(String[] args) throws InterruptedException {
        Vertx vertx = Vertx.vertx();
        CountDownLatch latch = new CountDownLatch(1);

        // Deploy Server
        vertx.deployVerticle(new MainVerticle())
            .onSuccess(id -> {
                System.out.println("[DEBUG_LOG] Server started.");
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

        String w1 = "hello";
        String w2 = "world";
        String w3 = "ai";

        System.out.println("[DEBUG_LOG] Testing similarity between '" + w1 + "', '" + w2 + "', and '" + w3 + "'");

        // 1. Check initial similarities
        checkSimilarity(client, w1, w2)
            .compose(sim12_initial -> {
                System.out.println("[DEBUG_LOG] Initial Similarity (" + w1 + ", " + w2 + "): " + sim12_initial);
                return checkSimilarity(client, w2, w3);
            })
            .compose(sim23_initial -> {
                System.out.println("[DEBUG_LOG] Initial Similarity (" + w2 + ", " + w3 + "): " + sim23_initial);

                // 2. Multi-word pull via JSON
                JsonObject pullPayload = new JsonObject()
                    .put("type", "pull")
                    .put("words", new JsonArray().add(w1).add(w2).add(w3))
                    .put("rate", 0.5f);

                System.out.println("[DEBUG_LOG] Sending multi-word PULL command...");
                return client.post("/train").sendJsonObject(pullPayload);
            })
            .compose(res -> {
                System.out.println("[DEBUG_LOG] Multi-word PULL response: " + res.bodyAsString());

                // 3. Check updated similarities
                return checkSimilarity(client, w1, w2);
            })
            .compose(sim12_after -> {
                System.out.println("[DEBUG_LOG] After PULL Similarity (" + w1 + ", " + w2 + "): " + sim12_after);
                return checkSimilarity(client, w2, w3);
            })
            .compose(sim23_after -> {
                System.out.println("[DEBUG_LOG] After PULL Similarity (" + w2 + ", " + w3 + "): " + sim23_after);

                // 4. Batch training via text
                String batchData = "pull 0.8 java vertx vector api\n" +
                                   "pull 0.8 performance high";
                System.out.println("[DEBUG_LOG] Sending batch training data...");
                return client.post("/train").sendBuffer(io.vertx.core.buffer.Buffer.buffer(batchData));
            })
            .compose(res -> {
                System.out.println("[DEBUG_LOG] Batch training response: " + res.bodyAsString());

                // 5. Verify batch training results
                return checkSimilarity(client, "java", "vertx");
            })
            .onSuccess(sim_java_vertx -> {
                System.out.println("[DEBUG_LOG] After Batch Similarity (java, vertx): " + sim_java_vertx);
                System.out.println("[DEBUG_LOG] Test Complete.");
                latch.countDown();
            })
            .onFailure(err -> {
                err.printStackTrace();
                latch.countDown();
            });
    }

    private static io.vertx.core.Future<Double> checkSimilarity(WebClient client, String w1, String w2) {
        JsonObject payload = new JsonObject()
            .put("type", "similarity")
            .put("word1", w1)
            .put("word2", w2);
        return client.post("/train").sendJsonObject(payload)
            .map(res -> res.bodyAsJsonObject().getDouble("similarity"));
    }
}
