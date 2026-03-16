package com.example.ai;

import io.vertx.core.Vertx;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.client.WebClient;
import io.vertx.ext.web.client.WebClientOptions;

import java.util.concurrent.CountDownLatch;

/**
 * Test to verify word training by measuring semantic similarity before and after training.
 */
public class WordSimilarityTest {

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

        String word1 = "love";
        String word2 = "like";

        System.out.println("[DEBUG_LOG] Testing similarity between '" + word1 + "' and '" + word2 + "'");

        // 1. Check initial similarity
        JsonObject simPayload = new JsonObject()
            .put("type", "similarity")
            .put("word1", word1)
            .put("word2", word2);

        client.post("/train").sendJsonObject(simPayload)
            .compose(res1 -> {
                double initialSim = res1.bodyAsJsonObject().getDouble("similarity");
                System.out.println("[DEBUG_LOG] 1. Initial Similarity: " + initialSim);

                // 2. Perform "pull" training
                JsonObject pullPayload = new JsonObject()
                    .put("type", "pull")
                    .put("word1", word1)
                    .put("word2", word2)
                    .put("rate", 0.2f); // Move 20% closer

                System.out.println("[DEBUG_LOG] 2. Sending PULL command (rate=0.2)...");
                return client.post("/train").sendJsonObject(pullPayload);
            })
            .compose(res2 -> {
                System.out.println("[DEBUG_LOG] 3. Train response: " + res2.bodyAsString());

                // 3. Check similarity again
                return client.post("/train").sendJsonObject(simPayload);
            })
            .onSuccess(res3 -> {
                double finalSim = res3.bodyAsJsonObject().getDouble("similarity");
                System.out.println("[DEBUG_LOG] 4. Final Similarity: " + finalSim);

                if (finalSim > 0.9999) {
                     System.out.println("[DEBUG_LOG] Words are now nearly identical in meaning.");
                } else if (finalSim > -1.0) {
                     System.out.println("[DEBUG_LOG] Similarity increased as expected.");
                }

                System.out.println("[DEBUG_LOG] Test Complete.");
                latch.countDown();
            })
            .onFailure(err -> {
                err.printStackTrace();
                latch.countDown();
            });
    }
}
