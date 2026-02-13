package com.example.ai;

import io.vertx.core.Vertx;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.client.WebClient;
import io.vertx.ext.web.client.WebClientOptions;

import java.util.concurrent.CountDownLatch;

public class EmbeddingTrainingTest {

    public static void main(String[] args) throws InterruptedException {
        Vertx vertx = Vertx.vertx();
        CountDownLatch latch = new CountDownLatch(1);

        // Deploy Server
        vertx.deployVerticle(new MainVerticle())
            .onSuccess(id -> {
                System.out.println("Server started.");
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

        // 1. Infer "love"
        client.post("/infer").sendBuffer(io.vertx.core.buffer.Buffer.buffer("love"))
            .compose(res1 -> {
                System.out.println("1. Initial 'love' output: " + res1.bodyAsString());

                // 2. Infer "like"
                return client.post("/infer").sendBuffer(io.vertx.core.buffer.Buffer.buffer("like"));
            })
            .compose(res2 -> {
                System.out.println("2. Initial 'like' output: " + res2.bodyAsString());

                // 3. Train: Pull 'love' and 'like' closer (High rate for visibility)
                JsonObject payload = new JsonObject()
                    .put("type", "pull")
                    .put("word1", "love")
                    .put("word2", "like")
                    .put("rate", 0.5f);

                System.out.println("3. Sending PULL command...");
                return client.post("/train").sendJsonObject(payload);
            })
            .compose(resTrain -> {
                System.out.println("4. Train response: " + resTrain.bodyAsString());

                // 4. Infer "love" again
                return client.post("/infer").sendBuffer(io.vertx.core.buffer.Buffer.buffer("love"));
            })
            .compose(res3 -> {
                System.out.println("5. Updated 'love' output: " + res3.bodyAsString());

                // 5. Infer "like" again
                return client.post("/infer").sendBuffer(io.vertx.core.buffer.Buffer.buffer("like"));
            })
            .onSuccess(res4 -> {
                System.out.println("6. Updated 'like' output: " + res4.bodyAsString());
                System.out.println("Test Complete. Check if (5) and (6) are closer than (1) and (2).");
                latch.countDown();
            })
            .onFailure(err -> {
                err.printStackTrace();
                latch.countDown();
            });
    }
}
