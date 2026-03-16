package com.example.ai;

import io.vertx.core.Vertx;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.client.WebClient;
import io.vertx.ext.web.client.WebClientOptions;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CountDownLatch;

/**
 * Demonstrates a Real-World Case: Semantic Search / Indexing.
 *
 * Flow:
 * 1. Define 3 "Files" with text content.
 * 2. Vectorize each "File" using the /infer endpoint (Indexing).
 * 3. Store these vectors in a local "Dictionary" (Semantic Index).
 * 4. User provides a Query.
 * 5. Vectorize the Query using /infer.
 * 6. Compare Query Vector against Semantic Index using Cosine Similarity.
 * 7. Filter results by threshold (> 0.9) to return matching files.
 */
public class RealWorldSemanticSearchTest {

    public static void main(String[] args) throws InterruptedException {
        Vertx vertx = Vertx.vertx();
        CountDownLatch latch = new CountDownLatch(1);

        vertx.deployVerticle(new MainVerticle())
            .onSuccess(id -> {
                System.out.println("[DEBUG_LOG] Server started for Real-World Semantic Search.");
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

        // 1. Training - Improve semantic separation
        String fruitsBatch = "pull 0.4 apple banana orange\n".repeat(100);
        String techBatch = "pull 0.4 code laptop cloud\n".repeat(100);
        String weatherBatch = "pull 0.4 sunny sky warm\n".repeat(100);
        String combinedBatch = fruitsBatch + techBatch + weatherBatch;

        System.out.println("[DEBUG_LOG] 1. Training semantic groups...");
        client.post("/train").sendBuffer(io.vertx.core.buffer.Buffer.buffer(combinedBatch))
            .compose(res -> {
                System.out.println("[DEBUG_LOG] 2. Training complete. Indexing documents...");

                // 2. Define some "Files" (Documents)
                Map<String, String> documents = new HashMap<>();
                documents.put("fruit_info.txt", "apple banana orange fresh fruit juice");
                documents.put("tech_news.txt", "laptop code cloud java vertx vector api");
                documents.put("weather.txt", "sunny day blue sky warm temperature");

                // 3. Indexing: Vectorize all documents
                Map<String, float[]> semanticIndex = new HashMap<>();

                return io.vertx.core.Future.<Map<String, float[]>>future(promise -> 
                    indexDocuments(client, documents, semanticIndex, () -> promise.complete(semanticIndex))
                );
            })
            .onSuccess(semanticIndex -> {
                System.out.println("[DEBUG_LOG] 3. Indexing complete. Total indexed: " + semanticIndex.size());

                // 4. Search Query
                String query = "i want to eat some apple and banana";
                System.out.println("[DEBUG_LOG] 4. Searching for query: \"" + query + "\"");

                client.post("/infer").sendBuffer(io.vertx.core.buffer.Buffer.buffer(query))
                    .onSuccess(res -> {
                        float[] queryVector = parseVector(res.bodyAsJsonArray());

                        // 5. Semantic Matching
                        System.out.println("[DEBUG_LOG] 5. Results:");
                        
                        // Find the best match
                        String bestFile = null;
                        double maxSim = -1.0;

                        for (Map.Entry<String, float[]> entry : semanticIndex.entrySet()) {
                            double sim = cosineSimilarity(queryVector, entry.getValue());
                            System.out.println(String.format("[DEBUG_LOG]    File: %-15s | Similarity: %.4f", entry.getKey(), sim));

                            if (sim > maxSim) {
                                maxSim = sim;
                                bestFile = entry.getKey();
                            }
                        }

                        if (bestFile != null) {
                            System.out.println("[DEBUG_LOG] 6. Best Match: " + bestFile + " (Confidence: " + String.format("%.4f", maxSim) + ")");
                            if (bestFile.equals("fruit_info.txt")) {
                                System.out.println("[DEBUG_LOG] SUCCESS: Correctly identified the Fruit document!");
                            }
                        }

                        System.out.println("[DEBUG_LOG] Test Complete.");
                        latch.countDown();
                    })
                    .onFailure(err -> {
                        err.printStackTrace();
                        latch.countDown();
                    });
            })
            .onFailure(err -> {
                err.printStackTrace();
                latch.countDown();
            });
    }

    private static void indexDocuments(WebClient client, Map<String, String> documents, Map<String, float[]> index, Runnable onComplete) {
        java.util.Iterator<Map.Entry<String, String>> it = documents.entrySet().iterator();
        indexNext(client, it, index, onComplete);
    }

    private static void indexNext(WebClient client, java.util.Iterator<Map.Entry<String, String>> it, Map<String, float[]> index, Runnable onComplete) {
        if (!it.hasNext()) {
            onComplete.run();
            return;
        }

        Map.Entry<String, String> entry = it.next();
        client.post("/infer").sendBuffer(io.vertx.core.buffer.Buffer.buffer(entry.getValue()))
            .onSuccess(res -> {
                index.put(entry.getKey(), parseVector(res.bodyAsJsonArray()));
                indexNext(client, it, index, onComplete);
            })
            .onFailure(err -> {
                err.printStackTrace();
                onComplete.run();
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
