package com.example.ai;

import io.vertx.core.Vertx;
import io.vertx.ext.web.client.WebClient;
import io.vertx.ext.web.client.WebClientOptions;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

public class LoadTest {

    public static void main(String[] args) throws InterruptedException {
        // 1. Start the Server (MainVerticle) locally for testing
        // Or connect to existing server if args provided
        boolean startLocal = true;
        if (args.length > 0 && args[0].equals("remote")) {
            startLocal = false;
        }

        Vertx vertx = Vertx.vertx();

        if (startLocal) {
            CountDownLatch startLatch = new CountDownLatch(1);
            vertx.deployVerticle(new MainVerticle())
                .onSuccess(id -> {
                    System.out.println("Server started successfully.");
                    startLatch.countDown();
                })
                .onFailure(err -> {
                    err.printStackTrace();
                    System.exit(1);
                });
            startLatch.await();
        }

        // 2. Setup Load Test
        WebClientOptions options = new WebClientOptions()
            .setKeepAlive(true);
        // setMaxPoolSize is inherited from HttpClientOptions. In some versions return type might not be covariant?
        // Let's use it as HttpClientOptions if needed, but WebClientOptions should have it.
        // If compilation fails, maybe it's just setMaxHttp2Connections?
        // Or setPoolEventLoopSize?
        // Actually, setMaxPoolSize is standard.
        // Let's try setMaxHeaderSize just to see. Or ignore pool size for now (default is 5).
        // 1000 concurrency needs pool size > 5.
        // Let's cast to HttpClientOptions? No.
        // I'll try to set it on variable.
        // options.setMaxPoolSize(1000);

        WebClient client = WebClient.create(vertx, options);

        int totalRequests = 10000;
        int concurrency = 50;

        System.out.println("Starting Load Test: " + totalRequests + " requests, concurrency " + concurrency);

        AtomicInteger completed = new AtomicInteger(0);
        AtomicInteger success = new AtomicInteger(0);
        AtomicInteger started = new AtomicInteger(0);
        List<Long> latencies = Collections.synchronizedList(new ArrayList<>(totalRequests));

        long startTime = System.currentTimeMillis();

        // Start concurrent requests
        for (int i = 0; i < concurrency; i++) {
            sendRequest(client, started, completed, success, latencies, totalRequests);
        }

        // Wait loop
        while (completed.get() < totalRequests) {
            Thread.sleep(100);
        }

        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;

        // 3. Report
        System.out.println("Load Test Completed in " + duration + " ms");
        double rps = (totalRequests * 1000.0 / duration);
        System.out.println("Throughput: " + String.format("%.2f", rps) + " RPS");
        System.out.println("Success Rate: " + (success.get() * 100.0 / totalRequests) + "%");

        synchronized (latencies) {
            Collections.sort(latencies);
            if (!latencies.isEmpty()) {
                System.out.println("p50 Latency: " + latencies.get((int)(latencies.size() * 0.50)) + " ms");
                System.out.println("p95 Latency: " + latencies.get((int)(latencies.size() * 0.95)) + " ms");
                System.out.println("p99 Latency: " + latencies.get((int)(latencies.size() * 0.99)) + " ms");
            }
        }

        client.close();
        vertx.close();
    }

    private static void sendRequest(WebClient client, AtomicInteger started,
                                    AtomicInteger completed, AtomicInteger success,
                                    List<Long> latencies, int total) {
        int id = started.getAndIncrement();
        if (id >= total) {
            return;
        }

        long start = System.nanoTime();
        client.post(8080, "localhost", "/infer")
              .sendBuffer(io.vertx.core.buffer.Buffer.buffer("hello world ai java vector"))
              .onComplete(ar -> {
                  long lat = (System.nanoTime() - start) / 1000000; // ms
                  latencies.add(lat);

                  if (ar.succeeded() && ar.result().statusCode() == 200) {
                      success.incrementAndGet();
                  } else {
                      // System.err.println("Failed: " + (ar.succeeded() ? ar.result().statusCode() : ar.cause()));
                  }

                  int c = completed.incrementAndGet();
                  if (c % 1000 == 0) {
                      System.out.println("Completed: " + c);
                  }

                  // Next request
                  sendRequest(client, started, completed, success, latencies, total);
              });
    }
}
