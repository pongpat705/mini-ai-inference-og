package com.example.ai;

import com.example.ai.verticles.InferenceVerticle;
import com.example.ai.verticles.PreProcessingVerticle;
import io.vertx.core.AbstractVerticle;
import io.vertx.core.DeploymentOptions;
import io.vertx.core.Promise;
import io.vertx.core.Vertx;
import io.vertx.core.VertxOptions;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.Router;
import io.vertx.ext.web.handler.BodyHandler;

public class MainVerticle extends AbstractVerticle {

    public static final int EMBEDDING_DIM = 128;
    public static final int OUTPUT_DIM = 256;

    @Override
    public void start(Promise<Void> startPromise) {
        // Deploy PreProcessingVerticle
        vertx.deployVerticle(() -> new PreProcessingVerticle(EMBEDDING_DIM), new DeploymentOptions().setInstances(1))
             .onFailure(Throwable::printStackTrace);

        // Deploy InferenceVerticle (Multiple Instances)
        // Input Dim = EMBEDDING_DIM, Output Dim = OUTPUT_DIM
        int cores = Math.max(1, Runtime.getRuntime().availableProcessors());
        vertx.deployVerticle(() -> new InferenceVerticle(EMBEDDING_DIM, OUTPUT_DIM), new DeploymentOptions().setInstances(cores))
             .onFailure(Throwable::printStackTrace);

        // HTTP Server
        Router router = Router.router(vertx);
        router.route().handler(BodyHandler.create());

        // 1. Inference Endpoint
        router.post("/infer").handler(ctx -> {
            String text = ctx.body().asString();
            if (text == null) text = "";

            // Send to PreProcessing
            vertx.eventBus().request(PreProcessingVerticle.ADDRESS_IN, text)
                .onSuccess(reply -> {
                    io.vertx.core.buffer.Buffer res = (io.vertx.core.buffer.Buffer) reply.body();

                    JsonArray jsonArray = new JsonArray();
                    int len = res.length() / 4;
                    for (int i = 0; i < len; i++) {
                        jsonArray.add(res.getFloat(i * 4));
                    }

                    var resp = ctx.response();
                    resp.putHeader("content-type", "application/json");
                    resp.end(jsonArray.encode());
                })
                .onFailure(err -> {
                    ctx.response().setStatusCode(500).end(err.getMessage());
                });
        });

        // 2. Training/Control Endpoint
        router.post("/train").handler(ctx -> {
            JsonObject body = null;
            try {
                body = ctx.body().asJsonObject();
            } catch (Exception e) {
                // Not a JSON
            }

            if (body == null) {
                String text = ctx.body().asString();
                if (text != null && !text.isEmpty()) {
                    // Try to treat as batch training text if not JSON
                    body = new JsonObject().put("type", "batch").put("data", text);
                } else {
                    ctx.response().setStatusCode(400).end("Missing body");
                    return;
                }
            }

            vertx.eventBus().request(PreProcessingVerticle.ADDRESS_CONTROL, body)
                .onSuccess(reply -> {
                    ctx.response()
                       .putHeader("content-type", "application/json")
                       .end(reply.body().toString());
                })
                .onFailure(err -> {
                    int code = 500;
                    if (err.getMessage().contains("400")) code = 400;
                    if (err.getMessage().contains("404")) code = 404;
                    ctx.response().setStatusCode(code).end(err.getMessage());
                });
        });

        vertx.createHttpServer().requestHandler(router).listen(8080)
            .onSuccess(server -> {
                System.out.println("HTTP server started on port 8080");
                startPromise.complete();
            })
            .onFailure(startPromise::fail);
    }

    public static void main(String[] args) {
        VertxOptions options = new VertxOptions();
        Vertx.clusteredVertx(options).onComplete(res -> {
            if (res.succeeded()) {
                Vertx vertx = res.result();
                vertx.deployVerticle(new MainVerticle());
                System.out.println("Cluster started!");
            } else {
                res.cause().printStackTrace();
            }
        });
    }
}
