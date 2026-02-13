package com.example.ai;

import com.example.ai.verticles.InferenceVerticle;
import com.example.ai.verticles.PreProcessingVerticle;
import io.vertx.core.AbstractVerticle;
import io.vertx.core.DeploymentOptions;
import io.vertx.core.Promise;
import io.vertx.core.Vertx;
import io.vertx.core.VertxOptions;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.Router;
import io.vertx.ext.web.handler.BodyHandler;

public class MainVerticle extends AbstractVerticle {

    @Override
    public void start(Promise<Void> startPromise) {
        // Deploy PreProcessingVerticle (Embedding Dim = 64)
        vertx.deployVerticle(() -> new PreProcessingVerticle(64), new DeploymentOptions().setInstances(1))
             .onFailure(Throwable::printStackTrace);

        // Deploy InferenceVerticle (Multiple Instances)
        // Input Dim = 64 (Dense Embedding), Output Dim = 128
        int cores = Math.max(1, Runtime.getRuntime().availableProcessors());
        vertx.deployVerticle(() -> new InferenceVerticle(64, 128), new DeploymentOptions().setInstances(cores))
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

                    StringBuilder sb = new StringBuilder("[");
                    int len = res.length() / 4;
                    int show = Math.min(len, 10);
                    for(int i=0; i<show; i++) {
                        sb.append(res.getFloat(i*4));
                        if(i < show-1) sb.append(", ");
                    }
                    if(len > show) sb.append(", ...");
                    sb.append("]");

                    var resp = ctx.response();
                    resp.putHeader("content-type", "application/json");
                    resp.end(sb.toString());
                })
                .onFailure(err -> {
                    ctx.response().setStatusCode(500).end(err.getMessage());
                });
        });

        // 2. Training/Control Endpoint
        router.post("/train").handler(ctx -> {
            JsonObject body = ctx.body().asJsonObject();
            if (body == null) {
                ctx.response().setStatusCode(400).end("Missing JSON body");
                return;
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
