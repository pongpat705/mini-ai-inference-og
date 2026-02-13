# Mini-AI Inference Loop with Java 21 & Vert.x 5

A high-performance, distributed AI inference pipeline demonstrating the power of **Java 21 Vector API** (SIMD) for matrix operations and **Vert.x 5** for reactive, non-blocking concurrency.

This project implements a complete "Mini-AI" system that takes text input, vectorizes it using **Dense Embeddings**, and performs a dense layer inference ($Y = WX + B$) using hardware acceleration. It also features a real-time **Training Channel** to update word embeddings on the fly.

---

## 🧠 The Brain: Internal AI Architecture
The core computational unit is the `LinearLayer` located in `src/main/java/com/example/ai/LinearLayer.java`.

- **Vector API (SIMD)**: Uses `jdk.incubator.vector.FloatVector` to perform dot products. This allows the CPU to process multiple floating-point operations in a single clock cycle (e.g., AVX-512), resulting in significant speedups (~8x faster than standard loops).
- **Off-Heap Memory**: Weights and biases are stored in off-heap memory using `java.lang.foreign.MemorySegment` and `Arena`. This reduces Garbage Collection (GC) overhead for large models.
- **Activation Function**: A vectorized ReLU (Rectified Linear Unit) activation is applied.

## 🫀 The Body: Distributed Pipeline
The system is built as a reactive pipeline using **Eclipse Vert.x 5**:

1.  **MainVerticle**: The entry point that deploys other verticles and starts the HTTP server. It configures Hazelcast for clustering.
2.  **PreProcessingVerticle**:
    -   Converts raw text input into a **Dense Vector** (64-dim) by summing word embeddings.
    -   Maintains a mutable `Map<String, float[]>` of word embeddings.
    -   Exposes a **Training Channel** to update embeddings or "pull" vectors closer (e.g., making "love" and "like" semantically similar).
3.  **InferenceVerticle**: Receives the dense vector and performs the matrix multiplication using the `LinearLayer`. It implements **backpressure** handling to protect the system from overload.

The components communicate asynchronously via the **Vert.x EventBus**.

## 🏋️ Training & Control
You can update the AI's "knowledge" (embeddings) at runtime via the `/train` endpoint.

**Example: Pulling words closer**
To make the embedding for "love" closer to "like":
```bash
curl -X POST http://localhost:8080/train \
     -H "Content-Type: application/json" \
     -d '{"type": "pull", "word1": "love", "word2": "like", "rate": 0.5}'
```

**Example: Updating a word vector**
```bash
curl -X POST http://localhost:8080/train \
     -H "Content-Type: application/json" \
     -d '{"type": "update", "word": "java", "vector": [0.1, 0.2, ...]}'
```

## 📊 The Evidence: Benchmarks
The project includes a JMH (Java Microbenchmark Harness) suite to verify performance gains.

### Benchmark Results (Example)
| Implementation | Mode | Score (ops/s) | Speedup |
| :--- | :--- | :--- | :--- |
| **Scalar Loop (Baseline)** | Throughput | ~5,050 | 1x |
| **Vector API (SIMD)** | Throughput | ~40,270 | **~8x** |

*(Results may vary based on hardware support for AVX/SIMD instructions)*

---

## 🚀 Getting Started

### Prerequisites
- **Java 21** (Required for Vector API and Foreign Function & Memory API).
- **Maven** 3.8+.

### 1. Build the Project
Since this project uses Preview Features (Vector API, FFM), you must enable them during compilation.

```bash
mvn clean package
```

### 2. Run the Server
The application runs a Vert.x HTTP server on port `8080`.

```bash
# Run the fat jar with preview features enabled
java --enable-preview --add-modules jdk.incubator.vector -jar target/mini-ai-inference-fat.jar
```

**Test with curl:**
```bash
curl -X POST http://localhost:8080/infer -d "hello world ai java vector"
```

### 3. Run Micro-Benchmarks (JMH)
To see the Vector API performance in action:

```bash
mvn integration-test
mvn exec:exec@run-benchmarks
```

### 4. Run Verification Tests
To verify the training channel logic:

```bash
# Compile tests
mvn test-compile

# Run EmbeddingTrainingTest class
java --enable-preview --add-modules jdk.incubator.vector \
     -cp target/classes:target/test-classes:$(mvn dependency:build-classpath -Dmdep.outputFile=/dev/stdout -q) \
     com.example.ai.EmbeddingTrainingTest
```

---

## Project Structure
```
src/
├── main/java/com/example/ai/
│   ├── ActivationFunction.java   # Interface for ReLU/Sigmoid
│   ├── LinearLayer.java          # Core Matrix Math (Vector API)
│   ├── MainVerticle.java         # Entry point & HTTP Server
│   └── verticles/
│       ├── InferenceVerticle.java    # AI Logic Worker
│       └── PreProcessingVerticle.java # Dense Embeddings & Training
└── test/java/com/example/ai/
    ├── EmbeddingTrainingTest.java # Verification of Training Channel
    ├── LoadTest.java             # System Load Test (Vert.x WebClient)
    └── benchmark/
        └── VectorBenchmark.java  # JMH Microbenchmark
```
