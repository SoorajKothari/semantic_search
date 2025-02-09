# Semantic Search with Vector DBs: LLMs + Qdrant in Rust  

This project demonstrates how to use **Ollama LLM models** with **Qdrant DB** to build a **semantic search system** in Rust. Instead of traditional keyword-based search, we use **vector embeddings** to find **similar texts efficiently**.  

## ✨ Features  
- Generate text embeddings using **Ollama (Llama models)**  
- Store and retrieve vectors using **Qdrant** 
- Perform **fast semantic search** using **vector similarity**  
- Fully **self-hosted** and runs locally! 🚀  

## 📦 Installation  

### 1️⃣ Run Qdrant (Docker)  
```sh
docker run -p 6333:6333 -p 6334:6334 \
    -e QDRANT__SERVICE__GRPC_PORT="6334" \
    qdrant/qdrant


### Run
```sh
Cargo run
