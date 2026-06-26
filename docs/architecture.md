# System Architecture

MindGraph++ follows **Microservices + Clean Architecture + Event-Driven** design.

## Layers

1. **Flutter Application** — offline-first mobile client
2. **API Gateway (FastAPI)** — routing, auth, rate limiting
3. **Microservices** — auth, feature extraction, inference, KG, analytics, notifications
4. **Data Layer** — PostgreSQL, Neo4j, Redis, object storage

## Data Flow

```
User → Flutter → On-Device Extraction → Encrypt → HTTPS → FastAPI
  → Inference → Neo4j KG → GNN → Explainability → Response → Dashboard
```

See root `README.md` for the architecture diagram.

## Principles

- Privacy first, security first, offline first
- No raw media upload by default
- Embeddings only over the network
- Every prediction includes explainability metadata
