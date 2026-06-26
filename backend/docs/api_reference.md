# MindGraph++ API Reference

Base URL: `/api/v1`

## Standard Response Envelope

```json
{
  "status": "success | error",
  "message": "string",
  "data": {},
  "errors": [{"code": "string", "message": "string", "field": "string|null"}],
  "timestamp": "ISO-8601"
}
```

## Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/api/v1/` | Service metadata | No |
| GET | `/api/v1/health` | Health check | No |

## Error Codes

| Code | HTTP | Description |
|------|------|-------------|
| `authentication_error` | 401 | Invalid or missing credentials |
| `authorization_error` | 403 | Insufficient permissions |
| `validation_error` | 422 | Invalid request payload |
| `inference_error` | 503 | Inference pipeline failure |
| `graph_error` | 503 | Knowledge graph operation failure |
| `internal_error` | 500 | Unexpected server error |

## Headers

| Header | Description |
|--------|-------------|
| `X-Request-ID` | Correlation ID for tracing |
| `X-Response-Time-Ms` | Request duration in milliseconds |
| `Authorization` | `Bearer <jwt>` (when auth is enabled) |
