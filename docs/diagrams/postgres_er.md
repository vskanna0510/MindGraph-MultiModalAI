# PostgreSQL ER Diagram

> Generated when authentication and user modules are implemented.

## Planned Entities

- `users` — accounts and profile metadata
- `audit_events` — security audit trail (implemented)
- `consent_records` — per-modality consent
- `sessions` — wellness check-in metadata
- `notification_preferences` — user settings

## Current

```mermaid
erDiagram
    AUDIT_EVENTS {
        uuid id PK
        string request_id
        string event_type
        string action
        string actor_id
        string ip_address
        string device
        string status
        timestamp created_at
    }
```
