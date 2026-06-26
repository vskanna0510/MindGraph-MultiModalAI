## Summary

<!-- What does this PR change? -->

## Motivation

<!-- Why is this change needed? Link issues if applicable. -->

## Architecture Impact

<!-- Which layers/services are affected? Does this preserve Controller → Service → Repository? -->

## Screenshots (if UI)

<!-- Before/after for Flutter changes -->

## Testing Evidence

<!-- Commands run, coverage delta, test names -->

```
make lint
make test
```

## Performance Impact

<!-- Latency, memory, startup — or "None" -->

## Security Impact

<!-- Auth, PII, secrets — or "None" -->

## Documentation Updated

- [ ] README / module docs
- [ ] OpenAPI / api_reference.md
- [ ] CHANGELOG (if user-facing)

## Checklist (Engineering Constitution)

- [ ] Compiles successfully
- [ ] `make lint` passes
- [ ] `make test` passes
- [ ] No hardcoded secrets, URLs, or credentials
- [ ] No business logic in Flutter widgets
- [ ] No SQL/Cypher in controllers or UI
- [ ] No `print()` in application code
- [ ] No TODO/FIXME/placeholder/fake data
- [ ] Type-safe (mypy / dart analyze)
- [ ] Architecture layers preserved
- [ ] Documentation updated
- [ ] Clinical/ethics copy compliant (if user-facing)
