# Immediate Stop regression suite

Run the focused Stop/session-transition checks with:

```bash
pytest tests/stopping/
```

The suite verifies the production `LiveSessionBoundary`, generation invalidation,
queue clearing, model reuse, stale-result suppression, Stop → language transitions,
rapid Stop/Start cycles, and the absence of joins on the UI Stop path. `STOP-010`
collects 20 real monotonic measurements and gates P95 Stop-to-ready overhead at
500 ms; ongoing stale CTranslate2 inference time is deliberately excluded.
