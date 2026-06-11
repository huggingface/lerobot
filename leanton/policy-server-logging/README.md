# policy-server-logging

**Target:** `src/lerobot/async_inference/policy_server.py`
**Status:** `active`
**GitHub:** Not filed (minor)

## What

Adds traceback logging to the policy server's error handling.

## Why

Policy server errors were silently swallowed — no stack trace in logs, making debugging impossible.

## Validate

**Agent:**
```bash
grep -q "traceback" ~/lerobot/src/lerobot/async_inference/policy_server.py && echo "Traceback logging ✅" || echo "MISSING"
```
