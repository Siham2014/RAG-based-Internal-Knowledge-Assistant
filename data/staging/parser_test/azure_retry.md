# Azure Retry Pattern

The Retry pattern helps an application handle transient failures.

Documentation: [Azure Architecture Center](https://learn.microsoft.com/en-us/azure/architecture/).

## Recommendations

Use the following recommendations:

- Use exponential backoff
- Limit the number of retries
- Log final failures

## Retry configuration

| Attempt | Delay |
|---------|-------|
| 1 | 1 second |
| 2 | 2 seconds |
| 3 | 4 seconds |

## Python example

```python
for attempt in range(3):
    execute_operation()
```

> Retry only operations that are safe to repeat.