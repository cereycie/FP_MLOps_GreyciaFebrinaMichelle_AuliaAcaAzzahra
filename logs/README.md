# README Logs

`predictions.jsonl` is not committed to Git, it is created automatically the first time `/risk-score` is called, and grows by one line per prediction after that.

## Format

One JSON object per line (JSON Lines), so it can be read either with a text editor or with `pandas.read_json("logs/predictions.jsonl", lines=True)`.

## Fields

| Field | Meaning |
| :-- | :-- |
| `timestamp` | When the request was received, UTC |
| `lat`, `lon` | Requested coordinates, as sent, before rounding to a grid cell |
| `requested_datetime` | The `datetime` query parameter, as sent |
| `nearby_crime_count` | Computed feature, how many historical events fell within 1200 meters |
| `risk_score`, `level` | What the API returned |
| `model_version` | Which champion version answered this request |
| `used_model` | `false` if the zero-crime shortcut answered instead of the model |
| `recent_incident_override` | Only present when `used_model` is `true`, whether the Very High override fired |
| `latency_ms` | Time spent inside the endpoint handler, excludes network time |

## Monitoring

`GET /metrics` on the running API reads this file plus `models/registry.json` and returns a live summary, current model version and its holdout metrics, the full promotion history across checkpoints, and recent prediction activity (average score, average latency, level distribution, how often the zero-crime shortcut fired).
