"""``GET /ml/streaming-status`` — derive live streaming worker health from DB.

Queries the gap between MAX(signal_history.ts) and MAX(feature_values.ts)
for every active/shadow signal. The frontend uses this to show a warning
banner when the streaming worker is stalled or has never run.

Three statuses:
* ``ok``       — every signal is within 3× its interval of the latest feature bar
* ``lagging``  — at least one signal has a gap > 3× its interval
* ``offline``  — no signal_history rows exist at all for any active/shadow signal
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends

from .deps import get_agent_name, get_db_conn


router = APIRouter(tags=["ml-monitor"])


_INTERVAL_SECONDS: dict[str, int] = {
    "5m": 300, "15m": 900, "1h": 3_600, "4h": 14_400,
}

_STATUS_SQL = """
SELECT
    sd.name                                AS signal_name,
    mr.interval                            AS interval_name,
    MAX(sh.ts)                             AS last_signal_ts,
    (
        SELECT MAX(fv.ts)
        FROM   feature_values fv
        WHERE  fv.symbol   = mr.symbol
          AND  fv.interval = mr.interval
    )                                      AS latest_feature_ts
FROM   signal_definition sd
JOIN   model_registry mr ON mr.id = sd.model_id
LEFT   JOIN signal_history sh ON sh.signal_id = sd.signal_id
WHERE  sd.status IN ('active', 'shadow')
  AND  mr.symbol   IS NOT NULL
  AND  mr.interval IS NOT NULL
GROUP  BY sd.name, mr.interval, mr.symbol
ORDER  BY sd.name
"""


@router.get("/ml/streaming-status")
async def get_streaming_status(
    conn=Depends(get_db_conn),
    agent: str = Depends(get_agent_name),  # noqa: ARG001
) -> dict[str, Any]:
    rows = [dict(r) for r in await conn.fetch(_STATUS_SQL)]
    now = datetime.now(timezone.utc)

    stalled: list[str] = []
    any_data = False

    for r in rows:
        last_ts: datetime | None = r["last_signal_ts"]
        feat_ts: datetime | None = r["latest_feature_ts"]
        interval_secs = _INTERVAL_SECONDS.get(r["interval_name"] or "", 3_600)

        if last_ts is not None:
            any_data = True
            if feat_ts is not None:
                gap_secs = (feat_ts - last_ts).total_seconds()
                if gap_secs > 3 * interval_secs:
                    stalled.append(r["signal_name"])
            else:
                # No features — not the streaming worker's fault; skip
                pass
        else:
            # Never written — flag if features exist
            if feat_ts is not None:
                stalled.append(r["signal_name"])

    if not rows:
        overall = "ok"
    elif not any_data:
        overall = "offline"
    elif stalled:
        overall = "lagging"
    else:
        overall = "ok"

    return {
        "status": overall,
        "stalledSignals": stalled,
        "checkedAt": now.isoformat(),
        "totalSignals": len(rows),
    }
