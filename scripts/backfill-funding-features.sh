#!/usr/bin/env bash
# Backfill the V35 funding columns on feature_store from funding_rate_history.
# Mirrors FundingFeatureSnapshot.compute() in Java:
#   - rate8h     = latest funding event with funding_time <= bar end_time
#   - rate7dAvg  = mean of all events in (end_time - 7d, end_time]
#   - rateZ      = (rate8h - peer_mean) / peer_stddev, where peer = window \ {rate8h}
#                  Requires n >= 4 so the peer stddev has >= 3 samples.
#
# One UPDATE per (symbol, interval). Pre-checks funding_rate_history is not cold.
# Idempotent: re-running overwrites with the same values.

set -euo pipefail

export PGPASSWORD="${PGPASSWORD:-admin}"
PG_HOST="${PG_HOST:-127.0.0.1}"
PG_USER="${PG_USER:-postgres}"
PG_DB="${PG_DB:-trading_db}"

PSQL=( psql -h "$PG_HOST" -U "$PG_USER" -d "$PG_DB" -v ON_ERROR_STOP=1 -q )

run_one() {
    local symbol="$1"
    local interval="$2"
    echo "=== $symbol / $interval ==="
    "${PSQL[@]}" <<SQL
WITH fs_rows AS (
    SELECT id, symbol, end_time
    FROM feature_store
    WHERE symbol = '${symbol}' AND interval = '${interval}'
),
joined AS (
    SELECT fs.id,
           fr.funding_time,
           fr.funding_rate,
           ROW_NUMBER() OVER (PARTITION BY fs.id ORDER BY fr.funding_time DESC) AS rn_desc
    FROM fs_rows fs
    JOIN funding_rate_history fr
      ON fr.symbol = fs.symbol
     AND fr.funding_time <= fs.end_time
     AND fr.funding_time >  fs.end_time - INTERVAL '7 days'
),
agg AS (
    SELECT id,
           MAX(funding_rate) FILTER (WHERE rn_desc = 1)  AS rate8h,
           AVG(funding_rate)                              AS rate7d_avg,
           AVG(funding_rate) FILTER (WHERE rn_desc > 1)  AS peer_mean,
           STDDEV_SAMP(funding_rate) FILTER (WHERE rn_desc > 1) AS peer_std,
           COUNT(*) AS n
    FROM joined
    GROUP BY id
)
UPDATE feature_store fs
SET funding_rate_8h     = agg.rate8h,
    funding_rate_7d_avg = agg.rate7d_avg,
    funding_rate_z      = CASE
        WHEN agg.n >= 4 AND agg.peer_std IS NOT NULL AND agg.peer_std > 0
        THEN ROUND(((agg.rate8h - agg.peer_mean) / agg.peer_std)::numeric, 10)
        ELSE NULL
    END
FROM agg
WHERE fs.id = agg.id;
SQL
}

# Pre-check funding_rate_history coverage.
"${PSQL[@]}" -c "
SELECT symbol, COUNT(*) AS rows, MIN(funding_time), MAX(funding_time)
FROM funding_rate_history GROUP BY symbol ORDER BY symbol;"

for symbol in BTCUSDT ETHUSDT; do
    for interval in 4h 1h 15m 5m; do
        run_one "$symbol" "$interval"
    done
done

echo ""
echo "=== verification ==="
"${PSQL[@]}" -c "
SELECT symbol, interval, COUNT(*) AS total,
       COUNT(funding_rate_8h)    AS with_rate8h,
       COUNT(funding_rate_7d_avg) AS with_avg,
       COUNT(funding_rate_z)     AS with_z
FROM feature_store
WHERE symbol IN ('BTCUSDT','ETHUSDT') AND interval IN ('5m','15m','1h','4h')
GROUP BY symbol, interval ORDER BY symbol, interval;"
