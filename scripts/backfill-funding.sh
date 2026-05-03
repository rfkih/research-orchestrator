#!/usr/bin/env bash
# Backfill funding_rate_history from Binance fapi for BTCUSDT + ETHUSDT.
# Idempotent — ON CONFLICT DO NOTHING on the (symbol, funding_time) PK.
# Replaces the JVM endpoint at /api/v1/historical/backfill-funding when the
# deployed research JAR is stale.
#
# Usage: bash scripts/backfill-funding.sh [SYMBOL ...]
#        defaults to BTCUSDT ETHUSDT

set -euo pipefail

export PGPASSWORD="${PGPASSWORD:-admin}"
PG_HOST="${PG_HOST:-127.0.0.1}"
PG_USER="${PG_USER:-postgres}"
PG_DB="${PG_DB:-trading_db}"

START_MS=1704067200000   # 2024-01-01T00:00:00Z
NOW_MS=$(date -u +%s000)
SYMBOLS=("$@")
[ ${#SYMBOLS[@]} -eq 0 ] && SYMBOLS=(BTCUSDT ETHUSDT)

backfill_symbol() {
    local symbol="$1"
    local cursor=$START_MS
    local total_rows=0
    local pages=0
    echo "[${symbol}] starting from $(date -u -d @$((cursor/1000)) +%Y-%m-%d)"

    while :; do
        local resp
        resp=$(curl -fsS "https://fapi.binance.com/fapi/v1/fundingRate?symbol=${symbol}&startTime=${cursor}&limit=1000")
        local count
        count=$(echo "$resp" | jq 'length')
        if [ "$count" -eq 0 ]; then
            echo "[${symbol}] empty page — done"
            break
        fi

        # Build a single multi-row VALUES clause from the JSON array.
        local values
        values=$(echo "$resp" | jq -r --arg sym "$symbol" '
            map([
                $sym,
                (.fundingTime / 1000 | strftime("%Y-%m-%d %H:%M:%S")),
                .fundingRate,
                (.markPrice // "")
            ] | @tsv) | .[]
        ' | awk -F'\t' '{
            mp = ($4 == "") ? "NULL" : "'\''" $4 "'\''"
            printf "%s('\''%s'\'', '\''%s'\'', %s, %s)", (NR>1 ? "," : ""), $1, $2, $3, mp
        }')

        psql -h "$PG_HOST" -U "$PG_USER" -d "$PG_DB" -q -v ON_ERROR_STOP=1 <<SQL
INSERT INTO funding_rate_history (symbol, funding_time, funding_rate, mark_price)
VALUES ${values}
ON CONFLICT (symbol, funding_time) DO NOTHING;
SQL

        total_rows=$((total_rows + count))
        pages=$((pages + 1))
        # Advance cursor to last row's fundingTime + 1ms; break if we're caught up.
        local last_ms
        last_ms=$(echo "$resp" | jq '.[-1].fundingTime')
        echo "[${symbol}] page ${pages}: ${count} rows, last=$(date -u -d @$((last_ms/1000)) +%Y-%m-%dT%H:%M:%SZ)"
        if [ "$last_ms" -ge "$NOW_MS" ] || [ "$count" -lt 1000 ]; then
            break
        fi
        cursor=$((last_ms + 1))
    done

    echo "[${symbol}] done: ${total_rows} rows fetched across ${pages} pages"
}

for sym in "${SYMBOLS[@]}"; do
    backfill_symbol "$sym"
done

echo ""
echo "=== verification ==="
psql -h "$PG_HOST" -U "$PG_USER" -d "$PG_DB" -c "
SELECT symbol, COUNT(*) AS rows, MIN(funding_time) AS first, MAX(funding_time) AS last
FROM funding_rate_history
GROUP BY symbol
ORDER BY symbol;"
