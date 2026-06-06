-- Minimal subset of the Flyway V1 baseline for Phase 1 track-isolation tests.
-- Source of truth: blackheart-trading-engine/src/main/resources/db/flyway/V1__baseline.sql
CREATE TABLE IF NOT EXISTS research_journal (
    journal_id      UUID         NOT NULL DEFAULT gen_random_uuid(),
    entry_type      VARCHAR(40)  NOT NULL,
    strategy_code   VARCHAR(60),
    interval_name   VARCHAR(20),
    instrument      VARCHAR(30),
    title           VARCHAR(300) NOT NULL,
    content         TEXT         NOT NULL,
    structured_data JSONB        NOT NULL DEFAULT '{}',
    status          VARCHAR(20)  NOT NULL DEFAULT 'ACTIVE',
    created_time    TIMESTAMP    NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_research_journal PRIMARY KEY (journal_id)
);

CREATE TABLE IF NOT EXISTS research_queue (
    queue_id        UUID        NOT NULL DEFAULT gen_random_uuid(),
    priority        INTEGER     NOT NULL DEFAULT 100,
    strategy_code   VARCHAR(60) NOT NULL,
    interval_name   VARCHAR(20) NOT NULL,
    instrument      VARCHAR(30) NOT NULL DEFAULT 'BTCUSDT',
    sweep_config    JSONB       NOT NULL,
    hypothesis      TEXT,
    status          VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    iteration_number INTEGER    NOT NULL DEFAULT 0,
    iter_budget     INTEGER     NOT NULL DEFAULT 5,
    early_stop_on_no_edge BOOLEAN NOT NULL DEFAULT TRUE,
    require_walk_forward  BOOLEAN NOT NULL DEFAULT TRUE,
    last_iteration_id     UUID,
    last_run_id           UUID,
    final_verdict         VARCHAR(40),
    walk_forward_id       UUID,
    created_time    TIMESTAMP   NOT NULL DEFAULT NOW(),
    created_by      VARCHAR(150),
    started_at      TIMESTAMP,
    completed_at    TIMESTAMP,
    notes           TEXT,
    CONSTRAINT pk_research_queue PRIMARY KEY (queue_id)
);

-- Re-discovery gate reads hypothesis_audit (and joins research_queue for the
-- per-track scope). Trimmed copy of the V1 baseline (FK-free for the harness).
CREATE TABLE IF NOT EXISTS hypothesis_audit (
    audit_id            UUID        NOT NULL DEFAULT gen_random_uuid(),
    strategy_code       VARCHAR(60) NOT NULL,
    symbol              VARCHAR(30),
    interval_name       VARCHAR(20),
    axis_set_hash       CHAR(64)    NOT NULL,
    param_combo_hash    CHAR(64)    NOT NULL,
    params_snapshot     JSONB       NOT NULL,
    queue_id            UUID,
    iteration_id        UUID,
    statistical_verdict VARCHAR(40),
    decision_verdict    VARCHAR(40),
    created_by          VARCHAR(80) NOT NULL,
    created_time        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_time        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_hypothesis_audit PRIMARY KEY (audit_id)
);

-- POST /queue's pre-insert existence check (find_account_strategy_id). FK-free
-- trimmed copy of the V1 baseline — only the columns the lookup reads.
CREATE TABLE IF NOT EXISTS account_strategy (
    account_strategy_id      UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id               UUID,
    strategy_code            VARCHAR(100) NOT NULL,
    is_deleted               BOOLEAN      NOT NULL DEFAULT FALSE
);

-- Buy-hold benchmark reads market_data closes (hedging gate). Trimmed copy of
-- the V1 baseline; column types match the Flyway baseline CREATE TABLE.
CREATE TABLE IF NOT EXISTS market_data (
    id                     BIGSERIAL      PRIMARY KEY,
    symbol                 VARCHAR(10)    NOT NULL,
    interval               VARCHAR(5)     NOT NULL,
    start_time             TIMESTAMP      NOT NULL,
    end_time               TIMESTAMP      NOT NULL,
    open_price             NUMERIC(24,12) NOT NULL,
    close_price            NUMERIC(24,12) NOT NULL,
    high_price             NUMERIC(24,12) NOT NULL,
    low_price              NUMERIC(24,12) NOT NULL,
    volume                 NUMERIC(24,12) NOT NULL,
    trade_count            BIGINT         NOT NULL,
    quote_asset_volume     NUMERIC(24,12) NOT NULL,
    taker_buy_base_volume  NUMERIC(24,12) NOT NULL,
    taker_buy_quote_volume NUMERIC(24,12) NOT NULL,
    created_time           TIMESTAMP      NOT NULL DEFAULT NOW()
);

-- Strategy daily-return series for the hedging gate reads backtest_trade. FK-free
-- trimmed copy of the V1 baseline — only the columns repo/trades.py:fetch_trades
-- SELECTs (plus the backtest_run_id filter key). Types match the Flyway baseline.
CREATE TABLE IF NOT EXISTS backtest_trade (
    backtest_trade_id              UUID          PRIMARY KEY,
    backtest_run_id                UUID          NOT NULL,
    side                           VARCHAR(10)   NOT NULL,
    status                         VARCHAR(30)   NOT NULL,
    exit_reason                    VARCHAR(100),
    realized_pnl_amount            NUMERIC(24,8),
    realized_r_multiple            NUMERIC(24,8),
    notional_size                  NUMERIC(24,8),
    max_favorable_excursion_r      NUMERIC(24,8),
    max_adverse_excursion_r        NUMERIC(24,8),
    bars_held                      INTEGER,
    entry_time                     TIMESTAMP     NOT NULL,
    exit_time                      TIMESTAMP,
    entry_adx                      NUMERIC(24,8),
    entry_rsi                      NUMERIC(24,8),
    entry_close_location_value     NUMERIC(24,8),
    entry_relative_volume20        NUMERIC(24,8),
    entry_trend_regime             VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS research_iteration_log (
    iteration_id        UUID        NOT NULL DEFAULT gen_random_uuid(),
    strategy_code       VARCHAR(60) NOT NULL,
    iteration_number    INTEGER     NOT NULL,
    params_snapshot     JSONB       NOT NULL DEFAULT '{}',
    metrics_snapshot    JSONB       NOT NULL DEFAULT '{}',
    verdict             VARCHAR(20) NOT NULL,
    statistical_verdict VARCHAR(40),
    created_time        TIMESTAMP   NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_research_iteration_log PRIMARY KEY (iteration_id),
    CONSTRAINT uq_research_iteration_strategy_n UNIQUE (strategy_code, iteration_number)
);
