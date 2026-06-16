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
    iteration_id_refs    UUID[],
    related_journal_ids  UUID[],
    created_time    TIMESTAMP    NOT NULL DEFAULT NOW(),
    created_by      VARCHAR(100),
    updated_time    TIMESTAMP,
    updated_by      VARCHAR(100),
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
    active_backtest_run_id UUID,
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

-- Real per-day strategy equity series the JVM records (one row per
-- (backtest_run_id, equity_date)). The hedging gate reads total_equity to
-- measure strat metrics from the TRUE mark-to-market curve instead of
-- reconstructing from sparse trade-exit P&L. FK-free trimmed copy of the prod
-- table — only the columns repo/backtest_equity.py:fetch_equity_points SELECTs
-- (plus the run/date key). Types match the Flyway baseline.
CREATE TABLE IF NOT EXISTS backtest_equity_point (
    backtest_run_id   UUID           NOT NULL,
    account_id        UUID,
    equity_date       DATE           NOT NULL,
    cash_balance      NUMERIC(24,8),
    asset_value       NUMERIC(24,8),
    total_equity      NUMERIC(24,8)  NOT NULL,
    drawdown_percent  NUMERIC(24,8),
    daily_return_pct  NUMERIC(24,8),
    open_positions    INTEGER,
    CONSTRAINT uq_backtest_equity_point UNIQUE (backtest_run_id, equity_date)
);

-- macro_raw series repo (funding rates + Deribit DVOL) reads this. Minimal copy
-- of the prod table (blackheart-trading-engine V66__add_ml_sentiment_schema.sql):
-- a simple id BIGSERIAL PK replaces prod's partitioning + composite PK
-- (irrelevant to reads). NOT NULL cols carried by prod (source_uri/content_hash/
-- ingestion_time) get defaults so the read tests stay decoupled from ingest shape.
CREATE TABLE IF NOT EXISTS macro_raw (
    id             BIGSERIAL       PRIMARY KEY,
    source         VARCHAR(80)     NOT NULL,
    source_uri     VARCHAR(500)    NOT NULL DEFAULT '',
    symbol         VARCHAR(20),
    series_id      VARCHAR(120)    NOT NULL,
    event_time     TIMESTAMPTZ     NOT NULL,
    ingestion_time TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    value          NUMERIC(28,10),
    content_hash   VARCHAR(64)     NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS research_iteration_log (
    iteration_id        UUID        NOT NULL DEFAULT gen_random_uuid(),
    strategy_code       VARCHAR(60) NOT NULL,
    iteration_number    INTEGER     NOT NULL,
    backtest_run_id     UUID,
    params_snapshot     JSONB       NOT NULL DEFAULT '{}',
    metrics_snapshot    JSONB       NOT NULL DEFAULT '{}',
    verdict             VARCHAR(20) NOT NULL,
    statistical_verdict VARCHAR(40),
    created_time        TIMESTAMP   NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_research_iteration_log PRIMARY KEY (iteration_id),
    CONSTRAINT uq_research_iteration_strategy_n UNIQUE (strategy_code, iteration_number)
);

-- Signal Pool / Combination Book (Flyway V147). The strategy pool (Phase 0)
-- AND the signal-level combination book (Phase 3) COEXIST in this one table;
-- they are discriminated by ``admission_metrics->>'kind'`` (a JSONB key, NOT a
-- physical column) so the two books' reads stay isolated WITHOUT any schema
-- migration. Strategy-pool rows are 'signal_pool' (or untagged = legacy);
-- combination rows are 'signal_combination'. Trimmed FK-free copy of the prod
-- table — only the columns the orchestrator reads/writes; types + the
-- active-unique index match V147 exactly so this schema can't mask a prod-only
-- column/index drift again.
CREATE TABLE IF NOT EXISTS signal_pool (
    pool_id            UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    iteration_id       UUID         NOT NULL,
    strategy_code      VARCHAR(60)  NOT NULL,
    symbol             VARCHAR(30)  NOT NULL,
    interval_name      VARCHAR(20)  NOT NULL,
    admitted_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    admission_metrics  JSONB        NOT NULL DEFAULT '{}'::jsonb,
    pool_weight        NUMERIC(7,5),
    weight_source      VARCHAR(24),
    weight_updated_at  TIMESTAMPTZ,
    status             VARCHAR(16)  NOT NULL DEFAULT 'active',
    evicted_at         TIMESTAMPTZ,
    evicted_reason     TEXT,
    created_time       TIMESTAMP    NOT NULL DEFAULT now(),
    created_by         VARCHAR(150),
    updated_time       TIMESTAMP    NOT NULL DEFAULT now(),
    updated_by         VARCHAR(150)
);

-- One active membership per (strategy_code, symbol, interval) surface — matches
-- prod V147. A combination member and a strategy member CANNOT both be active on
-- the identical surface; in practice combination surfaces are distinct
-- strategy_codes, so this does not collide.
CREATE UNIQUE INDEX IF NOT EXISTS signal_pool_active_unique
    ON signal_pool (strategy_code, symbol, interval_name)
    WHERE status = 'active';


-- /signal-screen integration surface: a TRIMMED feature_store (one numeric
-- signal column is enough for information_schema resolution + the exact-ts
-- join) and the feature_values series store (V70). Types match Flyway.
CREATE TABLE IF NOT EXISTS feature_store (
    id             BIGSERIAL     PRIMARY KEY,
    id_market_data BIGINT        NOT NULL DEFAULT 0,
    symbol         VARCHAR(20)   NOT NULL,
    interval       VARCHAR(10)   NOT NULL,
    start_time     TIMESTAMP     NOT NULL,
    end_time       TIMESTAMP     NOT NULL,
    price          NUMERIC(24,8) NOT NULL,
    rsi            NUMERIC(24,8)
);

CREATE TABLE IF NOT EXISTS feature_values (
    feature_name VARCHAR(120)  NOT NULL,
    version      INTEGER       NOT NULL DEFAULT 1,
    symbol       VARCHAR(20)   NOT NULL DEFAULT '',
    interval     VARCHAR(10)   NOT NULL DEFAULT '',
    ts           TIMESTAMP     NOT NULL,
    value        DOUBLE PRECISION,
    value_text   TEXT,
    compute_run_id UUID,
    PRIMARY KEY (feature_name, version, symbol, interval, ts)
);

-- ── Strategy Research Registry (Flyway V182) + the tables its live-metrics
--    join reads. backtest_run / walk_forward_run are trimmed FK-free copies
--    (only the columns the join touches); account_strategy gains the
--    live-status columns the join needs (added idempotently so the trimmed
--    CREATE above other tests rely on is untouched). ─────────────────────────
ALTER TABLE account_strategy ADD COLUMN IF NOT EXISTS symbol        VARCHAR(30);
ALTER TABLE account_strategy ADD COLUMN IF NOT EXISTS interval_name VARCHAR(20);
ALTER TABLE account_strategy ADD COLUMN IF NOT EXISTS enabled       BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE account_strategy ADD COLUMN IF NOT EXISTS simulated     BOOLEAN NOT NULL DEFAULT TRUE;

CREATE TABLE IF NOT EXISTS backtest_run (
    backtest_run_id  UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_code    VARCHAR(60),
    asset            VARCHAR(30),
    interval_name    VARCHAR(20),
    status           VARCHAR(20),
    start_time       TIMESTAMP,
    end_time         TIMESTAMP
);

CREATE TABLE IF NOT EXISTS walk_forward_run (
    walk_forward_id   UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_code     VARCHAR(60),
    instrument        VARCHAR(30),
    interval_name     VARCHAR(20),
    stability_verdict VARCHAR(40),
    created_time      TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS strategy_research_registry (
    registry_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug                     TEXT NOT NULL UNIQUE,
    rank                     INT,
    promise_tier             TEXT NOT NULL,
    display_name             TEXT NOT NULL,
    signal_family            TEXT,
    strategy_code            TEXT,
    symbol                   TEXT,
    interval_name            TEXT,
    verdict_tag              TEXT NOT NULL,
    lifecycle_status         TEXT NOT NULL,
    thesis                   TEXT NOT NULL,
    detail                   TEXT,
    evidence_iteration_id    UUID,
    evidence_walk_forward_id UUID,
    evidence_backtest_run_id UUID,
    journal_id               UUID,
    memory_ref               TEXT,
    is_offline_lead          BOOLEAN NOT NULL DEFAULT FALSE,
    archived                 BOOLEAN NOT NULL DEFAULT FALSE,
    auto_managed             BOOLEAN NOT NULL DEFAULT FALSE,
    created_time             TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_time             TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by               TEXT
);
