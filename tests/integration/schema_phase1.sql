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
    params_snapshot     JSONB       NOT NULL DEFAULT '{}',
    metrics_snapshot    JSONB       NOT NULL DEFAULT '{}',
    verdict             VARCHAR(20) NOT NULL,
    statistical_verdict VARCHAR(40),
    created_time        TIMESTAMP   NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_research_iteration_log PRIMARY KEY (iteration_id),
    CONSTRAINT uq_research_iteration_strategy_n UNIQUE (strategy_code, iteration_number)
);

-- Signal Pool / Combination Book (Flyway V147 + Phase-3 kind/track columns).
-- The strategy pool (kind='signal_pool', Phase 0) AND the signal-level
-- combination book (kind='signal_combination', Phase 3) COEXIST in this one
-- table; the kind discriminator (+ track) keeps their reads isolated without a
-- separate table. Trimmed FK-free copy of the prod table — only the columns the
-- orchestrator reads/writes; types match V147. The kind/track columns are the
-- Phase-3 additions (a Flyway migration carries them in prod).
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
    -- Phase-3 coexistence discriminator: 'signal_pool' (strategy book) or
    -- 'signal_combination' (residual combination book). Default keeps every
    -- existing strategy-pool write a 'signal_pool' row with no code change.
    kind               VARCHAR(24)  NOT NULL DEFAULT 'signal_pool',
    -- Dual-track tag for combination members ('trading' | 'hedging'); NULL for
    -- the (untracked) strategy pool.
    track              VARCHAR(20),
    evicted_at         TIMESTAMPTZ,
    evicted_reason     TEXT,
    created_time       TIMESTAMP    NOT NULL DEFAULT now(),
    created_by         VARCHAR(150),
    updated_time       TIMESTAMP    NOT NULL DEFAULT now(),
    updated_by         VARCHAR(150)
);

-- One active membership per (kind, surface) — so the same surface can hold both
-- a strategy-pool row and a combination row without colliding.
CREATE UNIQUE INDEX IF NOT EXISTS signal_pool_active_unique
    ON signal_pool (kind, strategy_code, symbol, interval_name)
    WHERE status = 'active';
