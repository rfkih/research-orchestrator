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
    created_time    TIMESTAMP   NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_research_queue PRIMARY KEY (queue_id)
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
