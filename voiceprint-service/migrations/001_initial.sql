CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(64) NOT NULL,
    display_name VARCHAR(128) NOT NULL,
    preferred_name VARCHAR(128),
    email VARCHAR(320),
    phone VARCHAR(32),
    locale VARCHAR(16) NOT NULL DEFAULT 'zh-CN',
    timezone VARCHAR(64) NOT NULL DEFAULT 'Asia/Shanghai',
    notes TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT users_username_not_blank CHECK (btrim(username) <> ''),
    CONSTRAINT users_display_name_not_blank CHECK (btrim(display_name) <> '')
);

CREATE UNIQUE INDEX IF NOT EXISTS users_username_unique_ci
    ON users (lower(username));
CREATE UNIQUE INDEX IF NOT EXISTS users_email_unique_ci
    ON users (lower(email)) WHERE email IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS users_phone_unique
    ON users (phone) WHERE phone IS NOT NULL;

CREATE TABLE IF NOT EXISTS voiceprints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    label VARCHAR(128),
    model_id VARCHAR(255) NOT NULL,
    model_revision VARCHAR(64) NOT NULL,
    embedding_dimension SMALLINT NOT NULL,
    embedding REAL[] NOT NULL,
    duration_ms INTEGER NOT NULL,
    sample_rate INTEGER NOT NULL,
    audio_sha256 CHAR(64) NOT NULL,
    quality JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT voiceprints_embedding_dimension_positive CHECK (embedding_dimension > 0),
    CONSTRAINT voiceprints_embedding_size_matches CHECK (
        array_ndims(embedding) = 1
        AND array_length(embedding, 1) = embedding_dimension
    ),
    CONSTRAINT voiceprints_duration_positive CHECK (duration_ms > 0)
);

CREATE INDEX IF NOT EXISTS voiceprints_active_user_idx
    ON voiceprints (user_id) WHERE is_active;
CREATE INDEX IF NOT EXISTS voiceprints_model_idx
    ON voiceprints (model_id, model_revision) WHERE is_active;

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS users_set_updated_at ON users;
CREATE TRIGGER users_set_updated_at
BEFORE UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION set_updated_at();
