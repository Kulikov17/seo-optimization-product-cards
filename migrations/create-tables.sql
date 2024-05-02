CREATE TABLE IF NOT EXISTS "users" (
    "chat_id" VARCHAR(256) NOT NULL PRIMARY KEY,
    "username" VARCHAR(256) NOT NULL,
    "first_name" VARCHAR(256) DEFAULT NULL,
    "last_name" VARCHAR(256) DEFAULT NULL,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);