use anyhow::{Context, Result};
use chrono::Utc;
use surrealdb::engine::local::RocksDb;
use surrealdb::Surreal;

use crate::config::Config;
use crate::models::Memory;

pub type Db = Surreal<surrealdb::engine::local::Db>;

pub async fn init_db(config: &Config) -> Result<Db> {
    // Ensure data directory exists
    std::fs::create_dir_all(&config.data_dir)
        .context("Failed to create data directory")?;

    let path = config.data_dir.to_string_lossy().to_string();
    let db = Surreal::new::<RocksDb>(&path)
        .await
        .context("Failed to initialize SurrealDB")?;

    db.use_ns("memorai")
        .use_db("memories")
        .await
        .context("Failed to select namespace/database")?;

    // Create table schema
    db.query(
        "DEFINE TABLE IF NOT EXISTS memory SCHEMAFULL;
         DEFINE FIELD IF NOT EXISTS text ON TABLE memory TYPE string;
         DEFINE FIELD IF NOT EXISTS tags ON TABLE memory TYPE array;
         DEFINE FIELD IF NOT EXISTS tags.* ON TABLE memory TYPE string;
         DEFINE FIELD IF NOT EXISTS source ON TABLE memory TYPE option<string>;
         DEFINE FIELD IF NOT EXISTS embedding ON TABLE memory TYPE array;
         DEFINE FIELD IF NOT EXISTS embedding.* ON TABLE memory TYPE float;
         DEFINE FIELD IF NOT EXISTS created_at ON TABLE memory TYPE datetime;
         DEFINE FIELD IF NOT EXISTS updated_at ON TABLE memory TYPE datetime;
         DEFINE INDEX IF NOT EXISTS idx_tags ON TABLE memory FIELDS tags;
         DEFINE INDEX IF NOT EXISTS idx_source ON TABLE memory FIELDS source;",
    )
    .await
    .context("Failed to define schema")?;

    tracing::info!("Database initialized at {}", path);
    Ok(db)
}

pub async fn create_memory(
    db: &Db,
    text: String,
    tags: Vec<String>,
    source: Option<String>,
    embedding: Vec<f32>,
) -> Result<Memory> {
    let now = Utc::now();

    let memory: Option<Memory> = db
        .create("memory")
        .content(Memory {
            id: None,
            text,
            tags,
            source,
            embedding,
            created_at: now,
            updated_at: now,
        })
        .await
        .context("Failed to create memory")?;

    memory.context("No memory returned after creation")
}

pub async fn get_all_memories(db: &Db) -> Result<Vec<Memory>> {
    let memories: Vec<Memory> = db
        .select("memory")
        .await
        .context("Failed to fetch memories")?;

    Ok(memories)
}

pub async fn get_memories_paginated(
    db: &Db,
    page: usize,
    per_page: usize,
    tag: Option<&str>,
    source: Option<&str>,
) -> Result<Vec<Memory>> {
    let offset = (page.saturating_sub(1)) * per_page;

    let query = match (tag, source) {
        (Some(t), Some(s)) => {
            db.query("SELECT * FROM memory WHERE $tag IN tags AND source = $source ORDER BY created_at DESC LIMIT $limit START $offset")
                .bind(("tag", t.to_string()))
                .bind(("source", s.to_string()))
                .bind(("limit", per_page))
                .bind(("offset", offset))
                .await
        }
        (Some(t), None) => {
            db.query("SELECT * FROM memory WHERE $tag IN tags ORDER BY created_at DESC LIMIT $limit START $offset")
                .bind(("tag", t.to_string()))
                .bind(("limit", per_page))
                .bind(("offset", offset))
                .await
        }
        (None, Some(s)) => {
            db.query("SELECT * FROM memory WHERE source = $source ORDER BY created_at DESC LIMIT $limit START $offset")
                .bind(("source", s.to_string()))
                .bind(("limit", per_page))
                .bind(("offset", offset))
                .await
        }
        (None, None) => {
            db.query("SELECT * FROM memory ORDER BY created_at DESC LIMIT $limit START $offset")
                .bind(("limit", per_page))
                .bind(("offset", offset))
                .await
        }
    };

    let mut result = query.context("Failed to query memories")?;
    let memories: Vec<Memory> = result.take(0).context("Failed to parse memories")?;
    Ok(memories)
}

pub async fn delete_memory(db: &Db, id: &str) -> Result<Option<Memory>> {
    let thing = format!("memory:{}", id);
    let memory: Option<Memory> = db
        .delete((&"memory", id))
        .await
        .context("Failed to delete memory")?;

    let _ = thing; // suppress warning
    Ok(memory)
}

pub async fn count_memories(db: &Db) -> Result<usize> {
    let mut result = db
        .query("SELECT count() FROM memory GROUP ALL")
        .await
        .context("Failed to count memories")?;

    #[derive(serde::Deserialize)]
    struct CountResult {
        count: usize,
    }

    let count: Option<CountResult> = result.take(0).ok().and_then(|v: Vec<CountResult>| v.into_iter().next());
    Ok(count.map(|c| c.count).unwrap_or(0))
}

pub async fn get_all_texts(db: &Db) -> Result<Vec<String>> {
    let mut result = db
        .query("SELECT text FROM memory ORDER BY created_at DESC LIMIT 100")
        .await
        .context("Failed to fetch texts")?;

    #[derive(serde::Deserialize)]
    struct TextOnly {
        text: String,
    }

    let texts: Vec<TextOnly> = result.take(0).context("Failed to parse texts")?;
    Ok(texts.into_iter().map(|t| t.text).collect())
}

pub async fn get_tag_counts(db: &Db) -> Result<Vec<(String, usize)>> {
    let memories = get_all_memories(db).await?;
    let mut tag_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for m in &memories {
        for tag in &m.tags {
            *tag_map.entry(tag.clone()).or_insert(0) += 1;
        }
    }
    let mut counts: Vec<(String, usize)> = tag_map.into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1));
    Ok(counts)
}

pub async fn get_source_counts(db: &Db) -> Result<Vec<(String, usize)>> {
    let memories = get_all_memories(db).await?;
    let mut source_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for m in &memories {
        if let Some(ref src) = m.source {
            *source_map.entry(src.clone()).or_insert(0) += 1;
        }
    }
    let mut counts: Vec<(String, usize)> = source_map.into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1));
    Ok(counts)
}
