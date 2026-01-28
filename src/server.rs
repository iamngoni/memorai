use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{delete, get, post},
    Router,
};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use crate::config::Config;
use crate::db::{self, Db};
use crate::embeddings::{cosine_similarity, EmbeddingClient};
use crate::models::*;
use crate::profile;

pub struct AppState {
    pub db: Db,
    pub config: Config,
    pub embeddings: EmbeddingClient,
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/memories", post(create_memory))
        .route("/v1/memories", get(list_memories))
        .route("/v1/memories/bulk", post(bulk_create))
        .route("/v1/memories/{id}", delete(delete_memory))
        .route("/v1/search", get(search))
        .route("/v1/stats", get(stats))
        .route("/v1/profile", get(get_profile))
        .route("/health", get(health))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok", "service": "memorai"}))
}

async fn create_memory(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateMemoryRequest>,
) -> (StatusCode, Json<ApiResponse<MemoryResponse>>) {
    if req.text.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::error("Text cannot be empty")),
        );
    }

    let embedding = match state.embeddings.embed(&req.text).await {
        Ok(e) => e,
        Err(err) => {
            tracing::error!("Embedding failed: {}", err);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::error(format!("Embedding failed: {}", err))),
            );
        }
    };

    match db::create_memory(&state.db, req.text, req.tags, req.source, embedding).await {
        Ok(memory) => (
            StatusCode::CREATED,
            Json(ApiResponse::success(MemoryResponse::from_memory(memory))),
        ),
        Err(err) => {
            tracing::error!("Failed to create memory: {}", err);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::error(format!("Failed to create memory: {}", err))),
            )
        }
    }
}

async fn list_memories(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListQuery>,
) -> (StatusCode, Json<ApiResponse<Vec<MemoryResponse>>>) {
    let page = query.page.unwrap_or(1);
    let per_page = query.per_page.unwrap_or(20).min(100);

    match db::get_memories_paginated(
        &state.db,
        page,
        per_page,
        query.tag.as_deref(),
        query.source.as_deref(),
    )
    .await
    {
        Ok(memories) => {
            let responses: Vec<MemoryResponse> =
                memories.into_iter().map(MemoryResponse::from_memory).collect();
            (StatusCode::OK, Json(ApiResponse::success(responses)))
        }
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::error(format!("Failed to list memories: {}", err))),
        ),
    }
}

async fn delete_memory(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> (StatusCode, Json<ApiResponse<String>>) {
    match db::delete_memory(&state.db, &id).await {
        Ok(Some(_)) => (
            StatusCode::OK,
            Json(ApiResponse::success("Memory deleted".to_string())),
        ),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(ApiResponse::error("Memory not found")),
        ),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::error(format!("Failed to delete memory: {}", err))),
        ),
    }
}

async fn search(
    State(state): State<Arc<AppState>>,
    Query(query): Query<SearchQuery>,
) -> (StatusCode, Json<ApiResponse<Vec<SearchResult>>>) {
    if query.q.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::error("Query cannot be empty")),
        );
    }

    let limit = query.limit.unwrap_or(5).min(50);

    // Embed the query
    let query_embedding = match state.embeddings.embed(&query.q).await {
        Ok(e) => e,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::error(format!("Embedding failed: {}", err))),
            );
        }
    };

    // Get all memories and compute similarity
    let memories = match db::get_all_memories(&state.db).await {
        Ok(m) => m,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::error(format!("Failed to fetch memories: {}", err))),
            );
        }
    };

    let mut scored: Vec<SearchResult> = memories
        .into_iter()
        .map(|m| {
            let score = cosine_similarity(&query_embedding, &m.embedding);
            SearchResult {
                memory: MemoryResponse::from_memory(m),
                score,
            }
        })
        .collect();

    // Sort by score descending
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);

    (StatusCode::OK, Json(ApiResponse::success(scored)))
}

async fn stats(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<ApiResponse<StatsResponse>>) {
    let total = match db::count_memories(&state.db).await {
        Ok(c) => c,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::error(format!("Failed to get stats: {}", err))),
            );
        }
    };

    let tag_counts = db::get_tag_counts(&state.db).await.unwrap_or_default();
    let source_counts = db::get_source_counts(&state.db).await.unwrap_or_default();

    let response = StatsResponse {
        total_memories: total,
        tags: tag_counts
            .into_iter()
            .map(|(tag, count)| TagCount { tag, count })
            .collect(),
        sources: source_counts
            .into_iter()
            .map(|(source, count)| SourceCount { source, count })
            .collect(),
    };

    (StatusCode::OK, Json(ApiResponse::success(response)))
}

async fn get_profile(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<ApiResponse<ProfileResponse>>) {
    match profile::generate_profile(&state.db, &state.config).await {
        Ok((profile_text, count)) => (
            StatusCode::OK,
            Json(ApiResponse::success(ProfileResponse {
                profile: profile_text,
                memory_count: count,
            })),
        ),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::error(format!("Failed to generate profile: {}", err))),
        ),
    }
}

async fn bulk_create(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BulkCreateRequest>,
) -> (StatusCode, Json<ApiResponse<BulkResponse>>) {
    let mut created = 0;
    let mut failed = 0;
    let mut errors = Vec::new();

    for (i, mem) in req.memories.into_iter().enumerate() {
        if mem.text.trim().is_empty() {
            failed += 1;
            errors.push(format!("Item {}: empty text", i));
            continue;
        }

        let embedding = match state.embeddings.embed(&mem.text).await {
            Ok(e) => e,
            Err(err) => {
                failed += 1;
                errors.push(format!("Item {}: embedding failed: {}", i, err));
                continue;
            }
        };

        match db::create_memory(&state.db, mem.text, mem.tags, mem.source, embedding).await {
            Ok(_) => created += 1,
            Err(err) => {
                failed += 1;
                errors.push(format!("Item {}: {}", i, err));
            }
        }
    }

    (
        StatusCode::OK,
        Json(ApiResponse::success(BulkResponse {
            created,
            failed,
            errors,
        })),
    )
}
