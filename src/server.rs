use actix_web::{web, HttpResponse, Scope};
use std::sync::Arc;
use tokio::sync::RwLock;

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

pub type SharedState = web::Data<Arc<RwLock<AppState>>>;

pub fn api_scope() -> Scope {
    web::scope("/v1")
        .route("/memories", web::post().to(create_memory))
        .route("/memories", web::get().to(list_memories))
        .route("/memories/bulk", web::post().to(bulk_create))
        .route("/memories/{id}", web::delete().to(delete_memory))
        .route("/search", web::get().to(search))
        .route("/stats", web::get().to(stats))
        .route("/profile", web::get().to(get_profile))
}

pub fn health_route() -> actix_web::Resource {
    web::resource("/health").route(web::get().to(health))
}

async fn health() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({"status": "ok", "service": "memorai"}))
}

async fn create_memory(
    state: SharedState,
    body: web::Json<CreateMemoryRequest>,
) -> HttpResponse {
    let req = body.into_inner();

    if req.text.trim().is_empty() {
        return HttpResponse::BadRequest().json(ApiResponse::<()>::error("Text cannot be empty"));
    }

    let state = state.read().await;

    let embedding = match state.embeddings.embed(&req.text).await {
        Ok(e) => e,
        Err(err) => {
            tracing::error!("Embedding failed: {}", err);
            return HttpResponse::InternalServerError()
                .json(ApiResponse::<()>::error(format!("Embedding failed: {}", err)));
        }
    };

    match db::create_memory(&state.db, req.text, req.tags, req.source, embedding).await {
        Ok(memory) => HttpResponse::Created()
            .json(ApiResponse::success(MemoryResponse::from_memory(memory))),
        Err(err) => {
            tracing::error!("Failed to create memory: {}", err);
            HttpResponse::InternalServerError()
                .json(ApiResponse::<()>::error(format!("Failed to create memory: {}", err)))
        }
    }
}

async fn list_memories(
    state: SharedState,
    query: web::Query<ListQuery>,
) -> HttpResponse {
    let page = query.page.unwrap_or(1);
    let per_page = query.per_page.unwrap_or(20).min(100);
    let state = state.read().await;

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
            HttpResponse::Ok().json(ApiResponse::success(responses))
        }
        Err(err) => HttpResponse::InternalServerError()
            .json(ApiResponse::<()>::error(format!("Failed to list memories: {}", err))),
    }
}

async fn delete_memory(
    state: SharedState,
    path: web::Path<String>,
) -> HttpResponse {
    let id = path.into_inner();
    let state = state.read().await;

    match db::delete_memory(&state.db, &id).await {
        Ok(Some(_)) => {
            HttpResponse::Ok().json(ApiResponse::success("Memory deleted".to_string()))
        }
        Ok(None) => {
            HttpResponse::NotFound().json(ApiResponse::<()>::error("Memory not found"))
        }
        Err(err) => HttpResponse::InternalServerError()
            .json(ApiResponse::<()>::error(format!("Failed to delete memory: {}", err))),
    }
}

async fn search(
    state: SharedState,
    query: web::Query<SearchQuery>,
) -> HttpResponse {
    if query.q.trim().is_empty() {
        return HttpResponse::BadRequest()
            .json(ApiResponse::<()>::error("Query cannot be empty"));
    }

    let limit = query.limit.unwrap_or(5).min(50);
    let state = state.read().await;

    let query_embedding = match state.embeddings.embed(&query.q).await {
        Ok(e) => e,
        Err(err) => {
            return HttpResponse::InternalServerError()
                .json(ApiResponse::<()>::error(format!("Embedding failed: {}", err)));
        }
    };

    let memories = match db::get_all_memories(&state.db).await {
        Ok(m) => m,
        Err(err) => {
            return HttpResponse::InternalServerError()
                .json(ApiResponse::<()>::error(format!("Failed to fetch memories: {}", err)));
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

    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);

    HttpResponse::Ok().json(ApiResponse::success(scored))
}

async fn stats(state: SharedState) -> HttpResponse {
    let state = state.read().await;

    let total = match db::count_memories(&state.db).await {
        Ok(c) => c,
        Err(err) => {
            return HttpResponse::InternalServerError()
                .json(ApiResponse::<()>::error(format!("Failed to get stats: {}", err)));
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

    HttpResponse::Ok().json(ApiResponse::success(response))
}

async fn get_profile(state: SharedState) -> HttpResponse {
    let state = state.read().await;

    match profile::generate_profile(&state.db, &state.config).await {
        Ok((profile_text, count)) => HttpResponse::Ok().json(ApiResponse::success(
            ProfileResponse {
                profile: profile_text,
                memory_count: count,
            },
        )),
        Err(err) => HttpResponse::InternalServerError()
            .json(ApiResponse::<()>::error(format!("Failed to generate profile: {}", err))),
    }
}

async fn bulk_create(
    state: SharedState,
    body: web::Json<BulkCreateRequest>,
) -> HttpResponse {
    let req = body.into_inner();
    let state = state.read().await;
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

    HttpResponse::Ok().json(ApiResponse::success(BulkResponse {
        created,
        failed,
        errors,
    }))
}
