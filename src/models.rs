use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use surrealdb::sql::Thing;

// Database record
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Memory {
    pub id: Option<Thing>,
    pub text: String,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub embedding: Vec<f32>,
    pub created_at: String,
    pub updated_at: String,
}

// API request to create a memory
#[derive(Debug, Deserialize)]
pub struct CreateMemoryRequest {
    pub text: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub source: Option<String>,
}

// API request for bulk import
#[derive(Debug, Deserialize)]
pub struct BulkCreateRequest {
    pub memories: Vec<CreateMemoryRequest>,
}

// API response for a memory
#[derive(Debug, Serialize)]
pub struct MemoryResponse {
    pub id: String,
    pub text: String,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

impl MemoryResponse {
    pub fn from_memory(m: Memory) -> Self {
        let id = m
            .id
            .map(|t| t.id.to_string())
            .unwrap_or_default();
        Self {
            id,
            text: m.text,
            tags: m.tags,
            source: m.source,
            created_at: m.created_at,
            updated_at: m.updated_at,
        }
    }
}

// Search result
#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub memory: MemoryResponse,
    pub score: f32,
}

// Search query params
#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    pub q: String,
    pub limit: Option<usize>,
}

// List query params
#[derive(Debug, Deserialize)]
pub struct ListQuery {
    pub page: Option<usize>,
    pub per_page: Option<usize>,
    pub tag: Option<String>,
    pub source: Option<String>,
}

// Stats response
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub total_memories: usize,
    pub tags: Vec<TagCount>,
    pub sources: Vec<SourceCount>,
}

#[derive(Debug, Serialize)]
pub struct TagCount {
    pub tag: String,
    pub count: usize,
}

#[derive(Debug, Serialize)]
pub struct SourceCount {
    pub source: String,
    pub count: usize,
}

// Profile response
#[derive(Debug, Serialize)]
pub struct ProfileResponse {
    pub profile: String,
    pub memory_count: usize,
}

// Bulk import response
#[derive(Debug, Serialize)]
pub struct BulkResponse {
    pub created: usize,
    pub failed: usize,
    pub errors: Vec<String>,
}

// Ollama API types
#[derive(Debug, Serialize)]
pub struct OllamaEmbedRequest {
    pub model: String,
    pub input: String,
}

#[derive(Debug, Deserialize)]
pub struct OllamaEmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
pub struct OllamaGenerateRequest {
    pub model: String,
    pub prompt: String,
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct OllamaGenerateResponse {
    pub response: String,
}

// Generic API response wrapper
#[derive(Debug, Serialize)]
pub struct ApiResponse<T: Serialize> {
    pub ok: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl<T: Serialize> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            ok: true,
            data: Some(data),
            error: None,
        }
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            ok: false,
            data: None,
            error: Some(msg.into()),
        }
    }
}
