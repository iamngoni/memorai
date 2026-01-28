use anyhow::{Context, Result};
use reqwest::Client;

use crate::config::Config;
use crate::models::{OllamaEmbedRequest, OllamaEmbedResponse};

pub struct EmbeddingClient {
    client: Client,
    ollama_url: String,
    model: String,
}

impl EmbeddingClient {
    pub fn new(config: &Config) -> Self {
        Self {
            client: Client::new(),
            ollama_url: config.ollama_url.clone(),
            model: config.embed_model.clone(),
        }
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embed", self.ollama_url);
        let request = OllamaEmbedRequest {
            model: self.model.clone(),
            input: text.to_string(),
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to connect to Ollama")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Ollama embedding request failed ({}): {}", status, body);
        }

        let embed_response: OllamaEmbedResponse = response
            .json()
            .await
            .context("Failed to parse Ollama embedding response")?;

        embed_response
            .embeddings
            .into_iter()
            .next()
            .context("No embedding returned from Ollama")
    }
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}
