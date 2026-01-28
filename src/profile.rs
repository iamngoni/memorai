use anyhow::{Context, Result};
use reqwest::Client;

use crate::config::Config;
use crate::db::{self, Db};
use crate::models::{OllamaGenerateRequest, OllamaGenerateResponse};

pub async fn generate_profile(db: &Db, config: &Config) -> Result<(String, usize)> {
    let texts = db::get_all_texts(db).await?;
    let count = texts.len();

    if count == 0 {
        return Ok((
            "No memories stored yet. Add some memories to generate a profile.".to_string(),
            0,
        ));
    }

    let memories_text = texts
        .iter()
        .enumerate()
        .map(|(i, t)| format!("{}. {}", i + 1, t))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        "Based on the following collection of memories/notes from a person, create a concise user profile summary. \
         Include their interests, expertise, personality traits, and any patterns you notice. \
         Be insightful but respectful of privacy. Write in third person.\n\n\
         Memories:\n{}\n\n\
         Profile summary:",
        memories_text
    );

    let client = Client::new();
    let url = format!("{}/api/generate", config.ollama_url);
    let request = OllamaGenerateRequest {
        model: config.chat_model.clone(),
        prompt,
        stream: false,
    };

    let response = client
        .post(&url)
        .json(&request)
        .send()
        .await
        .context("Failed to connect to Ollama for profile generation")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("Ollama generate request failed ({}): {}", status, body);
    }

    let gen_response: OllamaGenerateResponse = response
        .json()
        .await
        .context("Failed to parse Ollama generate response")?;

    Ok((gen_response.response.trim().to_string(), count))
}
