use std::env;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct Config {
    pub port: u16,
    pub ollama_url: String,
    pub embed_model: String,
    pub chat_model: String,
    pub data_dir: PathBuf,
}

impl Config {
    pub fn from_env() -> Self {
        let home = env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        Self {
            port: env::var("MEMORAI_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8484),
            ollama_url: env::var("MEMORAI_OLLAMA_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string()),
            embed_model: env::var("MEMORAI_EMBED_MODEL")
                .unwrap_or_else(|_| "mxbai-embed-large".to_string()),
            chat_model: env::var("MEMORAI_CHAT_MODEL")
                .unwrap_or_else(|_| "qwen2.5:14b".to_string()),
            data_dir: env::var("MEMORAI_DATA_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from(home).join(".memorai").join("data")),
        }
    }
}
