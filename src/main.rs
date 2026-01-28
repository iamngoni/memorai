mod config;
mod db;
mod embeddings;
mod models;
mod profile;
mod server;

use anyhow::Result;
use clap::{Parser, Subcommand};

use config::Config;

#[derive(Parser)]
#[command(
    name = "memorai",
    version,
    about = "Local-first AI memory system with semantic search"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the memorai API server
    Serve,
    /// Add a memory
    Add {
        /// The text to remember
        text: String,
        /// Comma-separated tags
        #[arg(short, long)]
        tags: Option<String>,
        /// Source of the memory
        #[arg(short, long)]
        source: Option<String>,
    },
    /// Search memories semantically
    Search {
        /// Search query
        query: String,
        /// Max results
        #[arg(short, long, default_value = "5")]
        limit: usize,
    },
    /// Show memory statistics
    Stats,
    /// Generate a user profile from stored memories
    Profile,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    let config = Config::from_env();

    match cli.command {
        Commands::Serve => serve(config).await,
        Commands::Add { text, tags, source } => {
            let tags: Vec<String> = tags
                .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_default();
            add_memory(config, text, tags, source).await
        }
        Commands::Search { query, limit } => search(config, query, limit).await,
        Commands::Stats => stats(config).await,
        Commands::Profile => generate_profile(config).await,
    }
}

async fn serve(config: Config) -> Result<()> {
    use std::sync::Arc;
    use embeddings::EmbeddingClient;

    let db = db::init_db(&config).await?;
    let port = config.port;
    let embeddings = EmbeddingClient::new(&config);

    let state = Arc::new(tokio::sync::RwLock::new(server::AppState {
        db,
        config,
        embeddings,
    }));

    let shared_state = actix_web::web::Data::new(state);

    println!(
        r#"
  ╔══════════════════════════════════════╗
  ║         memorai v{}          ║
  ║   Local AI Memory System             ║
  ╠══════════════════════════════════════╣
  ║  API:  http://localhost:{}         ║
  ║  Docs: http://localhost:{}/health  ║
  ╚══════════════════════════════════════╝
"#,
        env!("CARGO_PKG_VERSION"),
        port,
        port
    );

    tracing::info!("Listening on 0.0.0.0:{}", port);

    actix_web::HttpServer::new(move || {
        actix_web::App::new()
            .app_data(shared_state.clone())
            .wrap(actix_cors::Cors::permissive())
            .service(server::api_scope())
            .service(server::health_route())
    })
    .bind(format!("0.0.0.0:{}", port))?
    .run()
    .await?;

    Ok(())
}

fn api_url(config: &Config) -> String {
    format!("http://localhost:{}", config.port)
}

async fn add_memory(
    config: Config,
    text: String,
    tags: Vec<String>,
    source: Option<String>,
) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("{}/v1/memories", api_url(&config));

    let mut body = serde_json::json!({ "text": text, "tags": tags });
    if let Some(src) = &source {
        body["source"] = serde_json::json!(src);
    }

    println!("Adding memory...");
    let resp = client.post(&url).json(&body).send().await?;

    if resp.status().is_success() {
        let data: serde_json::Value = resp.json().await?;
        if let Some(mem) = data.get("data") {
            println!("✅ Memory stored (id: {})", mem["id"].as_str().unwrap_or("?"));
            println!("   Text: {}", mem["text"].as_str().unwrap_or(""));
            if let Some(tags) = mem["tags"].as_array() {
                if !tags.is_empty() {
                    let tag_strs: Vec<&str> = tags.iter().filter_map(|t| t.as_str()).collect();
                    println!("   Tags: {}", tag_strs.join(", "));
                }
            }
            if let Some(src) = mem["source"].as_str() {
                println!("   Source: {}", src);
            }
        }
    } else {
        let err: serde_json::Value = resp.json().await?;
        println!("❌ {}", err["error"].as_str().unwrap_or("Unknown error"));
    }
    Ok(())
}

async fn search(config: Config, query: String, limit: usize) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("{}/v1/search?q={}&limit={}", api_url(&config), urlencoding::encode(&query), limit);

    println!("Searching for: \"{}\"", query);
    let resp = client.get(&url).send().await?;

    if resp.status().is_success() {
        let data: serde_json::Value = resp.json().await?;
        if let Some(results) = data["data"].as_array() {
            if results.is_empty() {
                println!("No memories found.");
            } else {
                println!("\n🔍 Top {} results:\n", results.len());
                for (i, r) in results.iter().enumerate() {
                    let score = r["score"].as_f64().unwrap_or(0.0);
                    let text = r["memory"]["text"].as_str().unwrap_or("");
                    println!("{}. [score: {:.4}] {}", i + 1, score, text);
                    if let Some(tags) = r["memory"]["tags"].as_array() {
                        if !tags.is_empty() {
                            let tag_strs: Vec<&str> = tags.iter().filter_map(|t| t.as_str()).collect();
                            println!("   Tags: {}", tag_strs.join(", "));
                        }
                    }
                    println!();
                }
            }
        }
    } else {
        let err: serde_json::Value = resp.json().await?;
        println!("❌ {}", err["error"].as_str().unwrap_or("Unknown error"));
    }
    Ok(())
}

async fn stats(config: Config) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("{}/v1/stats", api_url(&config));

    let resp = client.get(&url).send().await?;

    if resp.status().is_success() {
        let data: serde_json::Value = resp.json().await?;
        if let Some(stats) = data.get("data") {
            println!("📊 memorai stats\n");
            println!("Total memories: {}", stats["total_memories"]);

            if let Some(tags) = stats["top_tags"].as_array() {
                if !tags.is_empty() {
                    println!("\nTop tags:");
                    for t in tags.iter().take(10) {
                        println!("  {} ({})", t["tag"].as_str().unwrap_or("?"), t["count"]);
                    }
                }
            }

            if let Some(sources) = stats["top_sources"].as_array() {
                if !sources.is_empty() {
                    println!("\nTop sources:");
                    for s in sources.iter().take(10) {
                        println!("  {} ({})", s["source"].as_str().unwrap_or("?"), s["count"]);
                    }
                }
            }
        }
    } else {
        println!("❌ Failed to get stats");
    }
    Ok(())
}

async fn generate_profile(config: Config) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("{}/v1/profile", api_url(&config));

    println!("Generating profile from stored memories...\n");
    let resp = client.get(&url).send().await?;

    if resp.status().is_success() {
        let data: serde_json::Value = resp.json().await?;
        if let Some(profile) = data.get("data") {
            println!("👤 Profile (based on {} memories):\n", profile["memory_count"]);
            println!("{}", profile["profile"].as_str().unwrap_or(""));
        }
    } else {
        println!("❌ Failed to generate profile");
    }
    Ok(())
}
