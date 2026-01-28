mod config;
mod db;
mod embeddings;
mod models;
mod profile;
mod server;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::sync::Arc;

use config::Config;
use embeddings::EmbeddingClient;

#[derive(Parser)]
#[command(name = "memorai", version, about = "Local-first AI memory system with semantic search")]
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
    let db = db::init_db(&config).await?;
    let port = config.port;
    let embeddings = EmbeddingClient::new(&config);

    let state = Arc::new(server::AppState {
        db,
        config,
        embeddings,
    });

    let app = server::create_router(state);
    let addr = format!("0.0.0.0:{}", port);

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

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("Listening on {}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn add_memory(
    config: Config,
    text: String,
    tags: Vec<String>,
    source: Option<String>,
) -> Result<()> {
    let db = db::init_db(&config).await?;
    let embeddings = EmbeddingClient::new(&config);

    println!("Generating embedding...");
    let embedding = embeddings.embed(&text).await?;

    let memory = db::create_memory(&db, text, tags, source, embedding).await?;
    let response = models::MemoryResponse::from_memory(memory);

    println!("✅ Memory stored (id: {})", response.id);
    println!("   Text: {}", response.text);
    if !response.tags.is_empty() {
        println!("   Tags: {}", response.tags.join(", "));
    }
    if let Some(ref src) = response.source {
        println!("   Source: {}", src);
    }
    Ok(())
}

async fn search(config: Config, query: String, limit: usize) -> Result<()> {
    let db = db::init_db(&config).await?;
    let embeddings = EmbeddingClient::new(&config);

    println!("Searching for: \"{}\"", query);
    let query_embedding = embeddings.embed(&query).await?;

    let memories = db::get_all_memories(&db).await?;

    let mut scored: Vec<(models::MemoryResponse, f32)> = memories
        .into_iter()
        .map(|m| {
            let score = embeddings::cosine_similarity(&query_embedding, &m.embedding);
            (models::MemoryResponse::from_memory(m), score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);

    if scored.is_empty() {
        println!("No memories found.");
    } else {
        println!("\n🔍 Top {} results:\n", scored.len());
        for (i, (mem, score)) in scored.iter().enumerate() {
            println!("{}. [score: {:.4}] {}", i + 1, score, mem.text);
            if !mem.tags.is_empty() {
                println!("   Tags: {}", mem.tags.join(", "));
            }
            println!();
        }
    }
    Ok(())
}

async fn stats(config: Config) -> Result<()> {
    let db = db::init_db(&config).await?;
    let count = db::count_memories(&db).await?;
    let tags = db::get_tag_counts(&db).await?;
    let sources = db::get_source_counts(&db).await?;

    println!("📊 memorai stats\n");
    println!("Total memories: {}", count);

    if !tags.is_empty() {
        println!("\nTop tags:");
        for (tag, cnt) in tags.iter().take(10) {
            println!("  {} ({})", tag, cnt);
        }
    }

    if !sources.is_empty() {
        println!("\nTop sources:");
        for (src, cnt) in sources.iter().take(10) {
            println!("  {} ({})", src, cnt);
        }
    }

    Ok(())
}

async fn generate_profile(config: Config) -> Result<()> {
    let db = db::init_db(&config).await?;

    println!("Generating profile from stored memories...\n");
    let (profile_text, count) = profile::generate_profile(&db, &config).await?;

    println!("👤 Profile (based on {} memories):\n", count);
    println!("{}", profile_text);
    Ok(())
}
