//! genai client construction. Adapter bound to Ollama; endpoint overridden via a
//! `ServiceTargetResolver` when the host is not the default (localhost:11434).
//!
//! This is a deliberately thin seam — a richer client abstraction lands next.

use genai::ModelIden;
use genai::adapter::AdapterKind;
use genai::resolver::{AuthData, Endpoint, ServiceTargetResolver};
use genai::{Client, ServiceTarget};

/// Build a genai client for `provider`. `ollama` forces the local `endpoint` and
/// disables auth; any other value uses genai's native routing — the adapter is
/// picked from the model name (e.g. `gemini-…` → Gemini, `gpt-…` → OpenAI,
/// `claude-…` → Anthropic).
///
/// `api_key` is injected via config (a desktop app supplies it from its own
/// settings/keychain — there is no process env var). When `None`, genai falls
/// back to the provider's default env var (e.g. `GEMINI_API_KEY`), handy for the
/// CLI. This is also how LLM and embedder can live on different providers.
pub fn client(provider: &str, endpoint: &str, api_key: Option<&str>) -> Client {
    match provider {
        "ollama" => ollama_client(endpoint),
        _ => match api_key.filter(|k| !k.is_empty()) {
            Some(key) => cloud_client(key),
            None => Client::default(),
        },
    }
}

/// Cloud client: keep genai's model-name adapter routing, but inject the API key
/// from config instead of the environment.
fn cloud_client(api_key: &str) -> Client {
    let key = api_key.to_string();
    Client::builder()
        .with_auth_resolver_fn(move |_: ModelIden| Ok(Some(AuthData::from_single(key.clone()))))
        .build()
}

/// Ollama client pointing at `endpoint` (e.g. "http://gmk-ai-master:11434/").
pub fn ollama_client(endpoint: &str) -> Client {
    let endpoint = endpoint.to_string();
    let resolver = ServiceTargetResolver::from_resolver_fn(
        move |mut target: ServiceTarget| -> genai::resolver::Result<ServiceTarget> {
            target.endpoint = Endpoint::from_owned(endpoint.clone());
            target.auth = AuthData::None;
            Ok(target)
        },
    );

    Client::builder()
        .with_adapter_kind(AdapterKind::Ollama)
        .with_service_target_resolver(resolver)
        .build()
}
