//! Immutable logical identities and version fences, independent of operations.
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Compatibility {
    pub namespace: String,
    pub cache_salt: Option<String>,
    pub layout_id: String,
    pub page_size: u64,
    pub components: Vec<String>,
    pub hash_scheme: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixTarget {
    pub kind: String,
    pub compatibility: Compatibility,
    pub start_token: u64,
    pub end_token: u64,
    pub token_ids: Option<Vec<u32>>,
}

/// Identity survives worker changes; a worker epoch is deliberately absent.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SessionIdentity {
    /// Optional admission generation. Preserve the caller's choice throughout the task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retirement_generation: Option<u64>,
    pub caller_id: String,
    pub session_id: String,
    pub incarnation: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimStatus {
    Active,
    Released,
    Expired,
    Invalidated,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimState {
    pub caller_id: String,
    pub task_incarnation: String,
    pub claim_id: String,
    pub revision: u64,
    pub state: ClaimStatus,
    pub owner_epoch: String,
    #[serde(default)]
    pub tombstone: bool,
    #[serde(default = "result_version")]
    pub protection_version: String,
    pub effective_protection: Option<String>,
    pub remaining_ttl: Option<f64>,
    pub priority: Option<i32>,
    #[serde(default)]
    pub actual_scope: Vec<Value>,
    pub location: Option<crate::protocol::Location>,
}

/// Carries observations, never creates claims or confirms queued placement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicResult {
    #[serde(default = "result_version")]
    pub result_version: String,
    #[serde(default)]
    pub requested_scope: Vec<Value>,
    #[serde(default)]
    pub accepted_scope: Vec<Value>,
    #[serde(default)]
    pub ready_scope: Vec<Value>,
    pub ready_location: Option<crate::protocol::Location>,
    #[serde(default)]
    pub preserved_scope: Vec<Value>,
    #[serde(default)]
    pub reclaimed_scope: Vec<Value>,
    #[serde(default)]
    pub missing_scope: Vec<Value>,
    #[serde(default)]
    pub blocked_scope: Vec<Value>,
    pub owner_epoch: String,
    pub state_version: u64,
    pub outcome_sequence: u64,
    pub cache_generation: Option<u64>,
    pub source_observed_at: Option<f64>,
    pub claim: Option<ClaimState>,
}
fn result_version() -> String {
    "1.0".into()
}
