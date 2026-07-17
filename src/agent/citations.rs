//! Inline citation handling: renumber `[sN]` handles to `[n]` for display, collect
//! them in order of first appearance, and normalise text for fuzzy quote checks.

/// Renumber inline handles for display and collect them in order of appearance.
/// Handles grouped `[s1, s3]`, adjacent `[s1][s3]`, case `[S1]`; leaves bare `[1]`.
pub(super) fn resolve_citations(text: &str) -> (String, Vec<String>) {
    let mut order: Vec<String> = Vec::new();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        let tail = &text[i..];
        if tail.starts_with('[')
            && let Some(rel) = tail[1..].find(']')
        {
            let (rebuilt, found) = rewrite_group(&tail[1..1 + rel], &mut order);
            if found {
                out.push('[');
                out.push_str(&rebuilt);
                out.push(']');
            } else {
                out.push_str(&text[i..i + 1 + rel + 1]);
            }
            i += 1 + rel + 1;
            continue;
        }
        let ch = tail.chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    (out, order)
}

fn rewrite_group(inner: &str, order: &mut Vec<String>) -> (String, bool) {
    let mut rebuilt = String::new();
    let mut found = false;
    let mut j = 0;
    while j < inner.len() {
        let ch = inner[j..].chars().next().unwrap();
        if ch.is_ascii_alphanumeric() {
            let start = j;
            while j < inner.len() {
                let c = inner[j..].chars().next().unwrap();
                if c.is_ascii_alphanumeric() {
                    j += c.len_utf8();
                } else {
                    break;
                }
            }
            let word = &inner[start..j];
            if is_handle(word) {
                found = true;
                rebuilt.push_str(&number_of(&word.to_ascii_lowercase(), order).to_string());
            } else {
                rebuilt.push_str(word);
            }
        } else {
            rebuilt.push(ch);
            j += ch.len_utf8();
        }
    }
    (rebuilt, found)
}

fn number_of(handle: &str, order: &mut Vec<String>) -> usize {
    if let Some(pos) = order.iter().position(|h| h == handle) {
        pos + 1
    } else {
        order.push(handle.to_string());
        order.len()
    }
}

fn is_handle(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some('s') | Some('S')) && {
        let rest: String = chars.collect();
        !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit())
    }
}

/// Whitespace/case-normalise, for fuzzy quote-in-passage verification.
pub(super) fn normalize(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::{normalize, resolve_citations};

    #[test]
    fn grouped_and_adjacent_handles() {
        let (body, order) = resolve_citations("A [s1, s3] B [s1] C [s7][s3] D");
        assert_eq!(order, ["s1", "s3", "s7"]);
        assert_eq!(body, "A [1, 2] B [1] C [3][2] D");
    }

    #[test]
    fn bare_numbers_untouched_and_case_insensitive() {
        let (body, order) = resolve_citations("see [1] and [S2]");
        assert_eq!(order, ["s2"]);
        assert_eq!(body, "see [1] and [1]");
    }

    #[test]
    fn no_handles_left_verbatim() {
        let (body, order) = resolve_citations("plain [note] text");
        assert!(order.is_empty());
        assert_eq!(body, "plain [note] text");
    }

    #[test]
    fn normalize_collapses_space_and_case() {
        assert_eq!(normalize("  Boule  de\nFEU "), "boule de feu");
    }
}
