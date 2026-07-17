# Enki — Spécifications

Librairie **Agentic RAG** en Rust. Deux facettes : **indexing** et **inference**.

## Contraintes

- **Rust + Tokio.**
- **Provider fourni par la config** — aucune opinion sur le provider dans le code.
- **LLM : `rust-genai` partout + boucle agentique maison.** `rig` écarté du chemin critique.
- **Domaine fourni par la config** (description + corpus + collections) — même binaire domain-agnostic → « assistant Control-M » ou « assistant Campagne » sans code métier.

### Pourquoi genai + boucle maison (pas rig)

Les 3 contraintes fortes tapent sur ce que rig impose ; son seul apport (boucle clé en main) est ce qu'on sait écrire.

- **Outils homogènes cross-provider** : genai normalise le JSON Schema par provider (Gemini/Vertex : `const`→`enum`, sanitize `additionalProperties`, strip keywords rejetés, accepte `schemars`) + `ToolChoice` neutre. rig passait le schéma quasi tel quel → cassait sur la profondeur Gemini.
- **Prompt cache** : genai expose `CacheControl` sur `ChatOptions` (requête) ET `MessageOptions` (breakpoint par message, TTL 5m/1h/24h). rig ne surface pas le breakpoint au niveau Agent.
- **Évolutivité** : genai ne donne que `ChatRequest{system, messages, tools, options}` → on possède la boucle, le system prompt, la rotation d'outils, les stop conditions.

**Non négociable : on possède le vecteur de messages** (cache + contrôle).

## Architecture retrieval

**Le retrieval est un RETOUR D'OUTIL, jamais du contexte pré-empilé.** (RAG statique empile dans le préfixe → change à chaque requête → cache mort. Agentique étend le suffixe → cache-friendly.)

### Traits (provider-agnostic)

| Trait | Rôle |
|---|---|
| `Embedder` | `embed(&[String]) -> Vec<Vec<f32>>`, `dims()` |
| `Index` | écriture : `upsert`, `delete` |
| `Retriever` | lecture, **par collection/source** |
| `GraphStore` | navigation outline (unifié, deux types de nœuds) |
| `Reranker` | séparé, **pluggable** — pas câblé en dur, ajouté quand il paie sa latence |

Une fine couche **fusion** au-dessus des Retrievers (dans l'outil `search`) merge en **RRF + tier-boost**. Composition interne à l'outil : `Dense ⊕ BM25 → RRF → rerank → top-n serrés`.

### Deux profils (pas un 2×2)

- **Local / zéro-infra** : brute-force (dense) + Tantivy (lexical).
- **Prod** : Qdrant (dense+sparse+fusion natifs dans une collection).

Les deux miroitent la même forme : un index/collection par tier + même fusion côté client.

### Priorité des sources : Livre < Campagne < Expert

- Le **tier** = métadonnée d'indexation (fait sur la donnée). La **priorité** = search (ranking/boost + conflit). **Jamais** un score figé dans l'index (le trust bouge, la priorité est contextuelle).
- **Une collection par tier** : cycles de vie disjoints (Livre immuable / Expert volatile), tuning indépendant, trust-gating = quelles collections on interroge.
- Règle : **partitionner par ce qui est stable + structurellement différent (tier)** ; **filtrer par ce qui est dynamique + fin (`trust_status`, `tags`, `type`, `status`)**. → 3 collections, pas 9.
- Qdrant ne joint pas les collections : `search` interroge les N en parallèle (Tokio) puis fusionne. **RRF nécessaire** (scores inter-collections non comparables ; RRF est rank-based). Le tier-boost s'y injecte.
- Même modèle d'embedding entre tiers (embed la query 1×) sauf raison forte. Chunking/params différents = OK.
- **Conflit : le search ordonne et étiquette, l'agent arbitre.** Règle dure au search seulement pour le cas net (correction Expert *endorsée* + arête `contradicts` explicite → démote/masque le chunk contredit). Jamais de suppression silencieuse sinon.

### Search service (composition modulaire)

Le search est une **composition** : dense (vecteur) + lexical (BM25) + rerank. Piège : certains backends (Qdrant) font dense+lexical **dans le même index** → une abstraction à deux slots fixes (`bm25`/`vector`) fuit (noop BM25 pour Qdrant). Solution : **une seule trait `Retriever`** = source de candidats, qui déclare sa `Modality` (Dense / Lexical / **Hybrid**). Un backend hybride = **une** source (jamais de noop).

Fusion à **deux axes** (à ne pas confondre) :
- **Modalité** (dense+lexical) = signal de **pertinence** → RRF (rank-based, fusionne des scores non comparables), **dans** une collection.
- **Tier/collection** (corpus < instance < expert) = signal d'**autorité** → tier-fuser, **entre** collections.

Structure :
```
Collection { name, tier, retrievers: Vec<Arc<dyn Retriever>> }   // Qdrant→1 retriever ; local→2 (brute+tantivy)
SearchEngine { embedder, collections, modality_fuser (RRF), tier_fuser, reranker: Option }
  search(text, k, filters): embed 1× → par collection [retrievers concurrents → RRF modalité] → tier-fuse → rerank → top-k
```
- « Plusieurs instances du retriever » = plusieurs instances d'**une** trait (une par collection), pas plusieurs traits.
- Embed de la query **une fois** dans l'engine → `Query { text, dense, k, filters }` fanout. Même modèle qu'à l'indexation.
- Config pilote l'engine : `dense: brute-force|qdrant`, `bm25: none|tantivy|qdrant`, `rerank: none|fastembed`. Le resolver **collapse** dense+lexical=même Qdrant en un retriever hybride. Chaque collection peut avoir son propre profil.
- `tier_fuser` **volontairement bête** (RRF simple) tant qu'on n'a pas de données de référence — pas de tuning prématuré des poids d'autorité.
- Le search renvoie `Vec<Scored>` ; le registre de handles reste dans la couche outil de l'agent (le search reste agnostique).

## Corpus & pivots

Deux pivots distincts, un seul `GraphStore` (deux types de nœuds) :

- **Livre = page-oriented** : `data/Vampire_La_Mascarade_5e_e_dition/` (439 pages). Markdown/page + `anchors.json` (bbox spatiale + `md_range`→offsets). Les « voir page 117 » = arêtes à extraire.
  - **`document.manifest.json`** = outline hiérarchisé : 173 sections (`title`, `level`, `page_start/end` = recouvrement, `content`). C'est l'unité de chunking retenue (pas la page) ET le futur `GraphStore`.
  - **Contextual embedding** : embarquer `title_path + content` (ex. « GANGREL > Disciplines »), pas `content` seul. Les titres se répètent à l'identique entre sections (Disciplines/Fléau par clan) → le chemin de titres désambiguïse. Les titres portent de l'info que la page ne restitue pas.
- **Campagne = entity-oriented** : `data/Campagne` (vault Obsidian VtM, ~95 md). Chaque fichier = un nœud ; frontmatter riche (`id`, `semantic_id`, `type`, `name`, `tags`, `status`, `updated`) + graphe de relations gratuit (`participants: [semantic_id…]` + `[[wikilinks]]`). GraphRAG-ready.

### Expert Knowledge = machine à états de confiance

Contribution (`Correction` / `Addition` / `Note`), statut `draft → public note → endorsed`. Orthogonal au retrieval : filtre + signal d'autorité. Non-endorsé n'influence pas l'agent (mentionné comme non vérifié). L'arête « corrige/contredit un document » est indexée ; son application est au search.

## System prompt — préfixe caché en couches

```
Cœur agentique universel   → identique à TOUTES les librairies, cache cross-chat, TTL 1h
[breakpoint cache 1]
Segment librairie          → {{LIBRARY_SCOPE}} + {{TIER_DEFINITIONS}}, statique/librairie, TTL 5m
[breakpoint cache 2]
Conversation               → messages + retours d'outils, dynamique
```

- Split justifié : 2 use cases sous **la même clé/org** → le cœur universel est la ligne de cache la plus chaude, réchauffe même les librairies peu sollicitées. Bénéfice **cross-chat**. (Collapser seulement si 1 librairie = 1 clé isolée.)
- **Library bound at chat creation** : pas de routage dynamique, retrieval tape toujours les collections de cette librairie.
- **Description = frame, pas contenu** (une ligne de scope, jamais du savoir qui périmerait). Le savoir vient du retrieval. Les house-rules sont du **corpus**, pas du prompt.
- Anatomie du cœur (6 sections) : 1. Rôle — 2. **Contrat d'ancrage** (LE levier qualité, override tout) — 3. Politique d'outils/boucle (1er `search` forcé, re-chercher si minces, **stop conditions explicites** + budget d'itérations) — 4. Priorité & arbitrage (**surfacer**, jamais résoudre en silence ; endorsed > note) — 5. Format & citations — 6. Incertitude/refus/hors-scope.
- Prompt en anglais (tokenisation/cache), sortie « in the user's language ». Few-shot quasi gratuit (caché) → format de citation + surfaçage de conflit.

## Contrats d'outils

### Trois espaces d'id (ne pas confondre)

| Id | Producteur/consommateur | Portée | Rôle |
|---|---|---|---|
| `chunk_id` (uuid ou `doc#block`) | store / système | durable | liens, logs, éval, dedup, arête `contradicts` |
| **handle `s7`** (index de contexte) | le **modèle** | la conversation | ce que le LLM cite / passe à `open`/`neighbors` |
| `call_id` (fourni par genai) | la boucle | un tool call | corréler requête↔réponse dans le trace |

**Le modèle n'émet jamais un uuid** (coût tokens + transposition). Il ne voit que des handles courts. Le **registre** = bijection `chunk_id ↔ handle`, scoped conversation : **accumulé, jamais renuméroté, dedup par `chunk_id`** (même chunk ressorti = même handle).

### Set d'outils

SoTA = jeu minimal + max de prompt. MVP à **2 outils** (`search` + `answer`) ; ajouter `open`/`neighbors` quand la navigation débloque des questions mesurées.

**`search(query, filters?)` → retourne par passage :**
```
{ handle, chunk_id, text, provenance, tier, trust_status }
```
`filters` porte le gate `trust_status` / `tier` / `tags`. Alimente le registre.

**`answer(...)` — terminaison + citations structurées :**
```jsonc
answer({
  answer_markdown: string,          // prose, langue de l'user, handles inline: "… un jet [s7]."
  citations: [ { passage: string, quote: string } ],   // required, minItems 1 ; quote = VERBATIM
  coverage:  { answered: boolean, gaps: string | null },
  conflicts: [ { summary, authoritative, conflicting } ]  // handles, optionnel
})
```
- **Citer par référence (handle), jamais recopier la provenance** → zéro hallucination de provenance, schéma plat (safe Gemini), payload minuscule. Résolution `handle → chunk_id → provenance` côté app.
- `quote` verbatim = levier anti-hallucination ; vérifié en fuzzy match côté app, citations non collantes flaguées.
- Pas de `confidence: float` (mal calibré) — `coverage` en prose est plus honnête.
- Rendu : renumérote `[s7]→[1]` pour l'affichage.

**Résolution de provenance côté app, par type de source :**

| Source | handle résout vers | rendu |
|---|---|---|
| Livre (paginé) | `{doc, page, page_label, bbox, md_range}` | « Vampire 5e, p.43 » + région cliquable |
| Campagne (entité) | `{semantic_id, type, name, file}` | « PNJ : Carmilla la Sybille » + lien nœud |
| Expert | `{contribution_id, author, trust_status, contradicts}` | « Correction (endorsée) — auteur » |

### Sémantique de terminaison (boucle)

- `StopReason = ToolCall(answer)` → finaliser (résoudre handles, vérifier quotes, rendre).
- `ToolCall(search|open|neighbors)` → exécuter, continuer.
- `Completed` + texte simple → non-réponse-savoir (clarification / refus / chitchat), laisser passer.

## POC (périmètre)

Prouver la colonne vertébrale, couper le reste.

**Fait (état actuel) :** code **anglais + agnostique du domaine** (scope injecté via `ENKI_LIBRARY_SCOPE`, `Tier`=u8, aucune notion RPG dans le code). **Séparation nette** : `cargo run -- index` (pipeline manifest → sections → chunks → embeddings persistés, à la demande) vs `ask "..."` (charge le cache, échoue si index absent). Chunking **section-based** via `document.manifest.json` + **contextual embedding** (title_path). genai config-driven (2 clients ollama, `ServiceTargetResolver` pour host custom). `Embedder`/`Retriever` derrière traits. Boucle agent `search`+`answer`, registre de handles, résolution provenance (title_path + pages) + vérif quotes. **911 chunks** indexés (bge-m3), testé bout-en-bout.

**Finding :** modèles locaux (gemma4:31b) répondent en **prose** avec handles inline plutôt que via l'outil `answer` → chemin prose-as-message implémenté (extraction `[sN]`, résolution, flag handles hallucinés).

**Coupé pour le POC :** Qdrant, BM25/Tantivy, rerank, graphe/navigation (`open`/`neighbors`), tiers multiples, machine de confiance Expert, Campagne. **À venir (demandé) :** meilleure abstraction du *search service* et de la *construction du client genai*.
