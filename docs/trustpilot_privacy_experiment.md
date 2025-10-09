# Trustpilot Privacy Experiment Design

## Dataset Snapshot
- Source: `bmark/trustpilot_reviews_2005.csv` with columns `category`, `company`, `description`, `title`, `review`, `stars`.
- Individuals: reviewer + company pairings, expressed through `review` text and `title`.
- Background attributes available to an attacker: public-facing `company` information, marketing `description`, aggregate ratings, and potential external corpora containing user-authored reviews.

## Scenario A — Company Linkage via Descriptions
- **Background knowledge:** Public marketing copy scraped from company websites or catalogs.
- **Anonymized release:** Suppress explicit `company` field, retain `description` so downstream models can predict company-level aggregates (e.g., mean star rating).
- **Risk evaluation:** A re-identification model could use near-verbatim matching between the anonymized `description` and publicly indexed text to recover the company identity.
- **Limitations for privacy testing:** The privacy subject is the company rather than the reviewer; even perfect linkage reveals little about a specific person. Utility-focused anonymization (predicting mean stars) provides weak evidence about human-level leakage.

## Scenario B — Individual Reviewer Linkage via Reviews
- **Background knowledge:** Collections of user-generated content (e.g., other review platforms, social media posts) containing similar writing styles, sentiments, and product references.
- **Anonymized release:** Remove direct reviewer identifiers, possibly perturb `review` text, and preserve `stars` for downstream tasks such as rating prediction.
- **Re-identification task:** Train a matcher that embeds attacker-side posts and anonymized reviews, ranks candidate identities per review, and measures top-k success. The model can leverage lexical cues, sentiment, product mentions, and stylistic fingerprints to recover individuals.
- **Why this is the more plausible privacy test:** Regulatory and ethical frameworks prioritize protecting people, not companies; reviewing behavior can expose health, financial, or location details about the reviewer; and cross-platform writing style matching is a realistic, well-documented attack vector. Testing anonymization robustness against reviewer linkage directly quantifies that risk.

## Comparative Summary
- Company linkage quantifies exposure of corporate identities but says little about personal privacy.
- Reviewer linkage evaluates the protection of individuals, aligning with privacy-by-design objectives and most legal risk assessments.
- Even if reviewer matching is harder technically, its stakes are higher, and success/failure provides actionable insights for improving anonymization.

## Recommended Experiment Roadmap
1. Split dataset into attacker background, anonymized release, and evaluation folds without overlapping reviewer-company pairs.
2. Implement anonymization variants for the `review` text (token masking, paraphrasing, differential privacy noise) while keeping rating utility intact.
3. Train ranking-based re-identification models (TF-IDF + cosine baseline, contrastive neural encoder) that operate on review text.
4. Measure top-1/top-5 accuracy and mean reciprocal rank to compare privacy leakage across anonymization strategies.
5. Report privacy risk per reviewer cohort (e.g., by `stars` or text length) to highlight where protections need reinforcement.
