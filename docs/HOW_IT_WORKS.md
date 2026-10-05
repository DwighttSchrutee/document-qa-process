# How the Document Q&A system works

This document explains, in plain English, everything that happens inside the system: how documents are
prepared, and what happens from the moment someone asks a question until the answer appears with its
sources. It also lists what has been built, which problems were found and fixed, and what is planned next.

Everything runs on one Mac. No document, question or answer is sent to any outside service.

---

## Contents

1. [The big picture](#1-the-big-picture)
2. [Before any question: preparing the documents](#2-before-any-question-preparing-the-documents)
3. [What happens when a user asks a question](#3-what-happens-when-a-user-asks-a-question)
4. [Understanding the question](#4-understanding-the-question)
5. [The search methods](#5-the-search-methods)
   - [Exact identifier lookup](#51-exact-identifier-lookup)
   - [Keyword search (and how it relates to BM25)](#52-keyword-search-and-how-it-relates-to-bm25)
   - [pg_trgm: fuzzy spelling match](#53-pg_trgm-fuzzy-spelling-match)
   - [BGE-M3 dense embeddings: search by meaning](#54-bge-m3-dense-embeddings-search-by-meaning)
   - [Clause-number search](#55-clause-number-search)
6. [Combining the searches: Reciprocal Rank Fusion (RRF)](#6-combining-the-searches-reciprocal-rank-fusion-rrf)
7. [The reranker](#7-the-reranker)
8. [Selecting and expanding the evidence](#8-selecting-and-expanding-the-evidence)
9. [Generating the answer](#9-generating-the-answer)
10. [Checking the answer](#10-checking-the-answer)
11. [How sources, pages and SRO numbers are attached](#11-how-sources-pages-and-sro-numbers-are-attached)
12. [Special cases](#12-special-cases)
13. [Safeguards against wrong or unsupported answers](#13-safeguards-against-wrong-or-unsupported-answers)
14. [The components and why each one is used](#14-the-components-and-why-each-one-is-used)
15. [What has been implemented](#15-what-has-been-implemented)
16. [Problems found and fixed](#16-problems-found-and-fixed)
17. [Known limitations today](#17-known-limitations-today)
18. [Planned improvements](#18-planned-improvements)
19. [Glossary](#19-glossary)

---

## 1. The big picture

There are two separate journeys:

**Journey A – preparing a document (happens once per document, when it is uploaded)**

```
PDF / Word / Excel / text
   │
   ├─ 1. fingerprint the file (skip it if it is already in the system)
   ├─ 2. read every page (OCR + the PDF's own text, merged)
   ├─ 3. quality gate: mark every row/line PASS, WARN or HOLD
   ├─ 4. cut the content into "chunks" (one table row, or one or more legal clauses)
   ├─ 5. find notices (SROs), their dates, and which notices amend or cancel which
   └─ 6. store everything in the database: text, numbers in normalised form, and a "meaning vector"
```

**Journey B – answering a question (happens every time someone asks)**

```
Question
   │
   ├─ 1. understand it: find HS codes, SRO numbers, dates, clause numbers, the language
   ├─ 2. search in several ways at once:
   │       exact codes · keywords · fuzzy spelling · meaning (vectors) · clause numbers
   ├─ 3. combine the result lists into one ranking (RRF)
   ├─ 4. re-read the best ~30 candidates carefully (reranker)
   ├─ 5. pick the evidence: best passages + table explanations + notice cards + amendments
   ├─ 6. the language model (qwen3:8b) writes the answer, citing every fact as [1], [2] …
   ├─ 7. check the answer: every number must appear in the cited evidence
   └─ 8. show the answer with clickable sources (document, page, cropped page image)
```

A simple example of the whole thing:

> **Question:** "What is the minimum value for shampoos under HS 3305.10.00?"
>
> - Step 1 finds the code `3305.10.00` and turns it into the key `33051000`.
> - Step 2's exact lookup finds every stored row containing `33051000`: one row in SRO 202/2024 (minimum
>   values) and one in the FY2025-26 tariff rate sheet.
> - Steps 3–5 put those rows at the top and add the sentence of SRO 202/2024 that explains what its table means.
> - Step 6: *"The minimum value for shampoos under HS 3305.10.00 is USD 4.00 per kg [1]."*
> - Step 7 confirms that "4.00" and "3305.10.00" appear in passage [1].
> - Step 8 shows `[1] nbr_06.pdf p5` – clicking it shows the cropped table row from page 5.

---

## 2. Before any question: preparing the documents

Answers can only be as good as what was extracted from the documents, so this part matters a great deal.

### 2.1 Fingerprinting (no duplicates)

Each uploaded file gets a SHA-256 fingerprint (a unique code computed from its bytes). If the same file is
uploaded again – even under another name – it is recognised and skipped. In the sample folder, four files
were exact duplicates of others (e.g. `nbr-01.pdf` = `nbr-03.pdf`).

### 2.2 Reading the pages

The PDFs here have a special problem. They contain a hidden **text layer** (the text you get when you copy
from a PDF), but:

- **Bangla in the text layer is wrong.** The font used (Nikosh) has a broken internal table, so
  "তারিখ" comes out as "তাবরখ", "বিভাগ" as "ববভ্াগ". About 4 in 10 Bangla words are wrong.
- **Numbers, codes and English in the text layer are exactly right.**

So the system uses two readers and merges them:

| Reader | Good at | Bad at |
|---|---|---|
| **Surya OCR** (reads the page *image*, runs on the Mac's GPU via llama.cpp) | Bangla text (≈0.65% character errors) | Sometimes misreads or *invents* numbers; sometimes skips, merges or repeats table rows |
| **PDF text layer** | Exact numbers, codes, English | Garbled Bangla |

**The merge** (file `ingest/fusion.py`) works block by block:

1. Surya's text and the text layer's words for the same area are lined up side by side.
2. **Bangla words are always taken from Surya.**
3. **Numbers, codes and English words are always taken from the text layer.** If Surya says `87033121` and
   the text layer says `87033211`, the text layer wins – recorded as a **CORRECTION**.
4. A value the text layer has but Surya skipped is put back – a **RESTORED** value (whole skipped table
   rows come back as proper rows).
5. A value Surya produced that the text layer does not have is kept but flagged **UNCONFIRMED**.

Example of the merge at work (tariff sheet page 1): Surya skipped 6 whole rows of frozen beef codes. The
merge rebuilt all 6 rows from the text layer, placing each word in the right column by its position on the page.

On the 13-page benchmark, Surya alone got 93.8% of HS codes right and **invented 12 codes**; with the merge it
is **100% correct with 0 invented**, and 192/192 tariff rate rows are exact.

**Scanned PDFs** (no text layer, e.g. `test_02.pdf`, a Gazette scan) use **Tesseract** as the second reader
instead. A number is only accepted when both engines read the same value. Two extra rules help here:

- Tesseract often reads the Bangla digits ৪ ৭ ০ as the Latin look-alikes 8 9 0; that counts as agreement.
- Legal Bangla writes numbers twice: "৯০ (নব্বই)", "২,০০০/- (দুই হাজার)". When the words in brackets say the
  same number, the number confirms itself.

### 2.3 Quality gate

Every page, table row and text line gets a status:

| Status | Meaning | Searchable? |
|---|---|---|
| **PASS** | Checked and fine | Yes |
| **WARN** | Probably fine, but something could not be verified | Yes – answers citing it say "could not be verified against the document" |
| **HOLD** | Too risky | **No**, until a person reviews it on the Review page |

What causes HOLD:

- A **number** that could not be confirmed (UNCONFIRMED) → only that row/line is held.
- A row whose cells don't fit their columns (a shifted row, e.g. "HS Code: Tyre of a kind used on bicycles |
  Unit: 3.00") → that row is held.
- **Whole page** held only for page-wide problems: more than 2% of its numbers needed correcting, or at least
  30% of its rows are held, or its Bangla is mostly not real words, or a page with visible content but
  almost no text.

The Review page shows the page image next to the extracted text with the problem values highlighted; a
reviewer can **accept**, **correct** or **reject** each one. Decisions are saved and re-applied whenever a
document is re-processed, and the document is re-indexed immediately.

### 2.4 Chunking: cutting documents into searchable pieces

The search engine does not search whole documents; it searches **chunks**.

- **Table rows**: one chunk per row, written with its column headers, e.g.
  `Heading: 69.07 | HS Code: All HS Code | Description of Goods: Ceramic Tiles > 600 mm X 600 mm | Unit: m2 | Minimum Value USD/Unit: 6.00`.
  - Headers are repeated on every row and carried across pages.
  - Dash hierarchies are resolved: "--- Other" becomes "-- Turkeys : > --- Other".
  - Rows under a group header (like the tile sizes under "Ceramic Tiles") carry the group's name.
- **Legal text**: split at headings and clause markers (১।, (১), (ক), 1., (a)). Whole clauses are kept together
  (up to about 450 tokens); a clause is only cut if it is longer than that, and the continuation repeats the
  clause's first line so it still makes sense on its own.
- Every chunk remembers its **document, page(s), position on the page, section, clause number, SRO, dates
  and quality status**.

### 2.5 Notices, dates and links

A file can contain one notice (SRO) or many. For each notice the system records:

- its reference ("এস.আর.ও. নং ২১৩-আইন/২০২৪/৬৫/কাস্টমস" → key `213/2024`),
- its own title ("সরাসরি রপ্তানিমুখী পোশাক শিল্প প্রতিষ্ঠান … বিধিমালা, ২০২৪"),
- the issue date, the effective date and expiry date when stated – plus **the exact sentence** that states
  them (e.g. "(৩) ইহা অবিলম্বে কার্যকর হইবে।" = takes effect immediately),
- which other notices it **amends**, **rescinds** (রহিত) or **refers to** – by SRO number or by title.

Example: `test_02.pdf` (SRO 241/2024) says it amends "…পোশাক শিল্প প্রতিষ্ঠান … বিধিমালা, ২০২৪" by title.
The system matches that title to SRO 213/2024 (`nbr_10.pdf`) and stores the link
"241/2024 amends 213/2024".

### 2.6 Storing

Everything goes into PostgreSQL (a database running only on this Mac):

- `chunks` – the text of each chunk, a normalised copy for searching, and its meaning vector (BGE-M3).
- `identifiers` – every code-like value, normalised: `8714.10.90`, `87141090` and `৮৭১৪.১০.৯০` are all stored
  as `87141090`; also the 4- and 6-digit prefixes, notice references (`REF:169/2024`) and dates.
- `segments` (notices) and `doc_links` (amends / rescinds / refers).
- `pages`, `extraction_changes`, `review_queue`, `qa_log` (every question asked), evaluation results.

Today the index holds 9 documents, 8,178 chunks and 23,912 identifiers.

---

## 3. What happens when a user asks a question

```
"Can a spinning mill import ring travellers and spindles at reduced duty?"
        │
        ▼
 (1) Understand         → language: English; no codes; no SRO number; no clause number
        │
        ▼
 (2) Search, 5 ways      exact identifiers · keywords · fuzzy spelling · meaning · clause numbers
        │                (each returns up to 30 chunks, in its own order)
        ▼
 (3) RRF                 → one combined ranking (up to 50 candidates)
        │
        ▼
 (4) Reranker            → the best 30 are re-read and scored for how well they answer the question
        │
        ▼
 (5) Evidence            → best ~6 passages within ~2,500 tokens
        │                  + the sentence that explains each table used ("TABLE-1 … exempt from CD above 1%")
        │                  + notice cards for any SRO named in the question
        │                  + amendments of the notices used
        ▼
 (6) qwen3:8b            → writes a short answer, citing [1], [2] …
        │
        ▼
 (7) Checks              → every number in the cited passages? citations real? no template text?
        │                  if not: regenerate once; if still wrong: remove the bad sentence + warn
        │                  if "not found": search again, wider, before saying so
        ▼
 (8) Answer + sources    → clickable [n] → passage, cropped page image, full page
```

Typical time on this Mac: 20–30 seconds for English table questions, 45–50 seconds for Bangla rule questions,
about 2 minutes when the system has to search twice and then says "not found".

---

## 4. Understanding the question

File: `answer/understand.py`, `retrieve/search.py`.

The system does **not** try to answer yet. It extracts the parts of the question that can be searched exactly:

| What | Example in the question | Becomes |
|---|---|---|
| HS / other codes | `8714.10.90`, `87141090`, `৮৭১৪.১০.৯০` | `87141090` (any form matches) |
| Code prefixes | `8714` | every code starting with 8714 |
| SRO numbers | `SRO 169/2024`, `এস.আর.ও. নং ১৬৯-আইন/২০২৪/২১/কাস্টমস` | `REF:169/2024` |
| SRO ranges / lists | `SRO 210-212`, `SROs 169, 202 and 210` | `REF:210/2024`, `REF:211/2024`, `REF:212/2024` … |
| Dates | `29 May, 2024`, `২৯ মে, ২০২৪` | `DATE:2024-05-29` |
| Clause numbers | `বিধি ১২`, `rule 7`, `section 3` (not sub-rules) | clause 12, 7, 3 |
| Language | Bangla letters vs Latin | `bn` / `en` |

Bangla digits are turned into ASCII digits for matching, but the original is kept for display.

In the first search round this is done with simple, instant pattern matching. qwen3:8b is only asked to
rewrite the question (3–4 versions, one in the other language) in the **second** round, because each such
call costs 5–10 seconds on this machine.

---

## 5. The search methods

Each method returns its own list of up to 30 chunks. They are good at different things, which is why
several are used together.

### 5.1 Exact identifier lookup

**What it does:** looks up the normalised codes, SRO references and dates from the question in the
`identifiers` table.

**Example:** "VAT for ৮৭০৩.৩২.১১?" → key `87033211` → the tariff-sheet row for 87033211 is found with
certainty, no matter how the code was written.

**When it is used:** whenever the question contains a code, SRO number or date. Hits from this method are
always ranked above fuzzy hits and are never dropped from the evidence in favour of fuzzier matches.

**Why:** codes are the most important facts in these documents and must never be "approximately" right.

### 5.2 Keyword search (and how it relates to BM25)

**BM25 in plain English.** BM25 is a classic way of ranking documents by keywords. It gives a passage a higher
score when:

- it contains the question's words, especially **rare** words (a match on "অব্যাহতি" counts more than a match
  on "এবং", because "এবং" is everywhere),
- it contains them **several times** (with diminishing returns),
- it is **not unusually long** (a short passage with the word is more focused than a long one).

**What this system actually uses:** PostgreSQL's built-in full-text search (`to_tsvector('simple', …)`), ranked
with `ts_rank_cd`. It is the same family of idea – score passages by how many of the question's words they
contain and how close together – but it is **not exactly BM25** (it does not weight rare words as strongly).

- The `simple` configuration is used because there is no Bangla word-stemmer; it matches whole words.
- Text is normalised first: Bangla digits → ASCII, lower case, Unicode cleaned.
- Words are joined with OR, so a passage containing some of the question's words is still found.

**Example:** "ইউডি যাচাই কমিটি কতদিন পরপর সভা করবে?" → words `ইউডি | যাচাই | কমিটি | কতদিন | পরপর | সভা | করবে`
→ the passage containing "ইউডি যাচাই … কমিটি … সভায়" scores high.

**When it is used:** on every question, for the original question (and, in the second round, for each
rewrite).

**Why:** meaning-based search can miss rare exact terms (form names, committee names, legal terms); keyword
search catches them.

### 5.3 pg_trgm: fuzzy spelling match

**What it is:** a PostgreSQL extension that compares text by **trigrams** – overlapping groups of three
characters. "spindle" → `spi, pin, ind, ndl, dle`. Two words that share most of their trigrams are similar
even if they are not spelled identically.

**Example:** the document says "Spindles/Bloser" or has an OCR slip like "Reconditoned"; the question says
"spindle" or "Reconditioned". Keyword search needs exact words and would miss; trigram similarity still
finds them. The system uses *word similarity* with a threshold of 0.45 (roughly: at least ~45% of a question
word's trigrams appear in a word of the passage).

**When it is used:** on every question, alongside keyword search.

**Why:** OCR text and real-world spelling are never perfect; this catches near-misses.

### 5.4 BGE-M3 dense embeddings: search by meaning

**What it is:** BGE-M3 is a model that turns a piece of text into a list of 1,024 numbers – a "meaning
vector". Texts with similar meaning get similar vectors, **even in different languages**.

**Example:**
- "How long can goods stay in a bonded warehouse?" and
- "ওয়্যারহাউসে পণ্য সংরক্ষণের মেয়াদ হইবে অনধিক ২৪ (চব্বিশ) মাস"

share almost no words, but their vectors are close, so the Bangla clause is found for the English question.

**How it is used:**
- When documents are indexed, every chunk is embedded once (together with its SRO and section, so a row
  "knows" which notice it belongs to). Embeddings are cached, so re-indexing only re-embeds changed chunks.
- When a question arrives, the question is embedded and compared with every chunk using **cosine
  similarity** (how closely two vectors point in the same direction).
- The comparison is **exact** (every chunk is checked; ~20 ms for 8,000 chunks). See problem #24 in section 16
  for why the faster approximate index (HNSW) was turned off.

**Where it runs:** locally in Ollama (`bge-m3`), on the Mac's GPU.

**When it is used:** every question.

**Why:** it handles paraphrases, descriptions instead of codes ("crude palm oil" instead of 1511.10.00), and
English↔Bangla questions.

### 5.5 Clause-number search

If the question says "বিধি ১২", "rule 7" or "section 3", every chunk whose clause number is 12 / 7 / 3 is
listed, most similar to the question first. Sub-rules ("উপ-বিধি ১") are deliberately ignored, otherwise every
document's rule 1 would match.

---

## 6. Combining the searches: Reciprocal Rank Fusion (RRF)

Each search method produces its own ranked list, with scores that cannot be compared with each other (a
keyword score and a vector similarity are different scales). RRF ignores the scores and uses only the
**positions**:

> score of a chunk = sum over all lists of 1 / (60 + its position in that list)

**Example:**

| Chunk | Keyword list | Meaning list | Trigram list | RRF score |
|---|---|---|---|---|
| A | 1st | 3rd | – | 1/61 + 1/63 = 0.0323 |
| B | – | 1st | – | 1/61 = 0.0164 |
| C | 5th | 2nd | 4th | 1/65 + 1/62 + 1/64 = 0.0472 |

C wins: it was found by **three** methods, even though it was first in none. Agreement between methods is a
strong signal.

On top of RRF, **exact code / SRO hits are always placed first**, whatever their RRF score.

The top 50 candidates go to the next stage.

---

## 7. The reranker

**What it is:** `BAAI/bge-reranker-v2-m3`, a "cross-encoder". Unlike the embedding model, which encodes the
question and the passage separately, the reranker reads **the question and one passage together** and
outputs a single relevance score. This is slower but much more precise.

**Example:** for "Can a spinning mill import ring travellers at reduced duty?":
- "HS Code: 8448.33.00 | Description: Ring Travellers" → high score,
- "HS Code: 8448.39.00 | Description: Cutting Blade" → low score, even though it is from the same table and
  looked similar to the vector search.

**How it is used:** the top 30 RRF candidates (plus any exact hits beyond them) are scored, and the evidence
is chosen from the highest scores. Passages are truncated to 512 tokens for scoring.

**Where it runs:** on the CPU. On this 16 GB Mac, running it on the GPU next to the answering model sometimes
stalled answers for minutes (see section 16).

---

## 8. Selecting and expanding the evidence

The language model can only read a limited amount of text quickly, and Bangla is expensive (about 1.2
"tokens" per Bangla character, three times English). So the evidence is carefully budgeted.

**Step 1 – pick the best passages.** Exact code/SRO hits first, then by reranker score, up to 6 passages and
about 2,500 tokens. At least the 2 best passages are always kept, even if large.

**Step 2 – add what a passage needs to make sense:**

| Addition | Why | Example |
|---|---|---|
| **Table explanation** | A table row alone does not say what applies to it | Row "TABLE-1: 8448.33.00 Ring Travellers" brings the sentence "(ক) TABLE-1 … কাস্টমস শুল্ক … ১ (এক) শতাংশ এর অতিরিক্ত … এবং সমুদয় মূল্য সংযোজন কর ও সম্পূরক শুল্ক …" (= CD above 1% and all VAT/SD exempt) |
| **Notice cards** | When the question names SROs, it needs their key facts, not their whole first page | "Notice SRO 213/2024: … বিধিমালা, ২০২৪ · Issued 2024-05-29 · Effective from 2024-05-29 — the notice says: '(৩) ইহা অবিলম্বে কার্যকর হইবে।'" (about 100 tokens instead of 1,300) |
| **Amendments** | A rule may have been changed by a later notice | A passage from SRO 213/2024 brings one passage of SRO 241/2024, which amends it (only for notices that are clearly relevant, one passage per amending notice, within budget) |

**Step 3 – label each passage** with its document, page, SRO, issue and effective dates, section, clause
number, and "UNVERIFIED" if the quality gate gave it a warning. The model sees these labels.

---

## 9. Generating the answer

**Model:** `qwen3:8b`, running locally in Ollama, with a context window of 8,192 tokens, temperature 0 (no
randomness), and "thinking" off by default (the chat has a *reasoning* switch for harder questions; it is
slower).

**Instructions given to the model** (file `answer/prompt.py`), in short:

1. Answer only from the numbered passages, and cite every fact as [n].
2. Copy numbers, codes, dates and rates exactly; never calculate, convert or guess.
3. If sources disagree, give each value with its document and date and say which is newer.
4. Say so when a cited passage is marked unverified.
5. If the passages do not contain the answer, start with `[NOT FOUND]` and name the closest passages.
6. Read legal Bangla for meaning; a small glossary is included (রহিত = rescinded, তবে শর্ত থাকে যে = provided
   that, অব্যাহতি = exemption, শুল্কায়নযোগ্য মূল্য = assessable value, অবিলম্বে = immediately, …).
7. Answer in the question's language; give the direct answer first, then only the conditions that matter.
8. A table row means what the notice's text says about that table.

The answer is streamed to the screen word by word.

---

## 10. Checking the answer

File: `answer/verify.py`, `answer/pipeline.py`.

| Check | What happens if it fails |
|---|---|
| **Every number** in the answer (Bangla or ASCII digits, codes, dates, percentages) must appear – as a whole number, not part of another – in a cited passage or its label | The answer is regenerated once with the problem stated ("the value 4.50 does not appear in the passages you cited"). If it still fails, the sentences containing the bad value are removed and a warning is shown. |
| Every **citation** [n] must refer to a passage that exists | Regenerated; impossible citations are removed. |
| The answer must **cite something** | Regenerated. |
| No **template text** (e.g. "<one sentence …>") | Regenerated. |
| The answer says **not found** | Before accepting that, a **second, wider search** runs: 100 candidates, question rewrites in both languages, identifier-only search, neighbouring chunks of the best hits, linked notices. Only if that also fails does the user see "not found", together with the three closest passages. |
| A cited passage is **unverified** (quality gate WARN) | A warning is shown under the answer. |
| An amendment of a cited notice still has **held** lines | A warning says the answer may not reflect the latest amendment. |

Every question, the searches, the scores, the answer and the check results are saved in `qa_log`.

Example: the model once wrote "The minimum value is 4.50" when the passage said 4.00. The check caught 4.50,
the model was asked again, and the second answer said 4.00.

---

## 11. How sources, pages and SRO numbers are attached

Nothing is "remembered" by the model; every reference comes from the stored chunk:

- **Document and page:** each chunk stores its file name and page numbers.
- **Position on the page:** each table row and text line stores its bounding box (taken from the text layer's
  word positions). Clicking [n] in the chat shows a crop of exactly that area of the page image, plus a link
  to the full page.
- **SRO number and dates:** each chunk belongs to a notice (segment) with its reference, issue date and
  effective dates.
- **Clause number:** stored per chunk (e.g. `৬(৩)`).

The model is shown these as passage labels, and the answer's [n] markers are mapped back to them in the UI.

---

## 12. Special cases

### Exact lookups (codes, SRO numbers)
Handled by the identifier table: any written form of a code finds it; exact hits always lead the evidence.

### Semantic (meaning) questions
"What happens if raw materials are damaged by fire in the bonded warehouse?" contains no codes; the vector
search and reranker find the clause about waiver for damaged materials.

### Bangla and English
- Bangla digits ↔ ASCII digits are equivalent everywhere.
- BGE-M3 matches meaning across the two languages.
- The answer comes in the language of the question.
- Known weakness: an English question whose answer is a formal Bangla proviso is sometimes missed by the
  model even though the right passage is in its context (see section 17).

### Tables
- One chunk per row, always with headers.
- Group headers, dash hierarchies, merged "heading" cells and page continuations are resolved.
- The notice's sentence explaining the table is added to the evidence.
- Shifted or garbled rows are held by the quality gate.

### Legal clauses
- Chunks follow the clause structure (১।, (১), (ক)); clauses are not cut in the middle.
- Clause numbers in questions ("বিধি ১২") are searched directly.

### Amendments
- Detected from wording (সংশোধন, প্রতিস্থাপিত, amend, substitute; রহিত, rescind) by SRO number or by the
  amended notice's title.
- Amending passages are added automatically when the amended notice is relevant.

### Superseded (rescinded) notices
- The rescission is detected and stored as a link (e.g. SRO 202/2024 rescinds SRO 198/2023).
- **Not yet done:** marking the rescinded notice as superseded and down-ranking it in answers (planned, see
  section 18). Today it does not matter much because none of the rescinded notices are in the index.

---

## 13. Safeguards against wrong or unsupported answers

1. **Numbers come from the text layer**, not from OCR guesses, on every digital PDF page.
2. **Two-engine agreement** for numbers on scanned pages.
3. **Quality gate**: unconfirmed numbers, shifted rows and badly read pages are kept out of search until reviewed.
4. **Answers only from passages**, every fact cited.
5. **Number check** after generation, with one regeneration and removal as a last resort.
6. **Second search before "not found"**.
7. **Warnings** for unverified passages and for amendments with held content.
8. **Conflicts** must be shown with both sources and dates (instructed; not yet enforced by code).
9. **Full logging** of every question for later checking.

---

## 14. The components and why each one is used

| Component | What it does here | Why this one |
|---|---|---|
| **Surya OCR 0.22** (+ llama.cpp `llama-server`, Metal GPU) | Reads page images, especially Bangla | Best measured Bangla accuracy on these documents |
| **PDF text layer** (PyMuPDF) | Exact numbers, codes, English, word positions | Exact where it is not garbled; free and instant |
| **Tesseract 5** (`tessdata_best`, Bangla + English) | Second reader for scanned pages; Bangla word list for the gate | Independent engine, runs on CPU |
| **PostgreSQL 16** | Stores everything | One reliable local database |
| **pgvector** | Stores and compares the 1,024-number vectors | Vectors in the same database as the text |
| **Full-text search (`simple`)** | Keyword search | Built in; works for Bangla whole words |
| **pg_trgm** | Fuzzy spelling search | Catches OCR slips and spelling variants |
| **BGE-M3** (Ollama) | Meaning vectors | Multilingual (Bangla + English), long inputs (8k tokens) |
| **bge-reranker-v2-m3** (CPU) | Precise re-scoring | Multilingual, strong, runs locally |
| **qwen3:8b** (Ollama) | Writes the answer | Required by the brief; good multilingual 8B model that fits this Mac |
| **FastAPI** | The web server and API | Simple, streaming answers |
| **Vue 3** (local copy) | The web interface | No build step, nothing loaded from the internet |

---

## 15. What has been implemented

- Upload (PDF, Word, Excel, CSV, text) with duplicate detection and a background ingestion job.
- Surya + text-layer merge (digital PDFs), Surya + Tesseract vote (scanned PDFs), with number-in-words
  confirmation.
- Quality gate (PASS / WARN / HOLD) with a Review page (accept / correct / reject, persistent).
- Structure-aware chunking for tables and legal clauses; notice detection with dates and links.
- PostgreSQL index: text, normalised identifiers, vectors, notices, links.
- Hybrid retrieval: identifiers, keywords, trigrams, vectors, clause numbers → RRF → reranker.
- Evidence assembly: table explanations, notice cards, amendments, token budget.
- Answering with qwen3:8b, citations, number/citation checks, regeneration, second search.
- Chat UI with clickable citations and page crops; "add to test set" button; evaluation tooling and report.
- 77 automated tests, an OCR benchmark and a quality-gate calibration.

Indexed now: the FY2025-26 tariff rate sheet, SRO 169/2024 (nbr_05), SRO 202/2024 (nbr_06), SROs 210–214/2024
(nbr_07, 08, 09, 10, 12) and SRO 241/2024 (test_02). Not indexed: `nbr-01` (tariff schedule) and `nbr-02`
(2026 SRO compilation; 49 of 355 pages already OCR'd and cached).

---

## 16. Problems found and fixed

### Reading documents

| # | Problem | Fix |
|---|---|---|
| 1 | Surya invents plausible HS codes (12 on 13 pages) | Numbers always taken from the text layer |
| 2 | Surya skipped whole table rows | Missing rows rebuilt from the text layer, placed by column position |
| 3 | Merged cells were repeated and then flagged as unconfirmed | Repeated copies marked as copies, not checked twice |
| 4 | Words split by a hyphen at a line end ("lub-" / "blending") looked different | Joined before comparing |
| 5 | A rate "0" was replaced by a word ("industry1") | Numbers can only be corrected by values of the same shape |
| 6 | A correct rate "3.00" was "corrected" into a code (stacked codes in one cell) | Values of different formats are never paired |
| 7 | A code was moved to the wrong row and passed | Re-ordering accepted only within the same row band |
| 8 | Dash count ("---" vs "--") wrong, breaking the hierarchy | Dashes corrected from the text layer too |
| 9 | Dense tariff pages timed out (12 min per page vs a 10 min limit) and were re-done in a slower mode | Time limit raised to 1 hour |
| 10 | Scanned PDF: Tesseract not found by the app, so every number was held | App uses the bundled Tesseract |
| 11 | Tesseract reads ৪ ৭ ০ as 8 9 0 | Counted as agreement |
| 12 | Numbers on scanned pages missed by Tesseract | Numbers repeated in words confirm themselves |

### Quality gate

| # | Problem | Fix |
|---|---|---|
| 13 | Shifted table rows passed | Column-type check holds them |
| 14 | A few bad rows held the **whole page**, hiding good rows (ceramic tiles on nbr_06 p8; ~120 tariff codes) | Page held only for page-wide problems; otherwise only the bad rows |

### Structure and links

| # | Problem | Fix |
|---|---|---|
| 15 | Rows under a group header ("Ceramic Tiles") had no product name | Group header carried to child rows |
| 16 | Notice references written "এস.আর.ও. নং-" not detected | Pattern fixed |
| 17 | Amendments naming the notice by title were not linked | Title-based linking |
| 18 | "অবিলম্বে কার্যকর" (effective immediately) not understood | Effective from the issue date |

### Search

| # | Problem | Fix |
|---|---|---|
| 19 | Clause numbers ("বিধি ১২") not searchable | Clause-number search |
| 20 | "SRO 210-212" read as one code | Ranges and lists of notices expanded |
| 21 | Bangla keyword search broken: words cut at vowel signs ("ইউডি" → "ইউড") | Split only at spaces and punctuation |
| 22 | Table rows without the sentence that explains the table | Table explanation added automatically |
| 23 | Irrelevant amendment passages crowded the evidence | Only for relevant notices, one per notice, within budget |
| 24 | The approximate vector index (HNSW) missed the best match among 7,000 similar tariff rows | Exact vector search (fast enough at this size) |
| 25 | Questions about several notices did not fit (each first page ~1,300 tokens) | Notice cards (~100 tokens each) |

### Answering and performance

| # | Problem | Fix |
|---|---|---|
| 26 | Prompts of 6,000–8,000 tokens: Bangla costs ~1.2 tokens per character | Budget measured in the model's tokens (~2,500); smaller Bangla chunks |
| 27 | Reranker on the GPU stalled answers for minutes | Reranker on the CPU |
| 28 | OCR and answering on the GPU at the same time slowed both ~8× and caused timeouts | Run one at a time; uploads can be paused |
| 29 | The model copied a template line from the prompt | Template removed; template text rejected by the checker |
| 30 | A correct date was removed because it came from the passage label, not the text | Checker also reads the labels |
| 31 | English questions missed Bangla legal terms | Legal glossary in the prompt |
| 32 | Ollama stopped responding | Restarted; documented |
| 33 | Upload button did nothing | Fixed |

---

## 17. Known limitations today

- **The model sometimes misreads formal Bangla for English questions.** Example: "If an old minimum value
  existed under the rescinded SRO, how is it used…?" – the right proviso was in the context, but qwen3:8b
  answered "not found".
- **Speed:** 20–60 s per answer; ~2 min when it says "not found". OCR is slow: 20–60 s per page; dense
  tables ~12 min per page in batches.
- **One page still fully held:** tariff sheet p57 (3.5% of its numbers needed correction).
- **No measured accuracy yet:** there is no gold question set; quality is known from individual tests.
- **Table structure** still comes from Surya, which can merge or repeat rows on complex layouts.
- **Conflicts** between documents are left to the model to describe.
- **Superseded notices** are not marked or down-ranked.

---

## 18. Planned improvements

### 18.1 Skip OCR on pages that do not need it

- **Problem:** pages with no Bangla (the tariff sheet, `nbr-01`) are still sent through Surya, which is slow
  (2 h 15 min for the tariff sheet) and caused all of its held pages, although their text layer is exact.
- **Why needed:** these pages are the most number-heavy, where accuracy matters most.
- **How:** for each page, check the fonts in the text layer. If there is no Bangla or legacy Bangla font, build
  the text and tables directly from the text layer: group words into lines by position, find column
  boundaries from the header row and word positions, and assemble rows. Surya is used only for pages with
  Bangla.
- **Expected effect:** these pages become exact (no invented, skipped or merged rows), and processing takes
  seconds instead of hours. `nbr-01` (356 pages) becomes practical to add.

### 18.2 Structured table rows and deterministic answers

- **Problem:** a rate or value question is answered by the model reading a text chunk; the model can
  misread or mix up columns, and it takes 20+ seconds.
- **Why needed:** "what is the VAT for 8703.32.11" has exactly one correct answer, already in the database.
- **How:** a generic `table_rows` table: document, table, row cells as `{header: value}`, normalised keys
  (codes, SRO), page and position. When the question contains a code and names a column (VAT, CD, minimum
  value …, matched against that table's own headers), the answer is built directly from the row, with its
  citation; the model is used only for explanations and prose questions.
- **Expected effect:** table lookups become exact and answer in about a second.

### 18.3 English–Bangla glossary at search time

- **Problem:** English questions miss Bangla passages that use legal terms ("exemption" vs "অব্যাহতি").
- **Why needed:** many users ask in English about Bangla documents.
- **How:** before keyword search, add the Bangla equivalents of known terms to the query (and the English
  equivalents for Bangla questions) – CD/কাস্টমস শুল্ক, SD/সম্পূরক শুল্ক, VAT/মূল্য সংযোজন কর,
  exemption/অব্যাহতি, rescind/রহিত, minimum value/ন্যূনতম মূল্য …
- **Expected effect:** more of the right Bangla passages found for English questions.

### 18.4 Superseded notices

- **Problem:** when a newer notice rescinds an older one, both could be shown as if equally valid.
- **Why needed:** giving an outdated rate or rule is a serious error.
- **How:** add a status to each notice (in force / amended / superseded) from the stored links; lower the
  score of superseded notices and label them "superseded by SRO …" in the passage label and the answer.
- **Expected effect:** current rules first, old ones clearly marked.

### 18.5 Retry "not found" with the matching evidence only

- **Problem:** sometimes the model says "not found" although a passage containing the question's HS code or
  SRO is in front of it.
- **Why needed:** this is the most frustrating failure for users.
- **How:** if the answer is "not found" but a passage contains the code/SRO from the question, ask again with
  only those passages.
- **Expected effect:** fewer false "not found" answers, at the cost of one extra model call in those cases.

### 18.6 Clause neighbours in the first search

- **Problem:** a condition or exception is often in the clause right after the one found.
- **How:** when a clause passage is selected, add the clause before/after it if it fits the budget (today
  this only happens in the second search round).
- **Expected effect:** more complete answers on rules questions.

### 18.7 A gold question set and real accuracy numbers

- **Problem:** there is no measured accuracy; each fix is checked on single questions.
- **Why needed:** without a measurement, a fix for one question can quietly break another.
- **How:** about 150 questions written from the documents (HS lookups, table meanings, Bangla clauses,
  cross-document conflicts, rescinded notices, plus questions with no answer), using the questions already
  saved from the chat. Measure search recall separately from answer accuracy (`make eval-user`,
  `eval/report.md`).
- **Expected effect:** an honest accuracy figure and a quick check after every change.

### 18.8 Table structure from Docling (to be benchmarked first)

- **Problem:** Surya sometimes merges or repeats table rows (nbr_06 p8), so good values get held.
- **Why needed:** these rows are lost from search until a person reviews them.
- **How:** benchmark Docling's table-structure model on the 13 benchmark pages and the problem pages; if it is
  better, use it for table structure on digital pages (cell text from the text layer, Bangla from Surya), and
  flag tables where Surya and Docling disagree.
- **Expected effect:** fewer held rows and more reliable tables. To be decided by the benchmark.

### 18.9 Faster or smarter answering (option)

- **Problem:** Bangla answers take 45–50 s; formal Bangla is sometimes misread.
- **How (options):** the reasoning switch for interpretation questions; a smaller model (qwen3:4b) for simple
  lookups; deterministic answers (18.2) remove the model from many questions entirely.
- **Expected effect:** faster answers; better comprehension where reasoning is used. Changing the model would
  be a change from the original brief and needs agreement.

---

## 19. Glossary

| Term | Plain meaning |
|---|---|
| **OCR** | Reading text from an image of a page |
| **Text layer** | The hidden text inside a PDF (what you get when you copy-paste) |
| **Chunk** | One searchable piece: a table row, or one or more legal clauses |
| **Embedding / vector** | A list of numbers representing the meaning of a text |
| **Cosine similarity** | How close two meaning vectors are (1 = same direction) |
| **BM25** | A classic keyword-ranking formula; this system uses PostgreSQL full-text ranking, a similar idea |
| **Trigram** | A group of three consecutive characters, used for fuzzy matching |
| **RRF** | Reciprocal Rank Fusion: combining several ranked lists by position |
| **Reranker** | A model that reads question + passage together and scores relevance |
| **Token** | The unit a language model reads; one Bangla character ≈ 1.2 tokens for qwen3 |
| **SRO / notice / segment** | A government notification; "segment" is the system's word for one notice inside a file |
| **PASS / WARN / HOLD** | Quality-gate statuses: searchable / searchable with a warning / kept out until reviewed |
| **CORRECTION / RESTORED / UNCONFIRMED** | Merge results: OCR value fixed / missing value put back / OCR value with no confirmation |
| **Notice card** | A short passage with an SRO's reference, title, dates and the sentence that sets them |
