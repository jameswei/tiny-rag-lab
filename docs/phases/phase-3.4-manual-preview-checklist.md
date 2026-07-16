# Phase 3.4 Manual Preview Checklist

**Status:** Owner accepted the remediated Studio experience on 2026-07-16. No
commit was created before acceptance.

Checked items below preserve what the owner actually exercised. Unchecked
Browser A/B evaluation items and explicitly noted optional/accessibility items
remain unchecked; automated and independent review coverage is recorded in the
taskboard, not presented here as owner verification. Original `comments:` are
retained, with resolution notes added after the corresponding remediation.

Use the current source preview at <http://127.0.0.1:5173/>. Complete the
checklist in several sessions if preferred. Add feedback immediately below an
item with the prefix `comments:` so it can be collected without changing the
checklist wording.

The updated project landing page is separately available at
<http://127.0.0.1:8790/>.

## 1. Entry point and course shape

- [√] Open the Studio and confirm the navigation order is Home, Learn,
  Retrieval, Explore, Build & Inspect, Failure Lab, Settings.
- [√] Open **Retrieval**. Confirm that it starts on **01 Lexical Search** and
  presents six directly selectable modules.
- [√] Confirm the page explains that these are live retrieval experiments and
  does not imply that an LLM provider is required.
- [√] Switch to 简体中文 and back to English. Confirm the shell and teaching
  copy change language while Cloudflare questions/source text remain English.

## 2. Lexical Search

- [√] Select the question containing `max_retries` and run live retrieval.
- [√] Confirm query tokens are visible.
- [√] Confirm each result shows BM25 score and per-term contribution details,
  including term frequency, document frequency, IDF, and length inputs.
  - comment: BM25 section shows for each chunk card, but there's layout/rendering issue, that the table seems not obey the max width, in maximum window, the last column get exceeds the right boundary of chunk card, and if resize window size, the last two columns all exceed the right boundary of chunk card. only if the windows size is narrow enough, BM25 section doesn't sit right of chunk details but at the bottom, then it looks obey the card boundary. 
  - resolution: Fixed with a bounded table region that scrolls internally at
    narrower widths; the owner accepted the result.
- [√] Confirm the individual contributions explain the displayed final score
  rather than presenting BM25 as a black box.
- [√] Open **Read the learning guide** and confirm it opens the local
  Retrieval Mechanics guide in the selected UI language.

## 3. Dense Retrieval

- [√] Select a paraphrase-oriented dense question and run it.
- [√] Confirm the query vector preview, result vector preview, dimensions,
  norms, dot product, cosine similarity, and final rank are visible.
- [√] Confirm negative vector components extend in the opposite direction from
  positive components; bar height must not silently use absolute values.
  - comments: the bars of both query vector and chunk vector render incorrectly, some of bars look in the right place (perhaps) but some of bars are flying in the air, you could see the screenshot, i put it under `.polish/cosine_cal_bar.png`.
  - resolution: Fixed with one signed scale, an explicit zero axis, and a
    non-wrapping column count derived from the returned 12-value preview. The
    owner accepted the transparent value and zero labels.
- [√] Confirm long source text remains bounded/expandable rather than making
  every result card arbitrarily tall.

## 4. From local vectors to a vector database

- [√] Open module 03. Confirm it first explains NumPy as the inspectable local
  path and Qdrant as a second index backend, not a different conceptual flow.
  - comments: there's no certain wordings to explain that 'there's no different concepture flow'. just explained numpy is able to inspect chunks and vectors, and Qdrant is an optional index for the same vectors, and payloads.
  - resolution: Strengthened the teaching copy to state that chunks, vectors,
    query, distance metric, and ranking concept remain the same while the
    storage/search backend changes.
- [√] Confirm Qdrant is shown as available but not prepared.
- [√] Click **Prepare Qdrant comparison**. This copies existing chunks/vectors;
  it should not show re-chunking or re-embedding work.
- [√] After preparation, confirm 537 points, 384 dimensions, exact search, and
  a source fingerprint are visible.
- [√] Run an unfiltered backend comparison. Confirm NumPy and Qdrant result
  cards appear side by side and parity is reported as equivalent.
- [√] Confirm Qdrant result payloads expose useful learner-facing fields such
  as chunk ID, document path/title, and source group.
  - comments: only the results/cards of Qdrant-exact has a expandable 'Inspect stored payload' but NumPy card doesn't have.Not sure if this was expected.
  - resolution: Kept this intentional contrast and explained it in the module:
    local NumPy exposes arrays/files directly, while Qdrant adds stored payload
    metadata to the same vectors.
- [√] Choose a source-group filter such as `r2` or `queues`. Confirm filtered
  Qdrant results are presented separately and are not described as a NumPy
  parity comparison.
- [√] Repeat preparation once. Confirm it safely reuses the verified prepared
  collection rather than duplicating or rebuilding it unnecessarily.
  - comments: i didn't find out how to repeat the preparation module, if i clicked 'Dense' for example, and clicked back to 'Vector database', there's no 'Prepare Qdrant comparison' again.
  - resolution: Added an explicit repeat-verification action; repeated prepare
    reports reuse of the already verified 537-point collection.

## 5. Hybrid Retrieval

- [√] Open module 04 and use a question with more than one gold document, such
  as the R2 consistency/durability question.
  - comments: i can the 'How do R2 consistency and durability differ?' question in dropdown list, but not sure how to check if it has more than 1 gold document.
  - resolution: Gold document paths are now shown alongside reviewed questions.
- [√] Confirm separate Dense and BM25 rankings are visible.
- [√] Confirm each result exposes the available `1 / (60 + rank)` contribution
  from both lists and the summed RRF score.
  - comments: there're 2 ranking items only have 'BM25 ranking' contribution card, you can check the screenshot `.polish/fused_ranking.png`.
  - resolution: Every fused row now shows both source contributions; a missing
    source is explicitly represented as `+0` rather than omitted.
- [√] Confirm the fused ordering matches the displayed contributions.

## 6. Reranking

- [√] Open module 05 and run a curated reranking question.
- [√] Confirm the flow identifies 20 first-stage candidates and a final top 5.
- [√] Confirm the complete audit table includes first rank/score,
  cross-encoder score, final rank, rank delta, and moved/unchanged/dropped
  outcome.
- [√] Confirm dropped candidates remain inspectable rather than disappearing.
- [√] Open **Read the learning guide** and confirm it opens the local Reranking
  guide, not the generic Retrieval Mechanics page.

## 7. Browser A/B evaluation

- [√] Open module 06. Confirm the fixed index identity and 16 reviewed
  questions are clear.
- [√] Run the **BM25 vs Dense** preset.
- [√] Confirm queued/running progress is visible and the finished comparison
  reports hit rate, MRR, context precision, and context recall separately.
  - comments: the progress bar is not good looking.
  - resolution: Replaced the native browser progress element with a compact
    status header, current/total counter, descriptive message, and a rounded
    themed meter. The counter now names its unit explicitly (`16 of 16
    questions` / `16 / 16 个问题`), and the owner accepted the result.
- [√] Confirm there is no composite winner.
- [√] Inspect several individual questions. Confirm each shows the question,
  gold document paths, per-side metrics, and complete retrieved evidence.
- [√] Confirm the displayed delta is clearly defined as B minus A.
- [√] Edit one side and try two identical configurations. Confirm the lab
  rejects the non-instructive comparison with a clear message.
- [ ] Run **Hybrid vs Hybrid + cross-encoder** when time permits and inspect
  whether aggregate changes agree with per-question evidence.
- [ ] While a comparison is running, navigate away and return. Confirm the
  active job and its exact configuration recover.
- [ ] Optional: request cancellation. Confirm it explains that cancellation
  occurs after the current model/retrieval operation and does not publish a
  complete result.
- [√] Open **Read the learning guide** and confirm it opens Evaluating
  Retrieval in the selected language.

## 8. Explore with reranking

- [√] Open Explore and select `cloudflare-state-structural-v1`.
- [√] Choose Hybrid, enable the cross-encoder reranker, retain final top-k 5,
  and use candidate depth 20.
- [√] Ask one of the curated reranking questions and click **Retrieve**.
    - comments: i don't know which is a curated reranking question, so i just ask a random question - 'what's R2 service?'
    - resolution: Replaced the competing reviewed/free controls with one
      question field whose suggestions identify the reviewed reranking set
      while still accepting free questions.
- [√] Confirm Explore shows the same candidate-to-final audit vocabulary as
  the Retrieval course.
  - comments: maybe due to i asked a not exact curated question, so i got a not exactly same candidate-to-final result table.
  - resolution: Explore now reuses the candidate, pre-rank, reranker score,
    final rank, movement, and dropped-result vocabulary from Retrieval.
- [√] Change candidate depth and confirm it remains greater than or equal to
  final top-k.
- [x] Optional provider check: if a tested OpenAI-compatible provider is
  already configured, use **Live Ask** with the same settings and confirm the
  retrieved/reranked evidence is the evidence passed to generation.
  - comments: not sure how to check if the retrieved/reranked evidence is the evidence passed to generation. because i only see some 'selected context evidence' cards. no more candidate-to-final result.
  - resolution: Live Ask now preserves the same reranking audit and marks the
    final evidence selected for context, so the retrieval-to-generation
    boundary remains inspectable.

## 9. Models and Settings

- [√] Open Settings. Confirm both pinned model identities and revisions are
  visible: the MiniLM embedding model and the MS MARCO MiniLM cross-encoder.
- [√] Confirm both show ready in this full/native preview.
    - comments: 'Ready locally...' text may need to align vertically in center, because i saw it almost in the top space of left 'green bar' indicator. you can check the screenshot `.polish/settings_model_ready.png`. and the 'OpenAI compatible provider' section is too close with 'Pinned model' section, there's no space, you can check the same screenshot.
    - resolution: Corrected model-state alignment and section spacing.
- [√] Confirm the surrounding explanation distinguishes which operations need
  embeddings and which additionally need the reranker.
  - comments: i didn't find any surrounding explanation. you can check the screenshot `.polish/settings_model_ready.png`.
  - resolution: Added distinct operation roles for the embedding and
    cross-encoder models.
- [√] Confirm no GPU or CUDA requirement is suggested.
  - comments: i didn't find any hint like this. you can check the screenshot `.polish/settings_model_ready.png`.
  - resolution: Added explicit CPU-only guidance; full and slim image smokes
    also confirmed no CUDA runtime or NVIDIA packages.

The slim-image download lifecycle will be presented as a separate isolated
verification after the main Studio experience is accepted; it does not need to
block this first walkthrough.

## 10. Learning Guides and public wording

- [√] From Retrieval, open the Retrieval Mechanics, Reranking, and Evaluating
  Retrieval guide targets through their corresponding modules.
- [√] In each guide, switch English/简体中文 and confirm the paired page exists.
  - comments: not sure if all sans fonts are better on docsite. now there're both serif and sans fonts mixed together shown in docsite, a little bit disharmony.
  - resolution: Aligned the Learning Guides on the accepted sans-serif system.
- [√] Confirm the guide navigation places Reranking between Retrieval Mechanics
  and Evaluating Retrieval.
- [√] Open <http://127.0.0.1:8790/>. Confirm the landing page presents the live
  retrieval course as an integrated Studio capability, not as a Phase 3.4
  changelog.
- [√] Confirm the landing page still communicates one core with two learning
  entry points: visual Studio and direct CLI.
  - comments: the wording in the landing page is visual workspace and direct CLI.

## 11. Responsive and accessibility pass

- [√] Resize the Studio through wide desktop, narrow desktop/tablet, and mobile
  widths. Confirm the six-module rail, controls, comparison columns, tables,
  vector charts, and evidence cards remain usable.
  - comments: i've reported the layout/render issue above.
- [√] Confirm wide audit tables scroll inside their own area instead of
  widening the whole page.
- [√] Navigate the Retrieval module rail, form controls, and guide link by
  keyboard. Confirm focus remains visible.
  - comments: didn't verified completely.
- [x] If reduced motion is enabled at OS/browser level, confirm no essential
  information depends on animation.
  - comments: not verified, i think modern devices no need 'motion-reduced' mode.

## 12. Overall acceptance

- [√] The recommended order from lexical search through evaluation is clear,
  but modules are not artificially locked.
- [√] Calculations are visible before backend abstractions.
- [√] Qdrant feels substantial and educational while remaining optional.
- [√] A learner can distinguish candidate generation, fusion, reranking, final
  evidence, and evaluation.
  - comments: i guess yes, assume i'm a learner.
- [√] The experience feels like part of tiny-rag-lab rather than a separate
  dashboard or a clone of another learning project.
- [√] No blocking or high-priority finding remains before final review.
    - comments: see findings above.
    - resolution: The findings above were remediated through successive owner
      previews; the owner then accepted the final result on 2026-07-16.
