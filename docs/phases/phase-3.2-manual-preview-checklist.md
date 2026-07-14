# Phase 3.2 Manual Preview Checklist

Use this checklist to preview the Phase 3.2 Studio a section at a time. It is
the owner's manual product-acceptance record: automated tests and independent
review establish technical readiness, but do not close the phase. Phase 3.2
remains active through preview/fix/review cycles until the owner accepts the
high-priority checks or explicitly defers them.

## Before You Start

Start the local Studio with the default full image and open
<http://127.0.0.1:8000>:

```bash
docker compose up -d studio
```

The normal visual review takes roughly 20–35 minutes. Building the bundled
watsonxDocsQA source corpus into an index can take a few extra minutes.

Do not delete Docker volumes or edit files under the managed seed data during
this preview. Seed integrity and recovery are deliberately verified by the
automated tests rather than through normal UI use.

## 1. Initial Sanity Check

- [√] The header, navigation, and footer render normally.
- [√] Navigation order is **Home → Learn → Explore → Build & Inspect → Failure
  Lab → Settings**.
- [√] In **Settings**, the full image reports **Default embedding model is
  ready**.
  - comments: is it better to let users know what the 'default embedding model' is?
- [√] Toggle **中文**. UI labels switch to Chinese, while the bundled English
  questions, source text, recorded answers, and citations remain English. This
  is intentional.

## 2. Guided Learn: First Complete Lesson

From **Home**, choose **Start guided lesson**. This is the primary Phase 3.2
learning path.

For the first lesson, confirm that its question, structural Cloudflare index,
and `dense · top-k 5` configuration are visible before beginning. Initially
only **Corpus** is available; later stages unlock when **Continue** is used.
> comments: how to decide the question? i saw a question 'How can a Worker use a Durable Object namesapce, stable ID, and stub to send requests for the same entity to one statefule coordinator?'. is this a fixed or preset question? if so, how can we let users know this?

Work through all six stages in order:

### Corpus

- [√] The page shows **40 documents**, **537 chunks**, and the NumPy backend.
- [√] The pinned source identity begins with `3dcb728c`.
  - comments: what does this revision mean? is it a git commit hash? if so, it's better to let users know which repo and branch this hash is from.
- [√] **Inspect raw artifact** shows the source/index provenance.
  - comments: a collapsable/expandable view can be opened, I can see a JSON format data.

### Chunks
> comments: clicked 'continue', transition works well, lesson moves to 'Chunks' stage.

- [√] Several real, source-linked candidate chunks are present—not a
  one-document toy example.
  - comments: is it possible to let users know these are real document snippets? this is another critical point that it's better to let users know there's a preset/built-in corpus which comes from Cloudflare's official documents.
- [√] Each card exposes a rank, source path, and excerpt.
  - comments: yes, rank, source path, and excerpt are all visible on each card.

### Embeddings
> comments: clicked 'cntinue', transition works well, lesson moves to 'Embeddings' stage. there's a bar chart showing with 'Vector preview' heading. But the value of each bar is still shown overlayer on the caption 'First 8 dimensions only -- ...'. I think there's chart rendering issue here. Bar values should be shown above the legend or caption, not overlayered.

- [√] The query-vector preview and complete raw vector are available.
- [x] Bar height represents numeric magnitude; the displayed value preserves
  the sign. A negative and positive value may have similar heights.
  - comments: according to the bar values, i think the bar heigh is rendered partially correctly, because there's two bars with values '-0.11' and '0.10', apparently, the bar with value '0.10' should be higher 2x times than the bar with value '-0.11' based on a linear scale, right? another experience issue is that how to let users know this raw artifact means? users see a JSON array with lots of lines, but don't know what it is. Is there a way to let users know this is the raw vector representation of the query or something else?

### Retrieve

- [√] Candidates are ranked and display retrieval scores.
- [√] Where present, **Score details** exposes the score components.
- comments: yes, I can see the 'Score' and extendable 'Score details' on each candidate card, but the 'cosine_similarity' is always the same value -- 'cosine_similarity[-1,1]', is there any calculation issue here?

### Context

- [√] Evidence indicates whether it is **Selected for context** or omitted.
  - comments: looks like there's nothing different with previous stage 'Retrieve', 'Selected for context' is shown on all cards.
- [√] **Grounded prompt** exposes the packed context actually used for the
  recorded result.
  - comments: yes, I can see the 'Grounded prompt' section, but users may not know what it is, and what it used for. Is there any better way to let users know this part will feed into the LLM model for generating the final answer?

### Answer

- [√] The answer is explicitly labelled **Recorded lesson result**.
  - comments: is it possible to make this 'label' more ease-to-understand for users? Since the LLM provider may not configured yet, so this answer is not generated by LLM, but recorded from the preset answer. So maybe we can make it more clear to users that this is not a really generated by LLM.
- [x] It does not claim to be a live-provider response.
  - comments: I didn't see any claim about 'this answer is not a live-provider response'.
- [√] Citations and the complete raw trace are available.
  - comments: yes, I can see the raw artifact section with detailed informations.

## 3. Guided Learn: Remaining Lessons

Use the lesson rail to complete the remaining three lessons. Each starts with
only its Corpus stage unlocked and has a distinct focus:

- [√] **Queue retries and batching**
- [√] **KV versus R2**
- [√] **Workflows pause, retry, and resume**

For every lesson, confirm the changed question/focus and that the entire replay
works without an API key, provider, or model download. Once completed, its
pipeline stages remain revisitable until the page is reloaded.

## 4. Explore: Bundled Cloudflare Retrieval

Open **Explore** and select `cloudflare-state-structural-v1`.
> comments: I think it's better to let users know what's the difference between 'cloudflare-state-structural-v1' and 'cloudflare-state-fixed-v1', and how this index was built and how it may affect the retrieval results.

Use a free question such as:

> How do Durable Object stubs route requests for the same entity?

Set **Dense**, **Top-k 5**, and **Context budget 0**, then choose **Retrieve**.
> comments: I think it's better to let users know what does each option means, and how it may affect the retrieval results.

- [√] Retrieval works without a provider and opens at the Retrieve stage.
  - comments: yes, and also able to see a 'Stage timings' section, which shows the time cost of 'embed' and 'retrieve'.
- [√] All six pipeline stages can be inspected for the same run.
  - comments: for the example question, after 'Retrieve' clicked, the default stage is '04 Retrieve', not sure if this is expected behavior. I think the previous stages are same as the preset question, is that right?
- [√] Corpus reports 40 documents and 537 chunks.
  - comments: yes, but need to manually click '01 Corpus' stage to see the corpus information.
- [√] Chunks, query vector, ranked candidates, context selection, prompt, and
  raw artifacts are coherent for the same query.
- [√] No **Gold-source check** appears for this free-text run.
  - comments: I didn't see any new stage or section about 'Gold-source check'.

Repeat the query with each retriever and with the fixed chunk index:

- [√] BM25
- [√] Hybrid (RRF)
- [√] `cloudflare-state-fixed-v1`, which reports 40 documents and 522 chunks
  - comments: in 'Retrieve' stage, I found the doc snippets were cropped, for example, the heading word is incomplete and the last word is also incomplete, is this expected behavior? I think it's better to let users know this is a preset index and the doc snippets are cropped for display purpose.

Different chunk boundaries and, potentially, different rankings are expected
when changing index or retriever.

## 5. Build And Inspect watsonxDocsQA

Open **Build & Inspect → Build index**.

1. Choose **watsonxDocsQA**, which contains 1,144 source documents.
> comments: the number in the '()' should be more clear to users, for example, '1144 documents (bundled with the image)', so that users can know this is a preset corpus and not a user-uploaded corpus.
2. Leave the backend as **NumPy**.
3. Choose **Build index** and wait for **Indexing locally…** to become
   **Index ready.**
> comments: indexing seems work well, I can see 'Indexing locally...' stage, and also 'Index ready.' stage. Is it possible to show a progress indicator or an estimated time? Otherwise users may guess if the application get stuck. And is it better to make button 'Build index' not sit in the 'Build index' section'? Because building index needs 2 steps, first a given corpus, then a given index backend. So button's position may implicate that users must make sure the corpus and backend are properly selected.

Then verify:

- [√] The completed job automatically opens **Inspect index**.
- [√] Index facts, sample chunks, vector previews, and raw manifest are
  inspectable.
  - comments: there's rendering issue, the text in sample chunk card is extended to the card's right edge.
- [√] The new watsonxDocsQA index appears as an available Explore index.
  - comments: yes, I can the the new index in the 'Inspect index' dropdown list, but I'm not sure what's the naming convension, it shows like 'index-465d351c89b2'.
  - comments: the 'Index ready.' stage banner doesn't have a close button, so it shows permanently.

This explicit background build is intentional even though the watsonxDocsQA
source documents are bundled with the image.

## 6. Catalog Questions And Gold-Source Checks

In **Explore**, select the newly built watsonxDocsQA index.

- [√] A **Catalog question** dropdown appears.
  - comments: yes, I can see an item named 'index-465d351c89b2' in the dropdown list.
- [√] It contains the 75 bundled questions; featured questions carry `★`.
  - comments: yes, I can see lots of questions in the 'Catalog question' dropdown list, and some of them with a star icon.
- [√] The menu reveals only question text—never expected answers or gold source
  metadata before a run.
  - comments: is it better to provider an 'empty' option in the dropdown list, it's a hint to let users know they can select preset questions or ask a free question. and only if users select 'Free question', the 'Ask a question' input box then got editable, otherwise, it's should read-only.

Choose a featured catalog question, confirm that it fills the Question field,
and choose **Retrieve**.
> comments: after I clicked 'Retrieve' for a featured question, i can see some returned document cards. But some of the docs' title show only a placeholder '{{ document.title.text }}', some of them are expected title. In each card, I can see a score shown on the bottom, and a 'reciprocal_rank_fusion' text besides. Also I can see a clickable 'Score details' at the bottom-right corner, it shows either 'dense_score', 'dense_rank' or 'bm25_score', 'bm25_rank'.

- [√] The completed run displays **Gold-source check** at Retrieve, Context,
  and Answer.
  - comments: yes, I can see a 'Gold-source check' section on the page bottom, it said 'A gold source was retrieved', as well as a 'Expected' doc, with a list of 'Retrieved' docs. Is it better to list all 'Retrieved' docs one by one and only one item per line.
- [√] The panel shows expected document IDs, retrieved document IDs, and a
  hit/miss result.

Next, manually edit the Question field and retrieve again:

- [√] The catalog selector reverts to **Free question**.
  - comments: yes, but I think it's better to refine this interaction behavior, selecting a 'Free question' then make the input box be an blank and editable field.
- [√] The next free-text result does not display a Gold-source check.
  - comments: yes, for a free question, no 'Gold-source check' section shown. But different things is: each returned doc cards now has correct doc title, for example, there's a card with title 'Tuning Studio'.

A hit or miss reports the retrieval result for that particular configuration;
the manual check is that the server-side evaluation metadata is revealed only
after a validated catalog run.
> comments: on the 'Explore' page, 'Live ask' button should be disabled if no a given LLM provider is configured. I can see the little hint text in grey said 'Live ask needs an OpenAI-compatible provider...', but the button is clickable, which may lead users confused. And it's better to make 'Retrieve' button has a 'clicked' effect, now it has a hover effect to let me know it's clickable, but when I clicked it, there's no effect which makes me not sure if I really clicked it.

## 7. Custom Corpus Path

Prepare two or three small `.md` or `.txt` files whose answer is easy to verify.
In **Build & Inspect → Build index**:

1. Choose **Add Markdown or text files** and upload them.
2. Confirm a separate **My corpus** entry appears.
3. Build a NumPy index.
4. Inspect its chunks and vector preview.
5. Select the new index in Explore and retrieve an answerable question.

- [ ] The custom corpus and index coexist with, rather than replace, the
  bundled Cloudflare and watsonxDocsQA assets.

## 8. Provider Boundary And Live Ask

Without a provider configured, choose **Live ask** for any valid index and
question.

- [√] The lab clearly asks you to configure an OpenAI-compatible provider.
  - comments: the little text in grey is always showing there. I think it's better to make 'Live ask' button disabled as well, to align with the hint text.
- [√] Retrieval-only use remains available without one.

To test live generation, enter a reachable OpenAI-compatible provider base URL,
model name, and (when required) API key in **Settings**. Return to Explore and
choose **Live ask**.
> comments: I strongly suggest to add a button like 'Test connection' to make sure user's given LLM provider configurations are correct and able to work for 'Live ask'.

- [√] A generated answer and citations appear at the Answer stage.
  - comments: firstly I think it's better to make such time-cost request has a floating notification to let users know that 'request is on the way, please wait', otherwise users may click this button for multiple times, which may lead unexpected behavior. Another finding is there's a 'Citations returned' banner shown above doc cards. I don't know what does it mean. And all cards now don't have scoring shown, nor 'Score details'. I don't know if this is expected. The most critical finding is, I don't see any answer there, the question I asked is 'What's the most advantage of IBM foundation model?'. Only doc cards which I think are citations. But I didn't find out where's the answer?

The API key is held only for the current browser session; do not include it in
screenshots. The provider URL and model may be remembered by the browser.

## 9. Retained Lab Capabilities

These pre-existing capabilities were required to continue working in Phase 3.2:

- [√] In **Failure Lab**, select each available scenario and compare baseline
  versus intervention configuration, evidence, context selection, answer,
  citations, and raw trace.
  - comments: I strongly suggest that, for each failure scenario, I can see a section shown below the failure scenario dropdown list. But the sample question should have different style thus could let users know clearly what's the question for this failure demonstration. So maybe we could separate failure scenario related information with sample question. Both of them are important, failure reason explain to user under what situation what failure would happen, sample question let users know what question under what situation will cause a failure. Do you got my points?
- [√] Open a learning-material link from both Learn and Explore; it should use
  the language currently selected in the UI.
- [√] Narrow the browser window (or use device emulation). Navigation, lesson
  rail, cards, and controls should remain usable.
  - comments: yes, but I found a minimal window width -- about 717px, if smaller than it, 'Build & Inspect' pages have issues, for example, in 'Build&Inspect' page, the chunk cards and content in each card extend out of the window.
- [√] If your operating system/browser requests reduced motion, the UI respects
  it.

## Optional Deployment Checks
> comments: not verified yet, let's make all required checks meet acceptance criteria.

These are outside the normal visual preview and are not needed before giving
product feedback:

- **Qdrant backend:** start it with
  `docker compose --profile qdrant up -d qdrant`, then build a separate index
  using **Qdrant (optional)**.
- **Slim image:** restart with `LAB_IMAGE_VARIANT=slim`. Guided Learn and BM25
  over an existing seeded index should work; dense/hybrid retrieval and
  indexing should request the explicit embedding-model download.

## Reporting Feedback

For each finding, note the page, selected index/lesson, question, retriever,
and stage. A screenshot is especially useful for layout or visual issues.
