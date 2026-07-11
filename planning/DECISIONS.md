# Decisions — HSK-5 Mandarin Tutor

> Dated log of notable choices and *why*, so rationale isn't lost. Newest first.

## 2026-07-10 — App UI: "teacher's red ink" paper aesthetic, light-only
- **Decision:** Redesigned the Gradio app around a scholarly rice-paper look — ink typography (EB Garamond + Songti/Kaiti), one cinnabar-red accent (pinyin, seal, user wash — the teacher's 朱批 red), transcript as a red-margined manuscript sheet. The app is **light-only**: a JS hook strips Gradio's `.dark` class (with CSS var overrides as backstop) since paper-on-dark was the original readability bug's root cause.
- **Why:** User found the default Gradio look ugly; the paper/ink direction extends the aesthetic the docs/ pages and reading layer already had. One palette to maintain instead of two.
- **Implications:** All styling lives in `PAGE_CSS`/`THEME` in app.py, passed to `launch(css=, theme=, js=)` (Gradio 6 moved these off `Blocks()`). Chinese fonts are macOS system fonts (Songti/Kaiti SC) with Noto Serif SC fallback — fine for a local demo. `gradio-app`'s inline dark background is pinned to paper by CSS.

## 2026-07-10 — Untranslated framing lines: inference-only prompt nudge (no retrain)
- **Decision:** The tuned model translated only example sentences, leaving greetings/closers/usage notes Chinese-only. Fix chosen: a serve-time-only `SYSTEM_PROMPT_APP` in config.py (adds one line demanding English for *every* Chinese sentence); `app.py` uses it, `gen_data.py` keeps the original `SYSTEM_PROMPT`.
- **Why:** The behavior is trained-in — the teacher data has the same pattern because the per-task instructions only required translations "per example sentence". A retrain is overkill for a cosmetic gap; the nudge is free and reversible. Tested live: usage notes and closers now get translated (occasionally a 2–4-word greeting like 好，我们来看看 stays Chinese-only — acceptable, decodable via the reading layer).
- **Implications:** Breaks the "train-time and serve-time prompts identical" invariant, deliberately and documented in config.py. If we ever regenerate data, fix the task instructions instead and drop `SYSTEM_PROMPT_APP`.

## 2026-07-10 — Reading layer in Gradio: keep gr.HTML, `!important` colors + CSS tooltip (no iframe)
- **Decision:** Render the transcript with `gr.HTML` (unchanged), fix readability with `!important` color rules, and show the hover gloss as a pure-CSS tooltip (`.hz:hover::after` reading a `data-tip` attribute) instead of the native `title` tooltip. `render_chat` rewrites `annotate()`'s `title=` → `data-tip=` for display; `annotate.py` and the standalone `docs/` pages keep native `title`.
- **Why:** DOM inspection showed `gr.HTML` does **no** sanitization (verbatim `innerHTML`) — `<style>`, `ruby/rt`, and all attributes survive, so the earlier "Gradio strips it" theory was wrong and an `<iframe srcdoc>` workaround is unnecessary. The real bug was CSS: Gradio ships `.gradio-container-* .prose * { color: var(--body-text-color) }`, a universal selector that recolors every element and beats inherited colors — in dark mode that's near-white text on our light bubbles. `!important` is the only clean win against a universal selector we don't control. The CSS tooltip replaces native `title` because native tooltips have a ~1s delay and are unreliable in embedded webviews; it's also instantly verifiable.
- **Implications:** Bubbles stay a fixed light "paper" palette on both themes (readable in dark mode by forcing dark text). If Gradio's `.prose *` rule ever changes, the `!important` rules are self-contained in `CHAT_CSS`. `app.py` also honors a `PORT` env var (dev tooling; default 7860 unchanged).
- **Decision:** Flashcard widget uses the **browser Web Speech API** (client-side, zero-dep) for 🔊 pronunciation — done. The **chat app** will use neural **`edge-tts`** (server-side, free, consistent zh-CN neural voice) to read tutor replies aloud, wired at ship.
- **Why:** The standalone flashcard file has no Python backend, so client-side TTS is the only option there (and macOS zh-CN voices are good). For the app, `edge-tts` gives consistent studio-quality audio regardless of the user's machine.
- **Implications:** flashcards done; app `edge-tts` is an M3/ship task. App-layer, no retraining.

## 2026-07-10 — Flashcards: in-app SM-2 review (self-contained widget), not export-only
- **Decision:** Build an in-app Anki-like review (**SM-2** spaced repetition) as a self-contained HTML/JS widget (`web/flashcards.html`), deck in `localStorage`. Chosen over export-only because the user won't reliably export to Anki. TSV/JSON export kept as a backup. To stay zero-friction (collect in chat → review, no import), the review will live as a **same-origin tab inside the Gradio app** at ship time; the widget is built + verified standalone now (SM-2 logic unit-tested in node).
- **Why:** A review tool you'll actually open beats a better one you won't. SM-2 over FSRS for simplicity / no dependency. Same-origin matters: a separate `file://` page can't share the app's `localStorage`, which would reintroduce an import step. Refines the export-only framing below.
- **Implications:** `web/flashcards.html` standalone now; integrate as an app tab + wire "collect word from chat" at M3. App-layer, no retraining.

## 2026-07-10 — Roadmap: Anki flashcard export (app-layer, reuse the reading layer)
- **Decision:** Extract vocab and export to Anki at the **app layer**, reusing the reading-layer primitives (jieba segmentation + pypinyin + CC-CEDICT). Selection: primary = user click-to-collect words on hover; secondary = an "auto-extract" button (jieba content words ∩ CC-CEDICT, minus a basic-word stoplist). Card = word (front) / pinyin + gloss + the example sentence from the conversation (back). Export: **TSV download** for v1 (zero-dep, Anki File→Import); optional `genanki` `.apkg` (styled + cloze) as polish.
- **Why:** The hover layer already computes word/pinyin/gloss for every token, so a card is ~free. User-chosen words are both easier to build and better pedagogy. Deterministic extraction (jieba/CC-CEDICT) beats an LLM here — faster, reliable, no cost. Do **not** train the tutor to emit a rigid vocab section (keeps responses natural, extraction stays app-side).
- **Implications:** v2 **app-layer** feature; **no change to training or data**.

## 2026-07-10 — Roadmap: two-model curriculum architecture (Claude planner + Qwen executor)
- **Decision:** Learning-goal / curriculum planning is done by a capable model with memory + reasoning (Claude), **not** by the fine-tuned Qwen. Claude discusses goals with the user and writes a persisted `curriculum.md` (and a companion `progress.md`); `app.py` loads `curriculum.md` and injects it into Qwen's system prompt each session. Qwen stays the fast, local, HSK-5 drill tutor.
- **Why:** Qwen is small and has no filesystem / long-term memory — it *executes* tutoring well but shouldn't *plan*. A curriculum is specific and changes often → belongs in **context**, not weights. The `curriculum.md` / `progress.md` files are the shared long-term memory between the planner (Claude) and the executor (Qwen). Standard "planner + specialized executor with a shared state file" pattern.
- **Implications:** v2 **app-layer** feature only — read `curriculum.md` → inject into the system prompt. **No change to training or data** (Qwen2.5-7B-Instruct already follows in-context instructions). Possible later automation: a scheduled Claude review that rewrites `curriculum.md` from `progress.md`.


## 2026-07-09 — Product spec v1 locked (D1–D7)
- **Decision:** Fully **bilingual** output (D1). Pinyin + per-word English gloss are an **app-side reading layer**, not model output (D2): `pypinyin` renders pinyin ruby over every character, `jieba` segments words, bundled **CC-CEDICT** supplies hover glosses, rendered as HTML `<ruby>` + `title` tooltips. Single-turn training data (D3). No quiz mode in v1 (D4). Above-level requests simplified + flagged (D5). Simplified characters only (D6). Gradio chat + starter buttons (D7). **Voice chat + TTS deferred to v2 roadmap.**
- **Why:** A 1.5B model shouldn't generate per-character pinyin (verbose, error-prone) — a library is always correct and needs zero training data. Bilingual is the only content change; the reading layer is pure rendering. Keeps v1 small while delivering the headline "readable" feature.
- **Implications:** `SYSTEM_PROMPT` + task instructions go bilingual and drop inline pinyin (done). New `annotate.py` module + CC-CEDICT asset (~4MB, CC-BY-SA → README attribution). `app.py` renders HTML (ruby + title). Hover tooltips are desktop-only (no mobile hover) — acceptable for a demo.

## 2026-07-09 — Base model: upgraded to Qwen2.5-7B-Instruct (supersedes 1.5B)
- **Decision:** Use `Qwen/Qwen2.5-7B-Instruct` as the base, for a genuinely reliable tutor.
- **Why:** Fine-tuning teaches format + level-control, not Chinese knowledge — so base quality is the ceiling. 7B is markedly more accurate on our tasks (corrections, grammar, natural sentences); a tutor that teaches wrong Chinese is worse than none. QLoRA 4-bit of a 7B is the standard case and trains fine on paid Colab (~30–60 min); only the model id + batch sizing change (per-device batch 4 × accum 4 = eff. 16, gradient checkpointing on).
- **Implications:** **Local demo must serve the merged model quantized** — GGUF via `llama.cpp` (~4.5GB, snappy) or MLX — not raw bf16 on MPS (~15GB, slow) → new convert-and-serve step in M3. **Free HF Spaces can't run a 7B** (CPU-only); local demo is primary, a public Space would need a paid GPU. Revisit if serving proves annoying (3B is the fallback).

## 2026-07-09 — Base model: Qwen2.5-1.5B-Instruct  [superseded above]
- **Decision:** Start with `Qwen/Qwen2.5-1.5B-Instruct`, not 3B.
- **Why:** Strong Chinese for its size; fast train + fast Gradio inference keeps the vibe-project loop tight. 3B is an easy one-line bump if M2 eval shows quality is short.
- **Implications:** Fast iteration; revisit model size after seeing eval results.

## 2026-07-09 — Training path: QLoRA on Colab (not local MLX)
- **Decision:** Train via HF `peft` + `trl` `SFTTrainer` + `bitsandbytes` 4-bit QLoRA on **paid Colab**; keep data-gen and the app local on the Mac.
- **Why:** This is the transferable, industry-standard fine-tuning stack (the user's stated learning goal) and the most reproducible/shareable path; it maps cleanly to HF Hub + Spaces shipping. MLX is excellent and fully local on the M5, but Apple-specific and doesn't map to the HF ecosystem. bitsandbytes 4-bit is CUDA-only, so training is a GPU/Colab step.
- **Implications:** `train.py` runs on Colab; `app.py` runs the merged model locally on MPS (no bitsandbytes at inference).

## 2026-07-09 — Data: synthetic, Claude teacher, ~800–1,000 pairs
- **Decision:** Generate ~800–1,000 chat-format pairs with Claude (Anthropic API), seeded from a curated HSK-5 vocab/grammar list, ~10% held out for eval. Six task types.
- **Why:** Enough signal for a LoRA on a small model, cheap + fast to generate in one session, and level-controllable via the seed list + system prompt. Matches "keep it small / don't chase eval perfection."
- **Implications:** Level control depends on seed quality + prompt constraints; start with a curated in-repo seed and expand only if variety is thin.
