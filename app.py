"""Gradio chat demo for the HSK-5 tutor.

Serves the merged+quantized 7B locally via llama.cpp (Metal on Apple Silicon) and
renders every reply through the reading layer (annotate.py): pinyin over each
character + hover gloss on each word.

Design note: we keep TWO copies of the conversation. The model always sees the
RAW text (feeding it our HTML back would poison the context); the chat panel
shows the ANNOTATED HTML. That separation is why the state plumbing below exists.

    python get_cedict.py      # once, for hover glosses
    python app.py             # needs outputs/hsk5-tutor-q4_k_m.gguf (see README)
"""

from __future__ import annotations

import html
import json
import os
import random
import re

import gradio as gr
from llama_cpp import Llama

import config as c
from annotate import HAS_CJK, ambiguous_words, annotate, sense_options, unglossed_words
from gen_data import extract_json_array, extract_json_object, load_seed

if not c.GGUF_FILE.exists():
    raise SystemExit(
        f"Model not found: {c.GGUF_FILE}\n"
        "Build it first: train (Colab) → merge.py → GGUF quantize → put the .gguf in outputs/. "
        "See the README."
    )

# n_gpu_layers=-1 offloads everything to Metal on Apple Silicon. The chat template
# is read from the GGUF metadata (Qwen2.5 ships a ChatML-style template).
llm = Llama(model_path=str(c.GGUF_FILE), n_ctx=4096, n_gpu_layers=-1, verbose=False)

# Fallback starter pool, ~4 per task type — used only when gen_starters()
# below fails. Normally the starters are written by the model at startup,
# seeded with random vocab/grammar so every launch gets a genuinely new set.
STARTER_POOL = [
    # explain_word
    "“毕竟”是什么意思？",
    "解释一下“临时”这个词",
    "“居然”和“竟然”有什么区别？",
    "“把握”是什么意思？怎么用？",
    # use_in_sentence
    "用“承担”造两个句子",
    "用“逐渐”造个句子",
    "用“珍惜”写三个例句",
    "用“体会”造句",
    # correct_sentence
    "帮我改这个句子：我昨天去过北京。",
    "这个句子对吗？我很喜欢吃中国菜非常。",
    "帮我改一下：他比我更高很多。",
    "这样说对吗？我明天要见面我的朋友。",
    # translate
    "How do I say “I can’t help but worry” in Chinese?",
    "怎么用中文说 “it’s not worth it”？",
    "Translate: 这件事说起来容易，做起来难。",
    "How do I say “long time no see” in a formal way?",
    # example_dialogue
    "给我一段在餐厅点菜的对话",
    "给我一段面试时的对话",
    "写一段在机场值机的对话",
    "给我一段去医院看病的对话",
    # grammar_point
    "怎么用“既然……就……”这个语法？",
    "“除非……否则……”怎么用？",
    "“越来越”和“越……越……”有什么不同？",
    "“连……都……”是什么意思？",
]


def _load_seed(path) -> list[str]:
    return load_seed(path) if path.exists() else []


HSK_VOCAB = _load_seed(c.VOCAB_FILE)
HSK_GRAMMAR = _load_seed(c.GRAMMAR_FILE)


def gen_starters() -> list[str]:
    """Model-written starter chips, one per task type, seeded with random vocab
    and grammar so every launch is a new set. Falls back to STARTER_POOL —
    for ANY failure, including missing/short seed files, so this can never
    block launch (everything lives inside the try)."""
    try:
        words = random.sample(HSK_VOCAB, 4)
        grammar = random.choice(HSK_GRAMMAR)
        prompt = (
            "为一个HSK5水平的学生写6个开场问题，模拟学生问中文老师时会说的话，每种一个。"
            "每一条都必须是学生对老师的【请求或提问】，不能是回答或对话本身：\n"
            f"1. 问“{words[0]}”是什么意思\n"
            f"2. 请老师用“{words[1]}”造句\n"
            f"3. 格式：“这个句子对吗？……”——句子由你编，含一个典型的学习者语法错误，用上“{words[2]}”\n"
            "4. 问一个日常英文表达用中文怎么说，格式：“How do I say ‘…’ in Chinese?”（选一个不能直译的表达）\n"
            f"5. 格式：“给我一段关于…的对话”，场景跟“{words[3]}”有关\n"
            f"6. 问“{grammar}”这个语法怎么用\n"
            "要求：口语化、简短（每个不超过25个字）。"
            '只返回一个JSON数组，包含6个字符串，例如 ["…","…","…","…","…","…"]。不要解释。'
        )
        # 600, not 400: six starters + JSON syntax can crest 400 tokens, and a
        # truncated array (no closing ]) fails extraction → needless fallback
        out = llm.create_chat_completion(
            [{"role": "user", "content": prompt}], temperature=0.8, max_tokens=600,
        )["choices"][0]["message"]["content"]
        arr = extract_json_array(out)
        starters = [s.strip() for s in arr if isinstance(s, str) and s.strip()][:6]
        if len(starters) >= 4:
            return starters
        raise ValueError(f"only {len(starters)} usable starters in: {out[:120]!r}")
    except Exception as e:  # noqa: BLE001 — starters are decoration, never block launch
        print(f"starters: generation failed ({e}); using the fallback pool", flush=True)
    return random.sample(STARTER_POOL, 6)


print("writing fresh starter prompts...", flush=True)
STARTERS = gen_starters()


# The transcript is rendered as raw HTML via gr.HTML — NOT gr.Chatbot. Gradio 6's
# Chatbot markdown pass sanitizes messages and strips the `title` attribute (and
# non-standard tags), which kills the hover gloss. gr.HTML renders our reading-layer
# markup untouched (verified: <style>, ruby/rt and attributes all survive).
#
# ---- Look: "teacher's red ink" (朱批) — a scholar's paper study ----
# Warm rice-paper page with a grain texture, ink typography (EB Garamond for
# Latin, Songti/Kaiti for hanzi), and ONE accent: cinnabar red — the color a
# Chinese teacher marks corrections in. Pinyin, the seal, the user's wash and
# every hover all draw from it. Global CSS goes to launch(css=...) (Gradio 6
# moved theme/css/js there); the theme object handles fonts + base variables.
#
# Two Gradio-specific quirks this CSS works around:
#  - Gradio ships `.gradio-container-* .prose * { color: var(--body-text-color) }`,
#    a universal selector that recolors every element and beats inherited colors —
#    the `!important` on the .msg color rules is what keeps the transcript readable.
#  - Native `title` tooltips are slow (~1s delay) and unreliable inside embedded
#    webviews, so the gloss is shown with a styled CSS tooltip instead: render_chat
#    rewrites annotate()'s title= to data-tip= and .hz:hover::after displays it.

# Paper-grain: inline SVG turbulence noise, tiled — atmosphere without an asset.
_GRAIN = (
    "data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20width='160'%20height='160'%3E"
    "%3Cfilter%20id='n'%3E%3CfeTurbulence%20type='fractalNoise'%20baseFrequency='0.9'%20numOctaves='2'/%3E"
    "%3CfeColorMatrix%20values='0%200%200%200%200%200%200%200%200%200%200%200%200%200%200%200%200%200%200.05%200'/%3E"
    "%3C/filter%3E%3Crect%20width='100%25'%20height='100%25'%20filter='url(%23n)'/%3E%3C/svg%3E"
)

PAGE_CSS = f"""
/* ---------- palette: rice paper + ink + cinnabar ---------- */
:root, .dark {{
  --paper:#f6f0e3; --sheet:#fdfaf2; --ink:#26201a; --ink-soft:#7a6d5c;
  --cinnabar:#b3302a; --cinnabar-deep:#8f241f; --wash:#f7e9e2; --hairline:#d9cdb6;
  --hanzi-serif:"Songti SC","Noto Serif SC","STSong",serif;
  --hanzi-kai:"Kaiti SC","STKaiti","Noto Serif SC",serif;
  --body-background-fill:var(--paper);
  --body-text-color:var(--ink);
  --background-fill-primary:var(--paper);
  --background-fill-secondary:var(--sheet);
  --border-color-primary:var(--hairline);
  --block-background-fill:transparent;
  color-scheme:light;
}}
/* gradio-app has an inline `background: var(--body-background-fill)` that goes
   dark under the OS dark-scheme media query — pin all three layers to paper. */
html, body, gradio-app {{ background:var(--paper) !important; }}
body::before {{
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:url("{_GRAIN}");
}}
.gradio-container {{
  background:var(--paper) !important; max-width:860px !important; margin:0 auto !important;
  font-family:"EB Garamond",var(--hanzi-serif) !important; color:var(--ink) !important;
}}
footer {{ display:none !important; }}

/* ---------- header: seal + double rule ---------- */
.hdr {{ display:flex; align-items:center; gap:16px; padding:20px 2px 16px;
       border-bottom:4px double var(--hairline); margin-bottom:14px; }}
.seal {{
  width:54px; height:54px; flex:none; display:grid; place-items:center;
  background:linear-gradient(140deg,#c23c30,var(--cinnabar-deep));
  color:var(--sheet); font-family:var(--hanzi-serif); font-size:32px; font-weight:700;
  border-radius:9px; transform:rotate(-3deg);
  box-shadow:inset 0 0 10px rgba(80,10,5,.45), 2px 3px 0 rgba(38,32,26,.12);
}}
.hdr-title {{ font-family:var(--hanzi-serif); font-size:1.7rem; line-height:1.15; color:var(--ink); }}
.hdr-title em {{ font-family:"EB Garamond",serif; font-style:italic; font-size:1.15rem; color:var(--ink-soft); }}
.hdr-sub {{ font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace; font-size:.62rem;
           letter-spacing:.14em; text-transform:uppercase; color:var(--ink-soft); margin-top:.35rem; }}
.hdr-sub b {{ color:var(--cinnabar); font-weight:600; }}

/* ---------- transcript: a manuscript sheet ---------- */
.chat {{
  display:flex; flex-direction:column; gap:1.1rem; padding:1.2rem 1.3rem 3.5rem; min-height:320px;
  background:var(--sheet); border:1px solid var(--hairline); border-radius:4px;
  box-shadow:0 1px 3px rgba(60,48,30,.08), 0 14px 34px -22px rgba(60,48,30,.35);
  /* the sheet scrolls instead of growing unbounded; the extra bottom padding
     leaves room for the hover tooltip on the last line */
  max-height:min(64vh, 680px); overflow-y:auto;
  scrollbar-width:thin; scrollbar-color:var(--hairline) transparent;
}}
.chat::-webkit-scrollbar {{ width:10px; }}
.chat::-webkit-scrollbar-track {{ background:transparent; }}
.chat::-webkit-scrollbar-thumb {{
  background:var(--hairline); border-radius:5px; border:2px solid var(--sheet);
}}
.chat::-webkit-scrollbar-thumb:hover {{ background:#c4b391; }}
@keyframes rise {{ from {{ opacity:0; transform:translateY(7px); }} }}
.b {{ max-width:84%; animation:rise .3s ease both; }}
.b.t {{ align-self:flex-start; border-left:2px solid var(--cinnabar); padding:.15rem 0 .15rem 1rem; }}
.b.u {{ align-self:flex-end; background:var(--wash); border:1px solid #e9d3c8;
       border-radius:3px; padding:.5rem .85rem; }}
.b .who {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.58rem;
          text-transform:uppercase; letter-spacing:.16em; margin-bottom:.4rem; }}
.spk.line {{ border:none; background:none; cursor:pointer; font-size:.72rem;
            padding:0 .15rem; margin-left:.35rem; opacity:.3; vertical-align:baseline;
            transition:opacity .15s; }}
.spk.line:hover {{ opacity:1; }}
.b.t .who {{ color:var(--cinnabar); }}
.b.u .who {{ color:var(--ink-soft); text-align:right; }}
.b .msg {{ font-family:"EB Garamond",var(--hanzi-kai); font-size:1.17rem; line-height:2.55; }}
.b .msg, .b .msg * {{ color:var(--ink) !important; }}
/* streaming: plain text needs no ruby headroom; cursor blinks cinnabar */
.b .msg.plain {{ line-height:1.9; }}
.cursor {{ color:var(--cinnabar) !important; animation:blink 1s steps(1) infinite; }}
@keyframes blink {{ 50% {{ opacity:0; }} }}
.b .who .note {{ text-transform:none; letter-spacing:.05em; opacity:.65; }}
/* .fill (filled-in translations, either direction) deliberately unstyled:
   they read exactly like model-authored text — the class is only a hook */
ruby {{ ruby-position:over; margin:0 .02em; }}
rt {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.42em; font-weight:500; }}
.b .msg rt {{ color:var(--cinnabar) !important; }}
.hz {{ position:relative; border-bottom:1px dotted #b49f88; cursor:pointer; }}
.hz:hover {{ background:#f3e3c9; z-index:5; }}
.hz:active {{ background:#eed9a8; }}
.hz:hover::after {{
  content:attr(data-tip);
  /* left-aligned to the word (a centered tooltip clips off-page for words near
     the sheet's left margin) */
  position:absolute; top:calc(100% + 5px); left:-.25rem;
  width:max-content; max-width:min(330px,60vw); white-space:normal;
  background:var(--ink); color:var(--paper); font-family:"EB Garamond",var(--hanzi-serif);
  font-size:.85rem; line-height:1.5; padding:.4rem .65rem; border-radius:3px;
  box-shadow:0 3px 12px rgba(38,32,26,.3); pointer-events:none; z-index:20;
}}
.fix-collect {{
  margin-top:.45rem; background:transparent; border:1px dotted var(--cinnabar);
  border-radius:2px; color:var(--cinnabar); cursor:pointer;
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.66rem;
  letter-spacing:.08em; padding:.28rem .6rem; transition:all .15s ease;
}}
.fix-collect:hover {{ background:var(--wash); border-style:solid; }}
.fix-collect.saved {{ border:1px solid #2f6e5d; color:#2f6e5d; cursor:default; background:transparent; }}
.empty {{ color:var(--ink-soft); text-align:center; padding:2rem 1rem 2.6rem;
         font-family:var(--hanzi-kai); font-size:1.05rem; }}
.empty::before {{ content:"学"; display:block; font-family:var(--hanzi-serif);
                 font-size:76px; line-height:1.3; opacity:.13; }}

/* ---------- input row + chrome ---------- */
#ask textarea {{
  background:var(--sheet) !important; border:1px solid var(--hairline) !important;
  border-radius:3px !important; box-shadow:none !important;
  font-family:"EB Garamond",var(--hanzi-kai) !important; font-size:1.08rem !important;
  color:var(--ink) !important;
}}
#ask textarea:focus {{ border-color:var(--cinnabar) !important;
                      box-shadow:0 0 0 3px rgba(179,48,42,.14) !important; }}
#ask .submit-button {{ background:var(--cinnabar) !important; color:var(--sheet) !important;
                      border:none !important; border-radius:3px !important;
                      align-self:center !important; margin-right:3px !important; }}
#ask .submit-button:hover {{ background:var(--cinnabar-deep) !important; }}
/* voice input: injected by APP_JS only when SpeechRecognition exists */
#ask, #ask .input-container {{ position:relative; }}
#ask textarea {{ padding-right:5.6rem !important; }}
.mic-btn {{
  /* vertically centered in the input bar, an 8px gap left of the send button */
  position:absolute; right:41px; top:50%; transform:translateY(-50%); z-index:5;
  width:30px; height:30px; display:grid; place-items:center; padding:0;
  background:var(--sheet); border:1px solid var(--hairline); border-radius:3px;
  color:var(--ink-soft); cursor:pointer; transition:all .15s ease;
}}
.mic-btn:hover {{ color:var(--cinnabar); border-color:var(--cinnabar); }}
@keyframes mic-pulse {{ 50% {{ box-shadow:0 0 0 6px rgba(179,48,42,.16); }} }}
.mic-btn.rec {{
  color:var(--sheet); background:var(--cinnabar); border-color:var(--cinnabar);
  animation:mic-pulse 1.1s ease-in-out infinite;
}}
.starters-row {{ display:flex; flex-wrap:wrap; gap:.45rem; align-items:center; margin-top:.55rem; }}
.starters-label {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.6rem;
                  letter-spacing:.13em; text-transform:uppercase; color:var(--ink-soft);
                  margin-right:.35rem; }}
.starter-chip {{
  background:var(--sheet); border:1px solid var(--hairline); border-radius:2px;
  color:var(--ink-soft); font-family:"EB Garamond",var(--hanzi-kai),serif;
  font-size:.95rem; padding:.35rem .7rem; cursor:pointer; transition:all .15s ease;
}}
.starter-chip:hover {{ color:var(--cinnabar); border-color:var(--cinnabar); transform:translateY(-1px); }}
#clear-btn {{ background:transparent !important; border:1px solid var(--hairline) !important;
             color:var(--ink-soft) !important; border-radius:3px !important;
             font-family:var(--hanzi-kai) !important; }}
#clear-btn:hover {{ color:var(--cinnabar) !important; border-color:var(--cinnabar) !important; }}

/* ---------- tabs + flashcards frame ---------- */
button[role="tab"] {{
  font-family:"IBM Plex Mono",ui-monospace,monospace !important; font-size:.72rem !important;
  letter-spacing:.13em; text-transform:uppercase; color:var(--ink-soft) !important;
  background:transparent !important; border-radius:0 !important;
}}
button[role="tab"][aria-selected="true"] {{
  color:var(--cinnabar) !important; border-bottom:2px solid var(--cinnabar) !important;
}}
.cards-frame {{ width:100%; height:760px; border:none; display:block; }}

/* ---------- conversation target words ---------- */
.hidden-input {{ display:none !important; }}
.targets-row {{
  display:flex; flex-wrap:wrap; gap:.7rem; align-items:baseline;
  margin:.55rem .1rem 0; font-size:1.08rem; line-height:2.1;
}}
.targets-label {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.6rem;
                 letter-spacing:.13em; text-transform:uppercase; color:var(--ink-soft); }}
.targets-row rt {{ color:var(--cinnabar); }}
.targets-row .tg {{ font-family:var(--hanzi-kai); color:var(--ink); }}

/* ---------- mode toggle ---------- */
#mode-row {{ align-items:center; gap:1rem; flex-wrap:wrap; }}
/* strip Gradio's block chrome: no border/padding, size to the label text */
#review-mode {{ flex:0 0 auto !important; width:auto !important;
               min-width:fit-content !important; border:none !important;
               background:transparent !important; padding:0 !important;
               box-shadow:none !important; overflow:visible !important; }}
#review-mode label {{ display:flex; align-items:center; gap:.45rem;
                     cursor:pointer; white-space:nowrap; }}
#review-mode .label-text {{ font-family:"IBM Plex Mono",ui-monospace,monospace !important;
                           font-size:.68rem !important; letter-spacing:.1em;
                           text-transform:uppercase; color:var(--ink-soft) !important; }}
#review-mode input {{ accent-color:var(--cinnabar); }}
#mode {{ margin:.15rem 0 .1rem; }}
#mode label {{
  background:transparent !important; border:1px solid var(--hairline) !important;
  border-radius:2px !important; color:var(--ink-soft) !important;
  font-family:"IBM Plex Mono",ui-monospace,monospace !important; font-size:.68rem !important;
  letter-spacing:.1em; text-transform:uppercase; padding:.3rem .65rem !important;
  box-shadow:none !important; cursor:pointer;
}}
#mode label.selected {{
  color:var(--cinnabar) !important; border-color:var(--cinnabar) !important;
  background:var(--sheet) !important;
}}
#mode input {{ display:none; }}

/* ---------- word-list tab ---------- */
.wl-head {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.68rem;
           letter-spacing:.13em; text-transform:uppercase; color:var(--ink-soft);
           margin:.4rem 0 .6rem; }}
.wl-head b {{ color:var(--cinnabar); }}
.wl-table {{ width:100%; border-collapse:collapse; background:var(--sheet);
            border:1px solid var(--hairline);
            box-shadow:0 1px 3px rgba(60,48,30,.08); }}
.wl-table th {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.6rem;
               letter-spacing:.12em; text-transform:uppercase; color:var(--ink-soft);
               text-align:left; padding:.55rem .75rem; border-bottom:1px solid var(--hairline); }}
.wl-table td {{ padding:.55rem .75rem; border-bottom:1px solid #ede4cd;
               font-size:.98rem; color:var(--ink); vertical-align:top; }}
.wl-table tr:last-child td {{ border-bottom:none; }}
.wl-table td.hanzi {{ font-family:var(--hanzi-kai); font-size:1.2rem; white-space:nowrap; }}
.wl-table td.hanzi.sent {{ font-size:1rem; white-space:normal; }}
.wl-table td.fixto {{ font-family:var(--hanzi-kai); color:#2f6e5d; }}
.wl-table td.py {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.82rem;
                  color:var(--cinnabar); white-space:nowrap; }}
.wl-table td.ex {{ color:var(--ink-soft); font-size:.9rem; }}
.wl-table td.st {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.75rem;
                  color:var(--ink-soft); white-space:nowrap; }}
.wl-open {{ cursor:pointer; }}
.wl-open:hover {{ color:var(--cinnabar); }}
.wl-edit {{ cursor:text; min-height:1.1em; }}
.wl-edit:hover {{ outline:1px dotted var(--hairline); outline-offset:-1px; }}
.wl-edit:focus {{ outline:2px solid var(--cinnabar); outline-offset:-2px;
                 background:var(--sheet); }}
.wl-edit:empty::before {{ content:attr(data-ph); opacity:.45; font-style:italic; }}
.wl-en {{ font-style:italic; font-size:.85rem; color:var(--ink-soft); margin-top:.15rem; }}
.wl-remove {{ border:none; background:none; color:var(--ink-soft); cursor:pointer;
             font-size:.95rem; padding:.1rem .3rem; transition:color .15s; }}
.wl-remove:hover {{ color:var(--cinnabar); }}
.wl-empty {{ color:var(--ink-soft); text-align:center; padding:2.4rem 1rem;
            font-family:var(--hanzi-kai); background:var(--sheet);
            border:1px solid var(--hairline); border-radius:4px; line-height:2; }}

/* ---------- collect toast ---------- */
#collect-toast {{
  position:fixed; right:22px; bottom:22px; z-index:100;
  background:var(--ink); color:var(--paper);
  font-family:"EB Garamond",var(--hanzi-kai),serif; font-size:.95rem;
  padding:.55rem .9rem; border-radius:3px; border-left:3px solid var(--cinnabar);
  box-shadow:0 4px 16px rgba(38,32,26,.3);
  opacity:0; transform:translateY(8px); transition:all .25s ease; pointer-events:none;
}}
#collect-toast.show {{ opacity:1; transform:translateY(0); }}
"""

# Page JS, run once at load. Injected via launch(head=...) as a self-invoking
# <script> — Gradio 6.20 ships launch(js=...) into the frontend config but never
# invokes it (verified empirically), so head= is the reliable hook.
#  - Light-only: the app is designed as paper; Gradio follows the OS and may add
#    .dark — strip it and keep it off (the :root,.dark CSS vars are the backstop).
#  - Click-to-collect: clicking any annotated word saves {word, pinyin, gloss,
#    example sentence} into the flashcards deck (localStorage, same key/schema as
#    web/flashcards.html, so the review tab reads the same deck). The transcript
#    HTML is replaced every turn, so the listener is delegated from document.
APP_JS = """
(() => {
  const root = document.documentElement;
  root.classList.remove('dark');
  new MutationObserver(() => root.classList.remove('dark'))
    .observe(root, { attributes: true, attributeFilter: ['class'] });

  const KEY = 'hsk5-tutor-deck-v1';   // shared with web/flashcards.html
  const loadDeck = () => { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch { return []; } };
  // Writing to a Gradio-bound textarea needs the native setter + an input
  // event, or Svelte's store never sees the change — one helper, used by every
  // programmatic write (ask box, deck mirror, card requests, starter chips).
  const setNative = (el, value) => {
    Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value')
      .set.call(el, value);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  };
  const toast = (msg) => {
    let t = document.getElementById('collect-toast');
    if (!t) { t = document.createElement('div'); t.id = 'collect-toast'; document.body.appendChild(t); }
    t.textContent = msg; t.classList.add('show');
    clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove('show'), 2000);
  };
  const plainText = (el) => {   // textContent with the ruby pinyin stripped
    const c = el.cloneNode(true);
    c.querySelectorAll('rt').forEach(r => r.remove());
    return c.textContent;
  };

  document.addEventListener('click', (e) => {
    const hz = e.target.closest('.hz');
    if (!hz) return;
    const front = plainText(hz).trim();
    if (!front) return;
    const deck = loadDeck();
    if (deck.some(c => c.front === front)) { toast('“' + front + '” 已在卡片里 · already in your deck'); return; }
    const tip = hz.dataset.tip || '';
    const [pinyin, gloss = ''] = tip.split(/ — (.*)/s, 2);
    // placeholder example: the sentence around the word, from the same bubble —
    // must be mostly Chinese (a quoted word inside an English sentence used to
    // slip through). Replaced by a model-written example via #card-req below.
    const msg = hz.closest('.msg');
    const sentence = msg
      ? (plainText(msg).match(/[^。！？!?\\n]*[。！？!?]?/g) || []).find(s =>
          s.includes(front) && (s.match(/[一-鿿]/g) || []).length / s.trim().length > 0.4)
      : '';
    const card = { id: front + ':' + Date.now(), front, pinyin, gloss,
                   example: (sentence || '').trim().slice(0, 120),
                   ease: 2.5, interval: 0, reps: 0, lapses: 0, due: 0 };
    deck.push(card);
    localStorage.setItem(KEY, JSON.stringify(deck));
    toast('已收藏 “' + front + '” · added to your deck (' + deck.length + ')');
    renderWordlist();
    syncDeckWords();
    pushDeckFile();
    requestExample(card);
  });

  // Ask the server to write a fresh example sentence for a new card; the reply
  // shows up in #card-res (watched by the observer below).
  const requestExample = (card) => {
    const ta = document.querySelector('#card-req textarea');
    if (ta) setNative(ta, JSON.stringify({ id: card.id, word: card.front, gloss: card.gloss || '' }));
  };

  // Mirror every deck change to the server's data/deck.json (debounced —
  // a review session rates cards in quick bursts).
  let pushT;
  const pushDeckFile = () => {
    clearTimeout(pushT);
    pushT = setTimeout(() => {
      const ta = document.querySelector('#deck-save textarea');
      if (ta) setNative(ta, localStorage.getItem(KEY) || '[]');
    }, 600);
  };
  // Restore: a browser with NO deck key at all (fresh browser / cleared site
  // data) adopts the server file. An empty-but-present deck is respected —
  // deleting your last card doesn't resurrect it on reload.
  const ensureDeckRestore = () => {
    const el = document.getElementById('deck-file');
    if (!el || el.dataset.done) return;
    el.dataset.done = '1';
    if (localStorage.getItem(KEY) !== null) return;
    const text = el.textContent.trim();
    if (!text) return;
    try {
      if (Array.isArray(JSON.parse(text))) {
        localStorage.setItem(KEY, text);
        renderWordlist();
        syncDeckWords();
      }
    } catch {}
  };
  let lastCardRes = '';
  const checkCardRes = () => {
    const el = document.getElementById('card-res');
    const text = el ? el.textContent.trim() : '';
    if (!text || text === lastCardRes) return;
    lastCardRes = text;
    let res;
    try { res = JSON.parse(text); } catch { return; }
    if (!res.id) return;
    const deck = loadDeck();
    const card = deck.find(c => c.id === res.id);
    if (!card) return;                     // removed before the model finished
    let got = [];
    if (res.example) {
      card.example = res.example;
      card.example_en = res.example_en || '';
      got.push('例句');
    }
    if (res.gloss && !card.gloss) {        // model-written definition fills an empty slot only
      card.gloss = res.gloss;
      got.push('释义');
    }
    if (!got.length) return;
    localStorage.setItem(KEY, JSON.stringify(deck));
    toast(got.join('和') + '写好了 · card filled in for “' + card.front + '”');
    renderWordlist();
    pushDeckFile();
  };

  // Correction cards: the server tags correctable tutor bubbles with a
  // .fix-collect chip; saving stores a kind:'fix' card (front = the student's
  // wrong sentence, back = the fix + rule) in the same deck/scheduler.
  document.addEventListener('click', (e) => {
    const chip = e.target.closest('.fix-collect');
    if (!chip) return;
    const { wrong, fix, why } = chip.dataset;
    if (!wrong || !fix) return;
    const deck = loadDeck();
    if (deck.some(c => c.kind === 'fix' && c.front === wrong && c.fix === fix)) {
      toast('这条纠错已经在卡片里 · correction already saved');
      chip.classList.add('saved');
      return;
    }
    deck.push({ id: 'fix:' + Date.now(), kind: 'fix', front: wrong, fix, why: why || '',
                ease: 2.5, interval: 0, reps: 0, lapses: 0, due: 0 });
    localStorage.setItem(KEY, JSON.stringify(deck));
    toast('已收藏纠错 · correction saved (' + deck.length + ')');
    chip.classList.add('saved');
    chip.textContent = '✓ 已收藏 · saved';
    renderWordlist();
    pushDeckFile();
  });

  // ---- word-list tab: table of the collected deck, with per-row removal.
  // Re-rendered whenever the deck changes: after a collect (above), after a
  // removal (below), and on `storage` events from the flashcards iframe
  // (ratings, manual adds, resets). Removals propagate back the same way.
  const escHtml = (s) => String(s).replace(/[&<>"]/g,
    (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]));
  const renderWordlist = () => {
    const el = document.getElementById('wordlist');
    if (!el) return;
    // don't re-render out from under an in-progress edit
    if (document.activeElement && document.activeElement.closest
        && document.activeElement.closest('.wl-edit')) return;
    const deck = loadDeck();
    if (!deck.length) {
      el.innerHTML = '<div class="wl-empty">生词表是空的 — 在对话里点一个词就能收藏。<br>' +
        'Nothing collected yet — click any word in the chat to add it.</div>';
      return;
    }
    // gloss/example/translation (fix/why on correction cards) are editable in place
    const editable = (c, field, extra) =>
      ' class="wl-edit' + (extra ? ' ' + extra : '') + '" contenteditable="true" spellcheck="false"'
      + ' data-fid="' + escHtml(c.id) + '" data-field="' + field + '"'
      + ' title="点击编辑 · click to edit"';
    const edit = (c, field, cls, val) =>
      '<td' + editable(c, field, cls) + '>' + escHtml(val || '') + '</td>';
    const rows = [...deck].reverse().map(c => {
      const open = ' class="wl-open" data-fid="' + escHtml(c.id)
        + '" title="打开卡片 · open this card"';
      const cells = c.kind === 'fix'
        ? '<td class="hanzi sent"><span' + open + '>' + escHtml(c.front) + '</span></td>'
          + '<td class="py">改错</td>'
          + edit(c, 'fix', 'fixto', c.fix)
          + edit(c, 'why', 'ex', c.why)
        : '<td class="hanzi"><span' + open + '>' + escHtml(c.front) + '</span></td>'
          + '<td class="py">' + escHtml(c.pinyin || '') + '</td>'
          + edit(c, 'gloss', '', c.gloss)
          // example + its translation as two separately-editable blocks
          + '<td class="ex"><div' + editable(c, 'example') + '>' + escHtml(c.example || '') + '</div>'
          + '<div' + editable(c, 'example_en', 'wl-en') + ' data-ph="translation…">'
          + escHtml(c.example_en || '') + '</div></td>';
      return '<tr>' + cells
        + '<td class="st">' + (c.reps > 0 ? c.reps + '×' : 'new') + '</td>'
        + '<td><button class="wl-remove" title="移除 · remove" data-fid="'
        + escHtml(c.id) + '">✕</button></td></tr>';
    }).join('');
    el.innerHTML =
      '<div class="wl-head">生词表 · collected words <b>' + deck.length + '</b></div>'
      + '<table class="wl-table"><thead><tr><th>词</th><th>拼音</th><th>释义</th>'
      + '<th>例句</th><th>复习</th><th></th></tr></thead><tbody>' + rows + '</tbody></table>';
  };
  // in-place edits: save on blur; Enter commits instead of inserting a newline
  document.addEventListener('focusout', (e) => {
    const td = e.target.closest('.wl-edit');
    if (!td) return;
    const deck = loadDeck();
    const card = deck.find(c => c.id === td.dataset.fid);
    if (!card) return;
    const val = td.textContent.trim();
    if ((card[td.dataset.field] || '') === val) return;
    card[td.dataset.field] = val;
    localStorage.setItem(KEY, JSON.stringify(deck));
    toast('已保存 · saved');
    pushDeckFile();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.closest('.wl-edit')) {
      e.preventDefault();
      e.target.closest('.wl-edit').blur();
    }
  });

  // clicking a word in the list opens its card in the flashcards tab; the
  // message is sent twice because a first-ever visit mounts the iframe lazily
  document.addEventListener('click', (e) => {
    const word = e.target.closest('.wl-open');
    if (!word) return;
    [...document.querySelectorAll('button[role="tab"]')]
      .find(t => t.textContent.includes('卡片'))?.click();
    const show = () => document.querySelector('.cards-frame')
      ?.contentWindow?.postMessage({ type: 'show-card', id: word.dataset.fid }, '*');
    setTimeout(show, 250);
    setTimeout(show, 900);
  });

  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.wl-remove');
    if (!btn) return;
    const deck = loadDeck();
    const card = deck.find(c => c.id === btn.dataset.fid);
    localStorage.setItem(KEY, JSON.stringify(deck.filter(c => c.id !== btn.dataset.fid)));
    toast('已移除 “' + (card ? card.front : '') + '” · removed');
    renderWordlist();
    syncDeckWords();
    pushDeckFile();
  });

  // Mirror the deck's word fronts (due-first, capped) into the hidden #deck-words
  // textbox so the server's pick_targets() can prefer the student's own words.
  const syncDeckWords = () => {
    const ta = document.querySelector('#deck-words textarea');
    if (!ta) return;
    const now = Date.now();
    const words = loadDeck().filter(c => c.kind !== 'fix');  // fix fronts are sentences
    setNative(ta, JSON.stringify({
      due: words.filter(c => c.due <= now).map(c => c.front).slice(0, 30),
      other: words.filter(c => c.due > now).map(c => c.front).slice(0, 30),
    }));
  };
  window.addEventListener('storage', (e) => {
    if (e.key !== KEY) return;
    renderWordlist();
    syncDeckWords();
    pushDeckFile();
  });
  // The mirror's real guarantee: re-sync when the ask box gains focus — the
  // user focuses before typing, giving Gradio's (async) store update seconds
  // to settle before submit. (A submit-instant sync was tested and loses the
  // race: the DOM updates but Gradio snapshots its store value first.)
  document.addEventListener('focusin', (e) => {
    if (e.target.closest('#ask textarea')) syncDeckWords();
  });

  // Browser TTS (same voice logic as web/flashcards.html): each tutor bubble
  // has a .spk button whose data-speak carries the Chinese-only text.
  let zhVoice = null;
  const pickVoice = () => {
    const vs = window.speechSynthesis ? speechSynthesis.getVoices() : [];
    zhVoice = vs.find(v => /^zh/i.test(v.lang)) || null;
  };
  if (window.speechSynthesis) { pickVoice(); speechSynthesis.onvoiceschanged = pickVoice; }
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.spk');
    if (!btn || !window.speechSynthesis) return;
    if (speechSynthesis.speaking) { speechSynthesis.cancel(); return; }   // click again to stop
    const u = new SpeechSynthesisUtterance(btn.dataset.speak || '');
    u.lang = 'zh-CN'; u.rate = 0.9; if (zhVoice) u.voice = zhVoice;
    speechSynthesis.speak(u);
  });

  // ---- voice input: Web Speech API (Chrome; the button only appears when the
  // API exists). Click to talk in Mandarin — interim results stream into the
  // ask box (appended to whatever's typed), stop on click or when you pause.
  // Recognition text is never auto-submitted: mis-hearings should be read (and
  // fixed) by the learner before they go to the tutor.
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let rec = null;
  const setAsk = (text) => {
    const ta = document.querySelector('#ask textarea');
    if (ta) setNative(ta, text);
  };
  const ensureMic = () => {
    if (!SR) return;
    // anchor inside the input bar itself (the submit button's container) so
    // top:50% centers against the bar, not the padded outer block
    const box = document.querySelector('#ask .input-container') || document.querySelector('#ask');
    if (!box || box.querySelector('.mic-btn')) return;
    const btn = document.createElement('button');
    btn.className = 'mic-btn'; btn.type = 'button';
    btn.innerHTML = '<svg viewBox="0 0 24 24" width="15" height="15" fill="none"'
      + ' stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
      + '<rect x="9" y="2" width="6" height="12" rx="3"/>'
      + '<path d="M5 10v1a7 7 0 0 0 14 0v-1"/><path d="M12 18v3"/></svg>';
    btn.title = '点一下，说中文 · click and speak Chinese';
    box.appendChild(btn);
    btn.addEventListener('click', () => {
      if (rec) { rec.stop(); return; }        // click again to stop
      rec = new SR();
      rec.lang = 'zh-CN'; rec.interimResults = true; rec.continuous = false;
      const ta = document.querySelector('#ask textarea');
      const base = ta ? ta.value : '';
      rec.onresult = (e) => {
        let heard = '';
        for (const r of e.results) heard += r[0].transcript;
        setAsk(base + heard);
      };
      rec.onerror = (e) => {
        const why = { 'not-allowed': '需要麦克风权限 · mic permission needed',
                      'no-speech': '没听到声音 · heard nothing' }[e.error] || e.error;
        toast('语音输入 · voice input: ' + why);
      };
      rec.onend = () => {
        rec = null; btn.classList.remove('rec');
        const t = document.querySelector('#ask textarea');
        if (t) t.focus();
      };
      btn.classList.add('rec');
      rec.start();
    });
  };

  // Starter chips: a fresh random six from the pool on every page load.
  const STARTER_POOL = __STARTERS__;
  const ensureStarters = () => {
    const row = document.getElementById('starters');
    if (!row || row.dataset.filled) return;
    row.dataset.filled = '1';
    const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
    const picks = [...STARTER_POOL].sort(() => Math.random() - 0.5).slice(0, 6);
    row.innerHTML = '<span class="starters-label">试一试 · try one</span>'
      + picks.map(p => '<button class="starter-chip">' + esc(p) + '</button>').join('');
  };
  document.addEventListener('click', (e) => {
    const chip = e.target.closest('.starter-chip');
    if (!chip) return;
    const ta = document.querySelector('#ask textarea');
    if (!ta) return;
    setNative(ta, chip.textContent);
    ta.focus();
  });

  // 问老师 from a flashcard back (the cards iframe posts {type:'ask-tutor'}):
  // switch to the chat tab, fill the ask box, submit. Only messages from OUR
  // iframe are honored (e.source check — srcdoc frames have origin 'null', so
  // source identity is the usable credential). This path submits without the
  // user ever focusing the ask box, so it must sync the deck mirror itself —
  // first thing, giving the store the full 600ms of the two beats below to
  // settle (the same lead-time lesson as the focusin listener above).
  window.addEventListener('message', (e) => {
    const cards = document.querySelector('.cards-frame');
    if (!cards || e.source !== cards.contentWindow) return;
    if (!e.data || e.data.type !== 'ask-tutor' || typeof e.data.text !== 'string') return;
    syncDeckWords();
    const chatTab = [...document.querySelectorAll('button[role="tab"]')]
      .find(t => t.textContent.includes('对话'));
    if (chatTab) chatTab.click();
    setTimeout(() => {
      setAsk(e.data.text);
      setTimeout(() => document.querySelector('#ask .submit-button')?.click(), 300);
    }, 300);
  });

  // The transcript is capped (overflow-y): jump to the newest message when one
  // arrives. Three traps, all hit in testing:
  //  - Gradio 6 patches the .chat node IN PLACE on update (it is not replaced),
  //    so new messages are detected by bubble count — which also means user
  //    scroll position survives everything except a new message, as it should;
  //  - this script runs in <head>, where document.body is still null — the
  //    observer must be installed after DOM ready or observe() throws;
  //  - the jump must repeat as layout settles: mutations fire pre-layout, and
  //    the web-font swap can grow scrollHeight again afterwards.
  // The same observer fills the starter row once Gradio mounts it (the Svelte
  // app renders well after DOMContentLoaded).
  const installObserver = () => {
    let lastCount = -1;
    const jumpToNewest = (chat) => {
      const jump = () => { chat.scrollTop = chat.scrollHeight; };
      requestAnimationFrame(jump);
      setTimeout(jump, 120);
      setTimeout(jump, 500);
    };
    // one-shot initial fill once Gradio mounts the container (guarded via
    // data-filled so our own innerHTML writes don't re-trigger through the
    // observer)
    const ensureWordlist = () => {
      const el = document.getElementById('wordlist');
      if (!el || el.dataset.filled) return;
      el.dataset.filled = '1';
      renderWordlist();
    };
    const ensureDeckSync = () => {
      const ta = document.querySelector('#deck-words textarea');
      if (!ta || ta.dataset.synced) return;
      ta.dataset.synced = '1';
      // best-effort early sync; the real guarantees are the focusin listener
      // on the ask box and the ask-tutor message handler (both re-sync with
      // lead time before their submits)
      syncDeckWords();
    };
    let lastHeight = 0;
    new MutationObserver(() => {
      ensureStarters();
      ensureWordlist();
      ensureDeckSync();
      ensureDeckRestore();
      ensureMic();
      checkCardRes();
      const chat = document.querySelector('.chat');
      if (!chat) return;
      if (chat.childElementCount !== lastCount) {
        lastCount = chat.childElementCount;
        jumpToNewest(chat);
      } else if (chat.scrollHeight !== lastHeight
                 && chat.scrollHeight - chat.scrollTop - chat.clientHeight < 160) {
        // streaming growth: stay pinned to the bottom unless the user scrolled up
        chat.scrollTop = chat.scrollHeight;
      }
      lastHeight = chat.scrollHeight;
    }).observe(document.body, { childList: true, subtree: true });
    ensureStarters();
    ensureWordlist();
    ensureDeckSync();
    ensureDeckRestore();
    ensureMic();
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', installObserver);
  } else {
    installObserver();
  }
})();
"""

HEAD_HTML = f"<script>{APP_JS.replace('__STARTERS__', json.dumps(STARTERS, ensure_ascii=False))}</script>"

HEADER_HTML = """
<div class="hdr">
  <div class="seal">文</div>
  <div>
    <div class="hdr-title">HSK-5 中文 <em>tutor</em></div>
    <div class="hdr-sub">pinyin above every character · <b>hover a word for its meaning · click to collect it</b></div>
  </div>
</div>
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#fbeae8", c100="#f6d5d1", c200="#eaaba3", c300="#dd8175",
        c400="#cc5a4c", c500="#b3302a", c600="#9c2a25", c700="#8f241f",
        c800="#731d19", c900="#571613", c950="#3a0f0d",
    ),
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("EB Garamond"), "Songti SC", "serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
)


# For the per-bubble TTS button: keep only the Chinese of a bilingual reply
# (per line-with-CJK, drop the Latin words) so the zh voice doesn't wade
# through the English translations.
_CJK_CHUNK = re.compile(r"[一-鿿]+[0-9，。！？、；：]*")


def chinese_only(text: str) -> str:
    lines = (line for line in text.split("\n") if HAS_CJK.search(line))
    return " ".join("".join(_CJK_CHUNK.findall(line)) for line in lines)


# A tutor correction: 应该说“正确的句子”，因为解释… (the trained shape) or
# 解释…，所以要说“正确的句子”。 The corrected sentence is the quoted group; the
# explanation is the rest of the sentence around the match (either side), plus
# the following sentence when it's the (mostly-ASCII) English rule. The
# (?<![不别]) guard skips 不要说/别说“错的” so we never capture the wrong
# version as the fix.
_FIX_RE = re.compile(r"(?<![不别])(?:应该|要)\s*说\s*[“\"]([^”\"]+)[”\"]")
_SENT_END = re.compile(r"[。！？!?.]")
# Praise/agreement in the same sentence means the quote is NOT a correction
# (你说得对，应该说“…”也可以) — no chip.
_NOT_A_FIX = re.compile(r"说得对|说得很好|没问题|没有错|是对的|很自然|用得很好")


def _mostly_ascii(s: str) -> bool:
    letters = [ch for ch in s if not ch.isspace()]
    return bool(letters) and sum(ch.isascii() for ch in letters) / len(letters) > 0.6


def extract_correction(reply: str, wrong: str) -> tuple[str, str] | None:
    """(corrected sentence, explanation) if the reply corrects `wrong`, the
    student's previous message.

    Guards against false chips on benign 应该说/要说 quoting: a real correction
    REVISES the student's sentence, so the quoted fix must be built mostly from
    the student's own characters, and the sentence must not be praise."""
    m = _FIX_RE.search(reply)
    if not m:
        return None
    ends = [e.end() for e in _SENT_END.finditer(reply, 0, m.start())]
    sent_start = ends[-1] if ends else 0
    end_m = _SENT_END.search(reply, m.end())
    sent_end = end_m.start() if end_m else len(reply)
    if _NOT_A_FIX.search(reply[sent_start:sent_end]):
        return None
    fix_chars = {ch for ch in m.group(1) if HAS_CJK.match(ch)}
    if len(fix_chars) < 2 or not wrong:
        return None
    if len(fix_chars & set(wrong)) / len(fix_chars) < 0.6:
        return None
    pre = reply[sent_start:m.start()].strip().strip("，、： ")
    if len(pre) <= 3:   # a bare subject ("你"/"这里"), not an explanation
        pre = ""
    post = reply[m.end():sent_end].strip().strip("，、： ")
    why = "，".join(p for p in (pre, post) if p)
    for tail in ("，所以要", "，所以", "所以要", "所以"):   # leftovers of "…，所以要说“X”"
        if why.endswith(tail):
            why = why[: -len(tail)]
            break
    if end_m:  # append the English rule when the next sentence is mostly ASCII
        rest = reply[end_m.end():]
        nxt_end = _SENT_END.search(rest)
        nxt = (rest[: nxt_end.start()] if nxt_end else rest).strip()
        if _mostly_ascii(nxt):
            why = f"{why}。{nxt}" if why else nxt
    return m.group(1), why[:200]


# --------------------------------------------------------------------------- #
# In-context sense disambiguation (地道: tunnel or authentic?). One extra short
# generation per turn, only when the new messages contain ambiguous words. The
# result feeds annotate() overrides, so tooltip AND ruby pinyin reflect the
# sense actually used. Failures fall back silently to the dictionary tips.
# --------------------------------------------------------------------------- #
_SENTENCE_SPLIT = re.compile(r"[。！？!?\n]")


def _tipped(text: str, ov: dict[str, tuple[list[str], str]] | None = None) -> str:
    """annotate() for in-app rendering: the native `title` tooltip (right for
    the standalone docs/ pages) becomes our styled CSS tooltip's data-tip.
    Safe: message text is html-escaped, so ` title="` can only come from
    annotate()'s own .hz spans."""
    return annotate(text, ov).replace(' title="', ' data-tip="')


def _sentence_around(text: str, word: str) -> str:
    for chunk in _SENTENCE_SPLIT.split(text):
        if word in chunk:
            return chunk.strip()[:80]
    return text.strip()[:80]


def disambiguate(texts: list[str]) -> dict[str, tuple[list[str], str]]:
    """One model call → tooltip overrides {word: (syllables, gloss)} for the
    new messages: sense PICKS for ambiguous words (地道: option number) and
    written DEFINITIONS for words the dictionary lacks entirely (一只). One
    flat JSON does both — a number means a pick, a string means a definition."""
    sense: dict[str, tuple[str, list[dict]]] = {}   # word -> (sentence, options)
    defs: dict[str, str] = {}                       # word -> sentence
    for text in texts:
        for w in ambiguous_words(text):
            if w not in sense and len(sense) < 6:
                sense[w] = (_sentence_around(text, w), sense_options(w))
        for w in unglossed_words(text):
            if w not in defs and len(defs) < 4:
                defs[w] = _sentence_around(text, w)
    if not sense and not defs:
        return {}
    blocks = []
    for w, (sentence, opts) in sense.items():
        lines = "\n".join(f"{i + 1}. {o['reading']} — {o['brief']}" for i, o in enumerate(opts))
        blocks.append(f"词：{w}\n句子：{sentence}\n{lines}")
    if defs:
        blocks.append(
            "下面的词没有选项——给每个词写一个简短的英文释义（词典格式，"
            "比如 to meet; to see each other）：\n"
            + "\n".join(f"词：{w}（句子：{s}）" for w, s in defs.items())
        )
    ex_pick = f'"{next(iter(sense))}": 1' if sense else ""
    ex_def = f'"{next(iter(defs))}": "a short English definition"' if defs else ""
    example = ", ".join(x for x in (ex_pick, ex_def) if x)
    prompt = (
        "你是词典助手。有选项的词：判断它在句子里用的是哪个意思，返回选项编号。"
        "没有选项的词：直接写英文释义。\n\n"
        + "\n\n".join(blocks)
        + "\n\n只返回一个JSON对象，例如 {" + example + "}。不要解释。"
    )
    try:
        out = llm.create_chat_completion(
            [{"role": "user", "content": prompt}], temperature=0, max_tokens=220,
        )["choices"][0]["message"]["content"]
        picks = extract_json_object(out)
        overrides = {}
        for w, v in picks.items():
            if w in sense and isinstance(v, (int, str)) and str(v).isdigit():
                opts = sense[w][1]
                i = int(v) - 1
                if 0 <= i < len(opts):
                    overrides[w] = (opts[i]["syllables"], opts[i]["gloss"])
            elif w in defs and isinstance(v, str) and v.strip() and _mostly_ascii(v):
                # empty syllables keep the pypinyin ruby; only the gloss changes
                overrides[w] = ([], v.strip()[:80])
        return overrides
    except Exception:  # noqa: BLE001 — a failed pick just keeps dictionary tips
        return {}


# --------------------------------------------------------------------------- #
# Q&A translation fill: the tuned model SOMETIMES skips the English line under
# a Chinese sentence. Rather than retrain (or bolt on a cloud API), one extra
# local call translates just the missed lines — only on turns that have any.
# --------------------------------------------------------------------------- #
_CJK_COUNT = re.compile(r"[一-鿿]")
_INLINE_EN = re.compile(r"[A-Za-z][A-Za-z ,.'’!?;:-]{7,}")
_NUMBERED = re.compile(r"\s*(\d+)[.、)]\s*(.+)")


def untranslated_lines(reply: str) -> dict[int, str]:
    """{line index: Chinese line} for substantial Chinese lines with no inline
    English and no mostly-English line following them."""
    lines = reply.split("\n")
    out: dict[int, str] = {}
    for i, line in enumerate(lines):
        s = line.strip()
        if len(_CJK_COUNT.findall(s)) < 4:      # empty / label / fragment
            continue
        if _INLINE_EN.search(s):                # translated inline already
            continue
        nxt = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
        if nxt and _mostly_ascii(nxt):          # translated on the next line
            continue
        out[i] = s
    return out


def fill_translations(reply: str) -> dict[int, str]:
    """One model call → {line index: English} for the reply's missed lines."""
    missing = untranslated_lines(reply)
    if not missing:
        return {}
    numbered = "\n".join(f"{i}. {s}" for i, s in missing.items())
    prompt = (
        "把下面每行中文翻译成英文。保持行号，逐行返回，格式：行号. 英文翻译。"
        "不要拼音，不要解释。\n\n" + numbered
    )
    try:
        out = llm.create_chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=0, max_tokens=60 * len(missing) + 40,
        )["choices"][0]["message"]["content"]
        fills = {}
        for line in out.splitlines():
            m = _NUMBERED.match(line)
            if m and int(m.group(1)) in missing and _mostly_ascii(m.group(2)):
                fills[int(m.group(1))] = m.group(2).strip()[:200]
        return fills
    except Exception:  # noqa: BLE001 — a failed fill just leaves the reply as-is
        return {}


def translate_user_to_chinese(user_msg: str) -> str:
    """Natural HSK-5 Chinese for an English prompt — if the student is asking
    in English, they probably don't know how to say it in Chinese yet, so the
    transcript shows them (annotated, collectible) under their own bubble."""
    prompt = (
        "把下面这句话翻译成自然的中文（HSK5水平的说法）。"
        "只返回中文翻译，不要拼音，不要解释。\n\n" + user_msg.strip()
    )
    try:
        out = llm.create_chat_completion(
            [{"role": "user", "content": prompt}], temperature=0, max_tokens=120,
        )["choices"][0]["message"]["content"]
        line = next((l.strip() for l in out.splitlines() if l.strip()), "")
        line = _UNLABEL.sub("", line).strip().strip('"“”')
        return line[:200] if HAS_CJK.search(line) else ""
    except Exception:  # noqa: BLE001 — no translation is just the old behavior
        return ""


def _spk(line: str) -> str:
    """A small per-line TTS button voicing just that line's Chinese."""
    return (f'<button class="spk line" title="朗读这一行 · read this line aloud"'
            f' data-speak="{html.escape(chinese_only(line), quote=True)}">🔊</button>')


def _render_msg(m: dict, tutor: bool) -> str:
    """A message's display HTML: annotated text line by line, a 🔊 per Chinese
    line (tutor lines, and Chinese fills anywhere), plus any filled-in
    translations under their lines. Fills go through the reading layer too —
    an English fill passes through untouched, a Chinese one (the translation
    of the student's English prompt) gets ruby + glosses + click-to-collect."""
    ov, fills = m.get("tips"), m.get("fills") or {}
    parts = []
    for i, line in enumerate(m["content"].split("\n")):
        h = _tipped(line, ov)
        if tutor and HAS_CJK.search(line):
            h += _spk(line)
        parts.append(h)
        if i in fills:
            f = _tipped(fills[i], ov)
            if HAS_CJK.search(fills[i]):
                f += _spk(fills[i])
            parts.append(f'<span class="fill">{f}</span>')
    return "<br>".join(parts)


def render_chat(raw: list[dict]) -> str:
    """Build the whole transcript as one self-contained HTML block, annotating every
    Chinese span (pinyin ruby + hover gloss). A message's disambiguation
    overrides travel ON the message dict (m["tips"]) so they can never
    misalign; strip_for_llm() removes them before generation."""
    bubbles = []
    for i, m in enumerate(raw):
        u = m["role"] == "user"
        who = "You" if u else "老师 Tutor"
        msg = _render_msg(m, tutor=not u)
        # When the tutor corrected the previous student turn, offer to save the
        # correction as a flashcard (front: the student's sentence, back: the
        # fix + rule). APP_JS handles the click; the data- attrs carry the card.
        chip = ""
        if not u and i and raw[i - 1]["role"] == "user":
            fix = extract_correction(m["content"], raw[i - 1]["content"])
            if fix:
                chip = (
                    f'<button class="fix-collect"'
                    f' data-wrong="{html.escape(raw[i - 1]["content"].strip()[:160], quote=True)}"'
                    f' data-fix="{html.escape(fix[0], quote=True)}"'
                    f' data-why="{html.escape(fix[1], quote=True)}">'
                    f'✚ 收藏纠错 · save correction</button>'
                )
        bubbles.append(
            f'<div class="b {"u" if u else "t"}"><div class="who">{who}</div>'
            f'<div class="msg">{msg}</div>{chip}</div>'
        )
    inner = "".join(bubbles) or '<div class="empty">用中文或英文问我… (ask me anything)</div>'
    return f'<div class="chat">{inner}</div>'


# Target words for conversation mode: the student's own collected flashcard
# words first (the deck lives client-side, so APP_JS mirrors the word list into
# a hidden textbox), padded to 5 with random HSK-5 seeds (HSK_VOCAB, loaded at
# the top). Picked once per conversation and kept until 清空 so the tutor can
# keep circling back to them.
def pick_targets(deck_json: str, review_only: bool = False) -> list[str]:
    """Conversation target words. Normal mode: up to 3 deck words (due first)
    padded to 5 with random HSK-5 seeds. Review mode: ONLY due cards — the
    conversation becomes the SRS review session (falls back to normal when
    nothing is due)."""
    def clean(ws):
        return [w for w in ws if isinstance(w, str) and w.strip()]
    try:
        data = json.loads(deck_json or "{}")
    except (json.JSONDecodeError, TypeError):
        data = {}
    if isinstance(data, list):                       # pre-split mirror shape
        due, other = clean(data), []
    else:
        due, other = clean(data.get("due", [])), clean(data.get("other", []))
    if review_only and due:
        return due[:5]
    targets = (due + other)[:3]
    pool = [w for w in HSK_VOCAB if w not in targets]
    targets += random.sample(pool, k=min(5 - len(targets), len(pool)))
    return targets


def render_targets(targets: list[str] | None) -> str:
    if not targets:
        return ""
    chips = "".join(f'<span class="tg">{_tipped(w)}</span>' for w in targets)
    return f'<div class="targets-row"><span class="targets-label">目标词 · practice</span>{chips}</div>'


def strip_for_llm(raw: list[dict]) -> list[dict]:
    """History for the model: role/content only — display metadata like the
    per-message "tips" overrides never reaches the chat template."""
    return [{"role": m["role"], "content": m["content"]} for m in raw]


def respond(user_msg: str, raw: list[dict], mode: str, deck_json: str,
            targets: list[str] | None, review_only: bool):
    """user message + raw history → model reply, STREAMED.

    A generator: tokens render live in a plain bubble (the reading layer needs
    whole words, so annotating a half-generated sentence would misparse), then
    a short 加注中 pass runs disambiguation and translation fills, and the
    final yield swaps in the fully annotated transcript.
    Yields (chat HTML, raw history, cleared box, targets state, targets bar)."""
    conversational = "聊天" in mode
    bar = lambda: render_targets(targets if conversational else None)  # noqa: E731
    if not user_msg.strip():
        yield (render_chat(raw), raw, "", targets, bar())
        return
    if conversational:
        if not targets:
            targets = pick_targets(deck_json, review_only)
        # shared with gen_data.py so training data and serving use the same prompt
        system = c.conversation_system(targets)
    else:
        system = c.SYSTEM_PROMPT_APP
    raw = raw + [{"role": "user", "content": user_msg}]
    messages = [{"role": "system", "content": system}] + strip_for_llm(raw)

    # history annotated once; the in-progress reply rides in a plain bubble
    base = render_chat(raw)[: -len("</div>")]

    def with_stream_bubble(text: str, note: str = "") -> str:
        inner = html.escape(text).replace("\n", "<br>")
        cursor = "" if note else '<span class="cursor">▌</span>'
        who = "老师 Tutor" + (f'<span class="note"> · {note}</span>' if note else "")
        return (base + f'<div class="b t"><div class="who">{who}</div>'
                f'<div class="msg plain">{inner}{cursor}</div></div></div>')

    reply = ""
    try:
        n = 0
        for chunk in llm.create_chat_completion(
            messages, temperature=0.7, max_tokens=512, stream=True,
        ):
            delta = chunk["choices"][0]["delta"].get("content")
            if not delta:
                continue
            reply += delta
            n += 1
            if n % 8 == 0:
                yield (with_stream_bubble(reply), raw, "", targets, bar())
    except Exception:  # noqa: BLE001 — keep whatever streamed before the error
        pass
    reply = reply.strip() or "（生成失败，请再试一次 · generation failed — try again）"
    raw = raw + [{"role": "assistant", "content": reply}]

    # reading-layer pass: sense picks, translation fills, prompt translation
    yield (with_stream_bubble(reply, note="加注中 annotating"), raw, "", targets, bar())
    ov = disambiguate([user_msg, reply])
    if ov:
        raw[-2]["tips"] = ov
        raw[-1]["tips"] = ov
    if not conversational:   # 聊天 mode is Chinese-only by design
        fills = fill_translations(reply)
        if fills:
            raw[-1]["fills"] = fills
    # an English prompt gets its Chinese under the student's own bubble
    if _mostly_ascii(user_msg) and len(user_msg.strip()) >= 8:
        zh = translate_user_to_chinese(user_msg)
        if zh:
            raw[-2]["fills"] = {user_msg.count("\n"): zh}
    yield (render_chat(raw), raw, "", targets, bar())


# List markers / labels the model sometimes prefixes to its lines ("1. ", "例句：")
_UNLABEL = re.compile(r"^\s*(?:\d+[.、)]|[-•*]|例句[:：]?|翻译[:：]?|Translation[:：]?)\s*")
# A trailing run of English glued onto the example line (我很高兴，I am happy):
# stripped so the zh TTS never voices it — the translation belongs in example_en.
_TRAILING_EN = re.compile(r"[，,\s]*[A-Za-z][A-Za-z0-9 ,.'’!?;:-]{7,}[.!?]?\s*$")


def gen_card_example(req_json: str) -> str:
    """Write a fresh HSK-5 example sentence for a just-collected flashcard —
    and, when the dictionary had no gloss for the word (CEDICT lacks many
    jieba compounds like 一只/很累), a brief English definition too.

    Called through a hidden request textbox (APP_JS writes {id, word, gloss}
    on collect); the result lands in a hidden HTML div the page JS observes,
    and fills the card. Runs on the same llm/queue as chat, so it simply
    waits its turn behind a generation."""
    try:
        req = json.loads(req_json)
        card_id, word, gloss = req["id"], req["word"], req.get("gloss", "")
    except (json.JSONDecodeError, KeyError, TypeError):
        return ""
    if gloss:
        prompt = (
            f"用“{word}”（意思：{gloss}）写一个HSK5水平的简单例句，能自然地体现这个词的意思。"
            "第一行：例句（不要拼音）。第二行：这个例句的英文翻译。只返回这两行，不要解释。"
        )
    else:
        prompt = (
            f"“{word}”是一个中文词或词组。\n"
            f"第一行：它的简短英文释义（词典格式，比如 to meet; to see each other）。\n"
            f"第二行：用“{word}”写一个HSK5水平的简单例句（不要拼音）。\n"
            "第三行：例句的英文翻译。只返回这三行，不要解释。"
        )
    out = llm.create_chat_completion(
        [{"role": "user", "content": prompt}], temperature=0.7, max_tokens=160,
    )["choices"][0]["message"]["content"].strip()
    lines = [_UNLABEL.sub("", line).strip().strip('"“”')
             for line in out.splitlines() if line.strip()]
    # the example is the first Chinese line containing the word; the definition
    # (only asked for when the dictionary had none) is an English line BEFORE
    # it, the translation an English line AFTER it
    ex_idx = next((i for i, l in enumerate(lines) if word in l and HAS_CJK.search(l)), None)
    resp: dict[str, str] = {"id": card_id}
    if ex_idx is not None:
        example = lines[ex_idx][:120]
        trimmed = _TRAILING_EN.sub("", example).strip()
        if trimmed and word in trimmed:      # keep the trim only if it leaves a valid example
            example = trimmed
        resp["example"] = example
        resp["example_en"] = next(
            (l[:160] for l in lines[ex_idx + 1:] if _mostly_ascii(l)), "")
    if not gloss:
        head = lines[:ex_idx] if ex_idx is not None else lines
        resp["gloss"] = next((l[:80] for l in head if _mostly_ascii(l)), "")
    if len(resp) == 1:                       # nothing usable — keep the scraped card
        return ""
    return html.escape(json.dumps(resp, ensure_ascii=False))


# File-backed deck: localStorage is per-browser and clearable, so every deck
# change is mirrored to data/deck.json (git-ignored, lives with the repo). On
# page load, a browser with NO deck at all restores from the file — switching
# browsers or clearing site data no longer loses the cards.
DECK_FILE = c.ROOT / "data" / "deck.json"


def save_deck(deck_json: str) -> None:
    try:
        if not isinstance(json.loads(deck_json), list):
            return
    except (json.JSONDecodeError, TypeError):
        return
    DECK_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DECK_FILE.with_suffix(".json.tmp")
    tmp.write_text(deck_json, encoding="utf-8")
    tmp.replace(DECK_FILE)


def deck_restore_html() -> str:
    """Contents of the deck file, re-read on every page load (the component
    value is this callable), so a fresh browser restores the latest deck."""
    try:
        if DECK_FILE.exists():
            return f'<div id="deck-file">{html.escape(DECK_FILE.read_text(encoding="utf-8"))}</div>'
    except OSError:
        pass
    return '<div id="deck-file"></div>'


def flashcards_srcdoc() -> str:
    """Embed web/flashcards.html as an <iframe srcdoc>. srcdoc iframes are
    same-origin, so the widget shares the click-to-collect localStorage deck and
    `storage` events keep it live-synced — no static-file routes needed."""
    page = (c.ROOT / "web" / "flashcards.html").read_text(encoding="utf-8")
    return f'<iframe class="cards-frame" title="flashcards" srcdoc="{html.escape(page, quote=True)}"></iframe>'


with gr.Blocks(title="HSK-5 中文 Tutor") as demo:
    gr.HTML(HEADER_HTML)
    with gr.Tab("对话 · chat"):
        chat_html = gr.HTML(render_chat([]))
        raw_state = gr.State([])
        targets_state = gr.State(None)
        targets_bar = gr.HTML("")
        with gr.Row(elem_id="mode-row"):
            mode = gr.Radio(
                ["问答 · Q&A", "聊天 · conversation"], value="问答 · Q&A",
                show_label=False, elem_id="mode", container=False,
            )
            review_mode = gr.Checkbox(
                False, label="复习模式 · target only due cards", elem_id="review-mode",
                container=False,
            )
        msg = gr.Textbox(
            placeholder="用中文或英文问我… (ask in Chinese or English)",
            show_label=False, submit_btn=True, elem_id="ask",
        )
        # Mirror of the client-side flashcard deck (word fronts as JSON, due-first).
        # The deck lives in localStorage, so APP_JS keeps this hidden box in sync —
        # it's how the student's own words reach pick_targets() on the server.
        deck_words = gr.Textbox("[]", elem_id="deck-words", elem_classes=["hidden-input"],
                                show_label=False, container=False)
        # Card-example channel: collect writes a request here; the model's fresh
        # example sentence lands in #card-res, which APP_JS's observer picks up.
        card_req = gr.Textbox("", elem_id="card-req", elem_classes=["hidden-input"],
                              show_label=False, container=False)
        card_res = gr.HTML("", elem_id="card-res", elem_classes=["hidden-input"])
        card_req.change(gen_card_example, card_req, card_res)
        # File-backed deck: every deck change is pushed here (debounced) and
        # written to data/deck.json; #deck-file carries the file back on load.
        deck_save = gr.Textbox("", elem_id="deck-save", elem_classes=["hidden-input"],
                               show_label=False, container=False)
        deck_save.change(save_deck, deck_save, None)
        gr.HTML(deck_restore_html, elem_classes=["hidden-input"])
        # Starter chips live in plain HTML; APP_JS fills them with a fresh random
        # sample from STARTER_POOL on every page load and wires the clicks.
        gr.HTML('<div class="starters-row" id="starters"></div>')

        msg.submit(respond, [msg, raw_state, mode, deck_words, targets_state, review_mode],
                   [chat_html, raw_state, msg, targets_state, targets_bar])
        clear = gr.Button("清空 · clear", elem_id="clear-btn")
        clear.click(lambda: (render_chat([]), [], "", None, ""), None,
                    [chat_html, raw_state, msg, targets_state, targets_bar])
    with gr.Tab("卡片 · flashcards"):
        gr.HTML(flashcards_srcdoc())
    with gr.Tab("词表 · word list"):
        # Rendered entirely client-side by APP_JS from the localStorage deck
        # (the deck never touches the server).
        gr.HTML('<div id="wordlist"></div>')

if __name__ == "__main__":
    # Static files for the flashcards' 写 practice mode: the hanzi-writer lib
    # and its per-character stroke data (data/strokes/, via get_strokes.py),
    # served at /gradio_api/file=<path>. Everything stays local/offline.
    gr.set_static_paths(paths=[c.ROOT / "web" / "vendor", c.ROOT / "data" / "strokes"])
    # PORT is set by dev tooling when 7860 is taken; default stays 7860.
    # LAN=1 binds to the local network so a phone/iPad on the same Wi-Fi can be
    # the touchscreen (great for 写 stroke practice; the file-backed deck means
    # a fresh device inherits your cards). Off by default — localhost only.
    # Note: the mic needs a secure context, so voice input stays a Mac feature.
    port = int(os.environ.get("PORT", "7860"))
    lan = os.environ.get("LAN") == "1"
    if lan:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))          # no packets sent — just picks the LAN interface
            ip = s.getsockname()[0]
        except OSError:
            ip = "<your-mac-ip>"
        finally:
            s.close()
        print(f"LAN mode: open http://{ip}:{port} on your phone/iPad (same Wi-Fi)", flush=True)
    # Gradio 6 takes theme/css/js at launch() (not Blocks()).
    demo.launch(
        server_name="0.0.0.0" if lan else None,
        server_port=port,
        theme=THEME,
        css=PAGE_CSS,
        head=HEAD_HTML,
    )
