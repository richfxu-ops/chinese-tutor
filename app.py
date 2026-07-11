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
import re

import gradio as gr
from llama_cpp import Llama

import config as c
from annotate import annotate

if not c.GGUF_FILE.exists():
    raise SystemExit(
        f"Model not found: {c.GGUF_FILE}\n"
        "Build it first: train (Colab) → merge.py → GGUF quantize → put the .gguf in outputs/. "
        "See the README."
    )

# n_gpu_layers=-1 offloads everything to Metal on Apple Silicon. The chat template
# is read from the GGUF metadata (Qwen2.5 ships a ChatML-style template).
llm = Llama(model_path=str(c.GGUF_FILE), n_ctx=4096, n_gpu_layers=-1, verbose=False)

# Starter-prompt pool, ~4 per task type. The page shows a fresh random six on
# every load (sampled client-side in APP_JS, so no server round-trip or reload
# of the Blocks app is needed).
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
.b .who .spk {{ border:none; background:none; cursor:pointer; font-size:.8rem;
               padding:0; margin-left:.5rem; opacity:.45; vertical-align:middle;
               transition:opacity .15s; }}
.b .who .spk:hover {{ opacity:1; }}
.b.t .who {{ color:var(--cinnabar); }}
.b.u .who {{ color:var(--ink-soft); text-align:right; }}
.b .msg {{ font-family:"EB Garamond",var(--hanzi-kai); font-size:1.17rem; line-height:2.55; }}
.b .msg, .b .msg * {{ color:var(--ink) !important; }}
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
                      border:none !important; border-radius:3px !important; }}
#ask .submit-button:hover {{ background:var(--cinnabar-deep) !important; }}
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
    // example: the sentence around the word, from the same bubble
    const msg = hz.closest('.msg');
    const sentence = msg
      ? (plainText(msg).match(/[^。！？!?\\n]*[。！？!?]?/g) || []).find(s => s.includes(front)) : '';
    deck.push({ id: front + ':' + Date.now(), front, pinyin, gloss,
                example: (sentence || '').trim().slice(0, 120),
                ease: 2.5, interval: 0, reps: 0, lapses: 0, due: 0 });
    localStorage.setItem(KEY, JSON.stringify(deck));
    toast('已收藏 “' + front + '” · added to your deck (' + deck.length + ')');
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
    Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value')
      .set.call(ta, chip.textContent);
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    ta.focus();
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
    new MutationObserver(() => {
      ensureStarters();
      const chat = document.querySelector('.chat');
      if (!chat || chat.childElementCount === lastCount) return;
      lastCount = chat.childElementCount;
      jumpToNewest(chat);
    }).observe(document.body, { childList: true, subtree: true });
    ensureStarters();
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', installObserver);
  } else {
    installObserver();
  }
})();
"""

HEAD_HTML = f"<script>{APP_JS.replace('__STARTERS__', json.dumps(STARTER_POOL, ensure_ascii=False))}</script>"

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
_HAS_CJK = re.compile(r"[一-鿿]")
_CJK_CHUNK = re.compile(r"[一-鿿]+[0-9，。！？、；：]*")


def chinese_only(text: str) -> str:
    lines = (line for line in text.split("\n") if _HAS_CJK.search(line))
    return " ".join("".join(_CJK_CHUNK.findall(line)) for line in lines)


def render_chat(raw: list[dict]) -> str:
    """Build the whole transcript as one self-contained HTML block, annotating every
    Chinese span (pinyin ruby + hover gloss)."""
    bubbles = []
    for m in raw:
        u = m["role"] == "user"
        who = "You" if u else "老师 Tutor"
        # annotate() emits native `title` tooltips (right for the standalone docs/
        # pages); in-app we show the gloss via the CSS tooltip instead, so rename
        # the attribute. Safe: message text is html-escaped, so ` title="` can
        # only come from annotate()'s own .hz spans.
        msg = annotate(m["content"]).replace(' title="', ' data-tip="')
        spk = "" if u else (
            f'<button class="spk" title="朗读中文 · read the Chinese aloud"'
            f' data-speak="{html.escape(chinese_only(m["content"]), quote=True)}">🔊</button>'
        )
        bubbles.append(
            f'<div class="b {"u" if u else "t"}"><div class="who">{who}{spk}</div>'
            f'<div class="msg">{msg}</div></div>'
        )
    inner = "".join(bubbles) or '<div class="empty">用中文或英文问我… (ask me anything)</div>'
    return f'<div class="chat">{inner}</div>'


def respond(user_msg: str, raw: list[dict]):
    """user message + raw history → model reply. Returns (chat HTML, raw history, cleared box)."""
    if not user_msg.strip():
        return render_chat(raw), raw, ""
    raw = raw + [{"role": "user", "content": user_msg}]
    messages = [{"role": "system", "content": c.SYSTEM_PROMPT_APP}] + raw
    reply = llm.create_chat_completion(messages, temperature=0.7, max_tokens=512)["choices"][0]["message"]["content"]
    raw = raw + [{"role": "assistant", "content": reply}]
    return render_chat(raw), raw, ""


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
        msg = gr.Textbox(
            placeholder="用中文或英文问我… (ask in Chinese or English)",
            show_label=False, submit_btn=True, elem_id="ask",
        )
        # Starter chips live in plain HTML; APP_JS fills them with a fresh random
        # sample from STARTER_POOL on every page load and wires the clicks.
        gr.HTML('<div class="starters-row" id="starters"></div>')

        msg.submit(respond, [msg, raw_state], [chat_html, raw_state, msg])
        clear = gr.Button("清空 · clear", elem_id="clear-btn")
        clear.click(lambda: (render_chat([]), [], ""), None, [chat_html, raw_state, msg])
    with gr.Tab("卡片 · flashcards"):
        gr.HTML(flashcards_srcdoc())

if __name__ == "__main__":
    # PORT is set by dev tooling when 7860 is taken; default stays 7860.
    # Gradio 6 takes theme/css/js at launch() (not Blocks()).
    demo.launch(
        server_port=int(os.environ.get("PORT", "7860")),
        theme=THEME,
        css=PAGE_CSS,
        head=HEAD_HTML,
    )
