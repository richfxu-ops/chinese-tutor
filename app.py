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

import os

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

STARTERS = [
    "“毕竟”是什么意思？",
    "用“承担”造两个句子",
    "帮我改这个句子：我昨天去过北京。",
    "How do I say “I can’t help but worry” in Chinese?",
    "给我一段在餐厅点菜的对话",
    "怎么用“既然……就……”这个语法？",
]


# The transcript is rendered as raw HTML via gr.HTML — NOT gr.Chatbot. Gradio 6's
# Chatbot markdown pass sanitizes messages and strips the `title` attribute (and
# non-standard tags), which kills the hover gloss. gr.HTML renders our reading-layer
# markup untouched (verified: <style>, ruby/rt and attributes all survive).
#
# Two Gradio-specific quirks this CSS works around:
#  - Gradio ships `.gradio-container-* .prose * { color: var(--body-text-color) }`,
#    a universal selector that recolors every element and beats inherited colors —
#    in dark mode that's near-white text on our light bubbles. The `!important`
#    on the .msg color rules is what keeps the transcript readable on both themes.
#  - Native `title` tooltips are slow (~1s delay) and unreliable inside embedded
#    webviews, so the gloss is shown with a styled CSS tooltip instead: render_chat
#    rewrites annotate()'s title= to data-tip= and .hz:hover::after displays it.
CHAT_CSS = """<style>
.chat { display:flex; flex-direction:column; gap:.6rem; padding:.4rem; }
.b { max-width:88%; border:1px solid #e4ded3; border-radius:12px; padding:.6rem .85rem; }
.b.u { align-self:flex-end; background:#f6e9e6; border-top-right-radius:3px; }
.b.t { align-self:flex-start; background:#fbfaf7; border-top-left-radius:3px; }
.b .who { font-family:ui-monospace,Menlo,monospace; font-size:.6rem; text-transform:uppercase; letter-spacing:.07em; margin-bottom:.35rem; }
.b.u .who { color:#b3302a; text-align:right; }
.b.t .who { color:#2f6e5d; }
.b .msg { font-size:1.06rem; line-height:2.5; }
.b .msg, .b .msg * { color:#211c19 !important; }
ruby { ruby-position:over; margin:0 .02em; }
rt { font-family:ui-monospace,Menlo,monospace; font-size:.45em; font-weight:500; }
.b .msg rt { color:#b3302a !important; }
.hz { position:relative; border-bottom:1px dotted #9a9086; cursor:help; border-radius:2px; }
.hz:hover { background:#e7f0ec; z-index:5; }
.hz:hover::after {
  content:attr(data-tip);
  position:absolute; top:calc(100% + 4px); left:50%; transform:translateX(-50%);
  width:max-content; max-width:min(320px, 70vw); white-space:normal;
  background:#211c19; color:#fffdf8; font-size:.78rem; line-height:1.5; font-weight:400;
  padding:.4rem .6rem; border-radius:6px; box-shadow:0 2px 10px rgba(0,0,0,.25);
  pointer-events:none; z-index:20;
}
.empty { color:#9a9086; text-align:center; padding:2.5rem 1rem; }
</style>"""


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
        bubbles.append(
            f'<div class="b {"u" if u else "t"}"><div class="who">{who}</div>'
            f'<div class="msg">{msg}</div></div>'
        )
    inner = "".join(bubbles) or '<div class="empty">用中文或英文问我… (ask me anything)</div>'
    return CHAT_CSS + f'<div class="chat">{inner}</div>'


def respond(user_msg: str, raw: list[dict]):
    """user message + raw history → model reply. Returns (chat HTML, raw history, cleared box)."""
    if not user_msg.strip():
        return render_chat(raw), raw, ""
    raw = raw + [{"role": "user", "content": user_msg}]
    messages = [{"role": "system", "content": c.SYSTEM_PROMPT_APP}] + raw
    reply = llm.create_chat_completion(messages, temperature=0.7, max_tokens=512)["choices"][0]["message"]["content"]
    raw = raw + [{"role": "assistant", "content": reply}]
    return render_chat(raw), raw, ""


with gr.Blocks(title="HSK-5 中文 Tutor") as demo:
    gr.Markdown("# HSK-5 中文 Tutor\nBilingual answers · pinyin over every character · **hover a word for its meaning**.")
    chat_html = gr.HTML(render_chat([]))
    raw_state = gr.State([])
    msg = gr.Textbox(placeholder="用中文或英文问我… (ask in Chinese or English)", label="", submit_btn=True)
    gr.Examples(examples=STARTERS, inputs=msg, label="Try one of the six tasks")

    msg.submit(respond, [msg, raw_state], [chat_html, raw_state, msg])
    gr.Button("清空 Clear").click(lambda: (render_chat([]), [], ""), None, [chat_html, raw_state, msg])

if __name__ == "__main__":
    # PORT is set by dev tooling when 7860 is taken; default stays 7860.
    demo.launch(server_port=int(os.environ.get("PORT", "7860")))
