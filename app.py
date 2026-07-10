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


def respond(user_msg: str, display: list[dict], raw: list[dict]):
    """user_msg + raw history → model reply. Returns (annotated display, raw history, cleared box)."""
    raw = raw + [{"role": "user", "content": user_msg}]
    messages = [{"role": "system", "content": c.SYSTEM_PROMPT}] + raw
    reply = llm.create_chat_completion(messages, temperature=0.7, max_tokens=512)["choices"][0]["message"]["content"]
    raw = raw + [{"role": "assistant", "content": reply}]
    display = display + [
        {"role": "user", "content": annotate(user_msg)},
        {"role": "assistant", "content": annotate(reply)},
    ]
    return display, raw, ""


CSS = """
rt { font-size: .5em; color: #b3302a; }
.hz { border-bottom: 1px dotted #aaa; cursor: help; }
.hz:hover { background: #e7f0ec; }
"""

with gr.Blocks(title="HSK-5 中文 Tutor") as demo:
    gr.Markdown("# HSK-5 中文 Tutor\nBilingual answers · pinyin over every character · **hover a word for its meaning**.")
    # Gradio 6: messages format is the default (no `type=`). sanitize_html=False +
    # allow_tags lets the reading layer's <ruby>/<rt>/<span title> survive.
    chatbot = gr.Chatbot(sanitize_html=False, render_markdown=True, height=460, label="对话")
    raw_state = gr.State([])
    msg = gr.Textbox(placeholder="用中文或英文问我… (ask in Chinese or English)", label="", submit_btn=True)
    gr.Examples(examples=STARTERS, inputs=msg, label="Try one of the six tasks")

    msg.submit(respond, [msg, chatbot, raw_state], [chatbot, raw_state, msg])
    gr.Button("清空 Clear").click(lambda: ([], [], ""), None, [chatbot, raw_state, msg])

if __name__ == "__main__":
    demo.launch(css=CSS)   # Gradio 6: css passes to launch(), not the Blocks constructor
