"""Single source of truth for the HSK-5 tutor project.

Imported by gen_data.py, train.py, eval.py and app.py so every stage shares the
same model ids, paths, tutor persona and task definitions. Keep configuration
here — the scripts stay thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"          # adapter + merged model land here
VOCAB_FILE = ROOT / "hsk5_vocab.txt"
GRAMMAR_FILE = ROOT / "hsk5_grammar.txt"

TRAIN_FILE = DATA_DIR / "train.jsonl"
EVAL_FILE = DATA_DIR / "eval.jsonl"

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"       # student model we fine-tune
TEACHER_MODEL = "claude-sonnet-5"             # generates the synthetic data
HSK_LEVEL = 5

# The tutor persona. Used as the `system` message in EVERY training example and
# again at inference in app.py — so the model is trained on the exact behaviour
# we ask of it at serve time. Keep them identical.
#
# Note on pinyin: we deliberately DON'T ask the model to emit pinyin. The app's
# reading layer (pypinyin) renders pinyin over every character deterministically,
# so the model's Chinese should stay clean for that annotator. The model's job is
# good bilingual tutoring; the display handles annotation.
SYSTEM_PROMPT = (
    "你是一位耐心、鼓励学生的中文老师，学生的中文水平大约是 HSK 5（中高级）。"
    "请遵守以下原则：\n"
    "- 用简体中文回答，语言控制在 HSK 5 或以下；如果内容本身超出这个水平，"
    "就简化，并温和地提醒学生这是超纲内容。\n"
    "- 完全双语：先给中文，再在后面附上对应的英文翻译，方便学生理解。\n"
    "- 不要在汉字上标注拼音——界面会自动显示拼音，请保持中文文本干净。\n"
    "- 例句和对话要自然、地道，长度适中。\n"
    "- 纠错时，先指出问题，再给出修改后的句子，最后用一句话说明原因。\n"
    "- 只用简体字。语气友好、简洁，多鼓励，不要长篇大论。"
)

# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
EVAL_FRACTION = 0.10          # held out per task for before/after eval
EXAMPLES_PER_CALL = 5         # ask the teacher for N examples per API call
GEN_TEMPERATURE = 1.0         # some variety in the synthetic data
GEN_MAX_TOKENS = 2048         # teacher response cap per call


@dataclass(frozen=True)
class TaskSpec:
    """One tutoring task type the model learns.

    `n`             how many examples to generate for this task
    `needs_vocab`   sample seed word(s) from hsk5_vocab.txt into the prompt
    `needs_grammar` sample a grammar point from hsk5_grammar.txt into the prompt
    `instruction`   what we tell the teacher to produce (the user turn + the
                    ideal assistant turn). Kept as data so gen_data.py stays thin.
    """

    name: str
    n: int
    needs_vocab: bool
    needs_grammar: bool
    instruction: str


# Six task types, ~900 examples total. Counts are deliberately uneven — the
# correction/explanation tasks are the ones a learner leans on most.
TASKS: list[TaskSpec] = [
    TaskSpec(
        name="explain_word",
        n=180,
        needs_vocab=True,
        needs_grammar=False,
        instruction=(
            "The user asks what a given HSK-5 word means. The assistant explains "
            "it: part of speech, a concise definition, and 1–2 natural example "
            "sentences at HSK-5 level, with an English translation of the "
            "explanation and examples. No inline pinyin."
        ),
    ),
    TaskSpec(
        name="use_in_sentence",
        n=150,
        needs_vocab=True,
        needs_grammar=False,
        instruction=(
            "The user asks for the given word to be used in a sentence (or a few). "
            "The assistant gives 1–3 natural HSK-5-level sentences using the word, "
            "each with an English translation. No inline pinyin."
        ),
    ),
    TaskSpec(
        name="correct_sentence",
        n=180,
        needs_vocab=True,
        needs_grammar=False,
        instruction=(
            "The user submits a Chinese sentence that contains a realistic "
            "learner error (wrong word order, measure word, 了/过 aspect, wrong "
            "collocation, etc.) — ideally involving the given word. The assistant "
            "points out the problem, gives the corrected sentence, then briefly "
            "explains why in simple terms, bilingually (Chinese + English). "
            "No inline pinyin."
        ),
    ),
    TaskSpec(
        name="translate",
        n=150,
        needs_vocab=True,
        needs_grammar=False,
        instruction=(
            "The user asks to translate a short English sentence (or a Chinese "
            "one to English) that naturally uses the given word. The assistant "
            "gives the translation and shows both languages, keeping the Chinese "
            "at HSK-5 level. No inline pinyin."
        ),
    ),
    TaskSpec(
        name="example_dialogue",
        n=120,
        needs_vocab=True,
        needs_grammar=False,
        instruction=(
            "The user asks for a short example dialogue on an everyday topic that "
            "uses the given word. The assistant writes a natural 4–8 line A/B "
            "dialogue at HSK-5 level, with an English translation of each line. "
            "No inline pinyin."
        ),
    ),
    TaskSpec(
        name="grammar_point",
        n=120,
        needs_vocab=False,
        needs_grammar=True,
        instruction=(
            "The user asks how to use the given HSK-5 grammar point. The assistant "
            "explains the structure and when to use it, and gives 2–3 example "
            "sentences at HSK-5 level, all with English translations. "
            "No inline pinyin."
        ),
    ),
]

# --------------------------------------------------------------------------- #
# Training (QLoRA — consumed by train.py on Colab)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainConfig:
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Qwen2 attention + MLP projections — the standard LoRA targets.
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # 4-bit base (bitsandbytes, CUDA-only → this runs on Colab)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    # SFT
    max_seq_len: int = 1024
    epochs: int = 3
    lr: float = 2e-4
    # 7B in 4-bit: keep per-device batch small so it fits a 24GB Colab GPU
    # (L4); grad accumulation keeps the effective batch at 16. train.py also
    # enables gradient checkpointing. Bump the batch up on an A100 if you have one.
    per_device_batch_size: int = 4
    grad_accum_steps: int = 4          # effective batch 16
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    logging_steps: int = 10
    seed: int = 42


TRAIN = TrainConfig()
