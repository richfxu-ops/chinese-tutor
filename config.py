"""Single source of truth for the HSK-5 tutor project.

Imported by gen_data.py, train.py, eval.py and app.py so every stage shares the
same model ids, paths, tutor persona and task definitions. Keep configuration
here — the scripts stay thin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"          # adapter + merged model land here
VOCAB_FILE = ROOT / "hsk5_vocab.txt"
GRAMMAR_FILE = ROOT / "hsk5_grammar.txt"

# Serve-time level options. HSK 5 is what the model was fine-tuned on; 4 and 6
# are prompt-steered variants (see system_prompt_app / conversation_system) with
# their own seed lists for starters, targets and reading passages.
HSK_LEVELS = (4, 5, 6)
LEVEL_VOCAB_FILES = {4: ROOT / "hsk4_vocab.txt", 5: VOCAB_FILE, 6: ROOT / "hsk6_vocab.txt"}
LEVEL_GRAMMAR_FILES = {4: ROOT / "hsk4_grammar.txt", 5: GRAMMAR_FILE, 6: ROOT / "hsk6_grammar.txt"}

TRAIN_FILE = DATA_DIR / "train.jsonl"
EVAL_FILE = DATA_DIR / "eval.jsonl"

MERGED_DIR = OUTPUT_DIR / "merged"                       # fp16 base+adapter (merge.py)
GGUF_FILE = OUTPUT_DIR / "hsk5-tutor-q4_k_m.gguf"        # quantized model app.py serves

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"      # student model we fine-tune (7B → 14B:
                                              # QA sweep 2026-07-11 found wrong-rule
                                              # confabulation is capability-bound, and the
                                              # train.jsonl audit found the rules are ~99%
                                              # clean — so a bigger base is the lever. QLoRA
                                              # 4-bit of a 14B wants an A100 on Colab; the
                                              # merged Q4 GGUF is ~9GB, ~2× serve latency.)
TEACHER_MODEL = "claude-sonnet-5"             # generates the synthetic data
HSK_LEVEL = 5

# The tutor persona. Used as the `system` message in EVERY training example.
# At inference app.py serves SYSTEM_PROMPT_APP — this prompt plus two serve-only
# lines (a deliberate, documented divergence; see SYSTEM_PROMPT_APP below and
# DECISIONS.md 2026-07-10).
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

# Served (app.py) variant of the prompt. The training data under-translates framing
# lines — greetings, closing encouragement, usage notes got no English (the per-task
# instructions only demanded translations for the example sentences), and the model
# learned that. This inference-only line nudges it toward fully bilingual output.
# Deliberately NOT used by gen_data.py: training data keeps the original SYSTEM_PROMPT
# so this stays a cheap, reversible experiment (see DECISIONS.md 2026-07-10).
SYSTEM_PROMPT_APP = SYSTEM_PROMPT + (
    "\n- 特别注意：所有中文句子——包括开头的回应、结尾的鼓励和用法说明——"
    "都必须附上对应的英文翻译，不能只翻译例句。"
    "\n- 每次回答的最后，用一个简短的双语问题结束，鼓励学生继续对话或练习。"
)

# HSK level → the parenthetical register label the prompts use.
_LEVEL_LABELS = {4: "中级", 5: "中高级", 6: "高级"}


def _leveled(text: str, level: int) -> str:
    """Swap a prompt's HSK-5 wording for another level. At level 5 both replaces
    are no-ops, so the trained/served level-5 prompts stay byte-identical — the
    default level cannot regress."""
    return (text.replace("HSK 5（中高级）", f"HSK {level}（{_LEVEL_LABELS[level]}）")
                .replace("HSK 5", f"HSK {level}"))


def system_prompt_app(level: int = 5) -> str:
    """The served Q&A prompt at the chosen HSK level (5 = the trained level)."""
    return _leveled(SYSTEM_PROMPT_APP, level)

# Conversation mode (app-only, never used in training). A different framing, not
# a bolted-on rule: the tutor LEADS a chat and corrects the student in passing.
# English translations are deliberately dropped — the app's reading layer
# (pinyin + hover gloss) carries comprehension — except for brief correction
# explanations. app.py appends a per-conversation target-word line (student's
# collected flashcard words + random HSK-5 seeds) so the tutor pushes specific
# vocabulary into use.
CONVERSATION_PROMPT = (
    "你是一位健谈、友好的中文老师，正在和一位 HSK 5 水平的学生用中文聊天，帮助他练习会话。"
    "对话由你来主导和推动——学生不应该需要费力找话题。请遵守以下原则：\n"
    "- 像朋友一样自然、口语化。每次回复 2–4 句话，只用简体中文，语言控制在 HSK 5 或以下。\n"
    "- 你来带话题：主动分享你自己的看法、经历或一个有趣的细节，再向学生提问。"
    "学生回答得短，就追问细节；一个话题聊完了，就自然地引出一个相关的新话题。\n"
    "- 每次回复必须以一个开放式问题结束——避免只能回答“是/不是”的问题，"
    "让学生不得不多说几句。\n"
    "- 积极帮学生练词汇：自然地用上目标生词，并时常直接邀请学生使用某个词，"
    "比如“你能用‘把握’说说你自己的情况吗？”\n"
    "- 学生的中文有错误时，先简短纠正：给出正确说法，用一句简短的英文解释原因，"
    "然后继续话题，不要让纠错打断对话的气氛。\n"
    "- 不要把句子翻译成英文——界面会自动显示拼音和词义。只有纠错的解释可以用英文。\n"
    "- 不要在汉字上标注拼音，保持中文文本干净。只用简体字。"
)


def conversation_system(targets: list[str], level: int = 5) -> str:
    """The exact conversation-mode system prompt served by app.py — and, at
    level 5, used verbatim as the system turn of generated conversation training
    data, so train and serve see the same prompt. Other levels are prompt-steered
    variants of the same text."""
    return _leveled(CONVERSATION_PROMPT, level) + (
        f"\n- 本次对话的目标生词：{'、'.join(targets)}。"
        "每次回复都要做到这两点之一：要么自然地用上一个目标生词，"
        "要么直接请学生用其中一个词回答你的问题（比如“用‘承受’说说你的感受”）。"
        "已经练过的词就换下一个。"
    )


# --------------------------------------------------------------------------- #
# Conversation task (multi-turn — generated separately from TASKS)
# --------------------------------------------------------------------------- #
CONV_N = 100            # conversations to generate (1 teacher call each)
CONV_WORDS_PER = 5      # target words per conversation (mirrors app.py pick_targets)
CONV_MAX_CHARS = 1100   # drop over-long conversations (seq-len safety, see TrainConfig)
CONV_ERROR_FREE_FRAC = 0.25   # fraction of conversations with NO planted mistakes — the
                              # tutor must see correct student turns it does NOT correct
                              # (the over-correction fix, applied to 聊天 too)
CONV_TOPICS = [
    "周末计划", "旅行的经历", "吃饭和做菜", "工作和学习的压力", "爱好和兴趣",
    "看电影和电视剧", "运动和健康", "城市生活", "家人和朋友", "季节和天气",
    "购物的习惯", "学中文的经历",
]

# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
EVAL_FRACTION = 0.10          # held out per task for before/after eval
EXAMPLES_PER_CALL = 5         # ask the teacher for N examples per API call
GEN_TEMPERATURE = 1.0         # some variety (also required=1 when the teacher thinks)
GEN_MAX_TOKENS = 8192         # per call — headroom for thinking + 5 bilingual examples.
                              # Bumped 4096→8192: the new correction modes (esp. ambiguous,
                              # which explains two readings + a clarifying question) produce
                              # longer replies that truncated the JSON array at 4096 and got
                              # dropped (found in the 2026-07-11 smoke test).


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
            "The user submits a Chinese sentence and asks the tutor to check it. "
            "The sentence may contain a real error, be already correct, be "
            "grammatical-but-awkward, or be ambiguous — the case is chosen per example "
            "by CORRECTION_MODES (see gen_data.build_correction_prompt). The assistant "
            "responds appropriately for that case and NEVER invents an error. "
            "Bilingual (Chinese + English), no inline pinyin."
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

# correct_sentence response-type mix (QA sweep + train.jsonl audit, 2026-07-11).
# The tuned 7B invented errors in already-correct sentences ~55% of the time
# because EVERY correction example contained an error to fix — it never learned
# that "leave it alone" is a valid answer. These four modes teach the full response
# space; a solid block of real errors keeps error-detection sharp. gen_data assigns
# one mode per teacher batch to keep each prompt focused; see build_correction_prompt.
CORRECTION_MODES: dict[str, float] = {
    "error": 0.40,      # a real learner error → correct it (rule MUST match the fix)
    "correct": 0.35,    # already correct → CONFIRM it, invent nothing
    "polish": 0.20,     # grammatical but awkward → optional nicer phrasing, NOT a "mistake"
    "ambiguous": 0.05,  # correctness depends on intent → ask, don't over-correct
}

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
    # (compute dtype is picked at runtime in train.py: bf16 on A100/L4, fp16 on T4)
    # SFT
    # 1280 (was 1024) to fit multi-turn conversation examples: ~490-char system
    # prompt + up to CONV_MAX_CHARS of turns ≈ 1.1k tokens with Qwen's tokenizer.
    # Truncation would cut the final assistant turn, which is where the loss is.
    max_seq_len: int = 1280
    epochs: int = 2            # enough for SFT on ~1k examples without overfitting
    lr: float = 2e-4
    # Sized in the 7B/T4 era (batch 4 OOMed — loss logits for a 152k vocab are
    # large on long-sequence batches) and kept for the 14B, which trains on an
    # A100: batch 2 × grad-accum 8 = effective batch 16 works on both.
    # train.py also enables gradient checkpointing.
    per_device_batch_size: int = 2
    grad_accum_steps: int = 8          # effective batch 16
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    logging_steps: int = 2     # log loss often so we can confirm it's learning quickly
    seed: int = 42


TRAIN = TrainConfig()
