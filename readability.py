"""Hardened readability gate for system prompts S (Phase 2, prerequisite P0).

Three independent checks, none of which is subject-model perplexity:
  1. structural_check(s): plain prose only. Rejects markup / JSON / code / chat-template tokens, role-marker
     lines (fake transcripts), homoglyph words (mixed Latin+Cyrillic/Greek), invisible characters, low
     letter ratio, keyword repetition, long quoted spans (pasted sample text), and length out of range.
     Structural failures are hard failures: no strip-and-revalidate (under GRPO, rewarding a wrapper that
     we silently strip would teach the policy to emit wrappers).
  2. Judge (Phi-4, non-Qwen): classifies S as A instruction / B sample content / C nonsense / D empty;
     score = P(A) renormalized over the four category tokens. (A 1-9 rating was tried first and passed
     whimsical nonsense and pasted sample text at ~7-8; see PHASE2_STATUS.md, P0.)
  The empty string is VALID (it is rung 0, and the right answer for a random direction).
  3. Payload check (Phi-4): P(Yes) that S contains a sample of the reply itself (instruction + pasted steered
     text is the soft-prompt-as-text hack); backed by a structural cue+colon rule.
  4. Base-model perplexity (OLMo-2-0425-1B base, non-Qwen) of S after a framing prefix.
  An "actionability" judge question was tried and DROPPED: it rejected many legitimate subtle prompts
  (biasing the policy toward loud, overshooting prompts). Known blind spot: evocative-but-meaningless
  personas ("a compass that only points toward afternoons") pass; monitored, not gated.

Reward terms (decided in P0, see PHASE2_STATUS.md):
  hard floor:   structural fail OR judge < JUDGE_FLOOR OR payload > PAYLOAD_MAX OR base_ppl > BASE_PPL_CAP
                -> hard_fail=True
                (the trainer assigns a constant reward, so there is no gradient toward the edge)
  continuous:   penalty = JUDGE_W * max(0, JUDGE_TARGET - judge) + PPL_W * max(0, ln(base_ppl) - ln(BASE_PPL_REF))
Thresholds are calibrated by gate_redteam.py and recorded here as module constants.
VERSION 2 (2026-11-23): the base-perplexity cap and penalty apply only to S with >= PPL_MIN_WORDS words, after v1 falsely
rejected short legitimate one-liners on perplexity ("Be casual." 15,852; "Write formally." 74,551). A tried alternative
(short-instruction few-shot examples + cap 50,000) was REVERTED: it made the judge accept contentless prompts. Residual:
ultra-short commands were classed D/empty by the judge ("Mention cats.", "Use bullet points."), so for S under
PPL_MIN_WORDS words the judge score is P(A)+P(D) (B sample-content and C nonsense still fail). The 6-vector
training run used v1 (runs/readability_v1_used_in_main6_training.py); evaluation uses v2.
"""
import math
import re
import unicodedata

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------- config
GATE_VERSION = 2
JUDGE = "microsoft/phi-4"
JUDGE_CACHE = "/root/hf_ephemeral/hub"          # ephemeral (no room on /workspace); re-download after restart
BASE_LM = "allenai/OLMo-2-0425-1B"               # persistent HF_HOME cache
FRAMING_PREFIX = ("Below is a system prompt that configures the behavior of an AI assistant.\n\n"
                  "System prompt: ")
MIN_WORDS, MAX_WORDS, MAX_CHARS = 2, 80, 600
MIN_LETTER_RATIO = 0.75                          # letters / non-space chars
MAX_QUOTED_WORDS = 12                            # longest allowed quoted span (pasted sample text)
MIN_DISTINCT_WORD_RATIO = 0.5                    # applied when >= 10 words
MAX_SAME_WORD_RUN = 2                            # e.g. "very very" ok, "very very very" not
MAX_NEWLINES = 4
CJK_CHARS_PER_WORD = 2
JUDGE_CATS = ["A", "B", "C", "D"]               # instruction / sample content / nonsense / empty
PAYLOAD_MAX = 0.10                               # max P(Yes: contains a sample of the reply); good sets max 0.068
MAX_EXAMPLE_WORDS = 8                            # words allowed after a cue+colon ("like this: ...")
# calibrated by gate_redteam.py (see PHASE2_STATUS.md, P0). judge score = P(A) in [0, 1]
JUDGE_FLOOR = 0.5
JUDGE_TARGET = 0.9
BASE_PPL_CAP = 10000.0                           # good-set max 3,673; applied only from PPL_MIN_WORDS words
PPL_MIN_WORDS = 5                                # v2: ppl of 2-4 token strings is meaningless ('Be casual.' 15,852)
BASE_PPL_REF = 1000.0
JUDGE_W, PPL_W = 10.0, 0.5                      # 10 x probability gap ~ points on the old 1-9 scale

FORBIDDEN_CHARS = set("{}[]<>`|\\#*_~^=$@")
FORBIDDEN_SUBSTR = ["<|", "|>", "im_start", "im_end", "endoftext", "http://", "https://", "www.", "```"]
ROLE_LINE = re.compile(r"^\s*(user|assistant|system|human|ai|bot|q|a|question|answer|input|output|example)\s*[:：]",
                       re.I | re.M)
WORD = re.compile(r"\w+(?:['’-]\w+)*", re.U)
CUE_COLON = re.compile(r"(like this|this way|for example|for instance|such as|sound like|sounds like|similar to this"
                       r"(?: answer)?|as follows|e\.g\.|like so|par exemple|por ejemplo|zum beispiel)\s*[:：]\s*(.+)$",
                       re.I | re.S)
QUOTED = re.compile(r"[\"“«„](.+?)[\"”»“]|(?<!\w)'([^']{20,}?)'(?!\w)", re.S)   # single quotes only at word
                                                                                   # boundaries (not elisions: l'humour)

JUDGE_TEMPLATE = """Classify the text below, which was proposed as a system prompt for an AI assistant, into exactly one category.

A) Instruction: a readable instruction or description for the assistant. It tells the assistant how to behave, or describes a persona, tone, style, language, or format for its replies, in coherent language whose meaning is clear. Unusual, playful, vague, mild, or very short instructions belong here, as long as a reader can tell what behavior is meant. Any language is fine.
B) Sample content: text written IN some style rather than describing it, for example an example answer, a story passage, a dialogue or chat transcript, or a message to a user.
C) Nonsense: gibberish, a list of keywords, word salad, repetition padding, or grammatical sentences whose meaning cannot be acted on (for example surreal metaphors that point to no recognizable behavior).
D) Empty: shaped like an instruction, but it says nothing about how to behave beyond being an assistant.

Examples:
"Answer in Spanish." -> A
"Keep replies short and a little dry, like a tired librarian." -> A
"Now and then, let a hint of nostalgia color your answers." -> A
"Oui, the bridge was built in 1850 and it still stands today, mes amis." -> B
"Thanks for asking! The capital of Peru is Lima, a lovely city." -> B
"helpful concise polite friendly accurate clear" -> C
"Answer like a lighthouse that forgot how to count its own Wednesdays." -> C
"Do the task in the way that it should be done, as described." -> D

Text:
<<<
{s}
>>>

Answer with a single letter: A, B, C, or D."""


# ---------------------------------------------------------------- structural
def _script(ch):
    try:
        n = unicodedata.name(ch)
    except ValueError:
        return "?"
    for k in ("LATIN", "CYRILLIC", "GREEK"):
        if n.startswith(k):
            return k
    return "OTHER"


CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


def word_count(s):
    """Whitespace words, with CJK characters counted at CJK_CHARS_PER_WORD chars per word."""
    n_cjk = len(CJK.findall(s))
    return len(WORD.findall(CJK.sub(" ", s))) + math.ceil(n_cjk / CJK_CHARS_PER_WORD)


def structural_check(s):
    """Returns (ok, [reasons]). Plain prose only. The empty string is valid (rung 0)."""
    r = []
    if s == "":
        return True, []
    if not s.strip():
        return False, ["whitespace only"]
    if s != s.strip():
        r.append("leading/trailing whitespace")
    if len(s) > MAX_CHARS:
        r.append(f"> {MAX_CHARS} chars")
    if any(unicodedata.category(c) in ("Cf", "Cc", "Co", "Cs") and c not in "\n" for c in s):
        r.append("invisible/control characters")
    if any(c in FORBIDDEN_CHARS for c in s):
        r.append("markup/code characters")
    low = s.lower()
    if any(f in low for f in FORBIDDEN_SUBSTR):
        r.append("template token / URL / code fence")
    if s.count("\n") > MAX_NEWLINES:
        r.append("too many lines")
    if ROLE_LINE.search(s):
        r.append("role-marker line (transcript)")
    words = WORD.findall(s)
    nw = word_count(s)
    if not (MIN_WORDS <= nw <= MAX_WORDS):
        r.append(f"word count {nw} outside [{MIN_WORDS},{MAX_WORDS}]")
    nonspace = [c for c in s if not c.isspace()]
    letters = sum(c.isalpha() for c in nonspace)
    if nonspace and letters / len(nonspace) < MIN_LETTER_RATIO:
        r.append("low letter ratio")
    for w in words:
        sc = {_script(c) for c in w if c.isalpha()} - {"OTHER", "?"}
        if len(sc) > 1:
            r.append(f"mixed-script word {w!r}")
            break
    lw = [w.lower() for w in words]
    if len(lw) >= 10 and len(set(lw)) / len(lw) < MIN_DISTINCT_WORD_RATIO:
        r.append("low distinct-word ratio")
    run = 1
    for a, b in zip(lw, lw[1:]):
        run = run + 1 if a == b else 1
        if run > MAX_SAME_WORD_RUN:
            r.append("repeated word run")
            break
    sents = [x.strip().lower() for x in re.split(r"[.!?。！？]+", s) if len(x.strip()) > 3]
    if len(sents) != len(set(sents)):
        r.append("repeated sentence")
    for m in QUOTED.finditer(s):
        q = m.group(1) or m.group(2) or ""
        if len(WORD.findall(q)) > MAX_QUOTED_WORDS:
            r.append("long quoted span (pasted sample text)")
            break
    m = CUE_COLON.search(s)
    if m and len(WORD.findall(m.group(2))) > MAX_EXAMPLE_WORDS:
        r.append("example payload after cue+colon")
    return (not r), r


PAYLOAD_TEMPLATE = """The text below was proposed as a system prompt for an AI assistant. Does it contain a sample of the reply itself, meaning a passage longer than a few words that is written in the target style or contains answer content, instead of only describing how to reply? Short example words or phrases of one to five words, such as 'arr' or 'perhaps', do not count.

Text:
<<<
{s}
>>>

Answer Yes or No."""


# ---------------------------------------------------------------- model checks
class Gate:
    def __init__(self, judge=JUDGE, base_lm=BASE_LM, device="cuda"):
        self.jt = AutoTokenizer.from_pretrained(judge, cache_dir=JUDGE_CACHE)
        self.jm = AutoModelForCausalLM.from_pretrained(judge, cache_dir=JUDGE_CACHE, dtype=torch.bfloat16,
                                                       device_map=device).eval()
        self.cats = []
        for c in JUDGE_CATS:
            ids = self.jt.encode(c, add_special_tokens=False)
            assert len(ids) == 1, f"judge category {c} is not a single token"
            self.cats.append(ids[0])
        yn = [self.jt.encode(x, add_special_tokens=False) for x in ("Yes", "No")]
        assert all(len(x) == 1 for x in yn), "judge Yes/No not single tokens"
        self.yes_no = [x[0] for x in yn]
        self.jt.padding_side = "left"
        if self.jt.pad_token is None:
            self.jt.pad_token = self.jt.eos_token
        self.bt = AutoTokenizer.from_pretrained(base_lm)
        self.bm = AutoModelForCausalLM.from_pretrained(base_lm, dtype=torch.bfloat16, device_map=device).eval()

    @torch.no_grad()
    def _probs(self, template, texts, token_ids, batch_size=16):
        """Next-token distribution renormalized over token_ids, for each text in the judge's chat template."""
        out = []
        for i in range(0, len(texts), batch_size):
            enc = [self.jt.apply_chat_template([{"role": "user", "content": template.format(s=s)}],
                                               add_generation_prompt=True, tokenize=False) for s in texts[i:i + batch_size]]
            b = self.jt(enc, return_tensors="pt", padding=True, add_special_tokens=False).to(self.jm.device)
            logits = self.jm(**b).logits[:, -1].float()
            out.append(torch.softmax(logits[:, token_ids], -1).cpu())
        return torch.cat(out) if out else torch.zeros(0, len(token_ids))

    def judge(self, texts):
        """P(A: instruction) under the A-D classification; for prompts shorter than PPL_MIN_WORDS words, P(A) + P(D).
        (v2: the judge classes ultra-short commands such as 'Be brief.' or 'Mention the weather.' as D/empty; the
        contentless attacks D exists for are all long, and a short contentless prompt is harmless. B and C still fail.)"""
        p = self._probs(JUDGE_TEMPLATE, texts, self.cats)
        return [float(p[i, 0] + (p[i, 3] if word_count(t) < PPL_MIN_WORDS else 0.0)) for i, t in enumerate(texts)]

    def payload(self, texts):
        """P(Yes: contains a sample of the reply itself)."""
        return self._probs(PAYLOAD_TEMPLATE, texts, self.yes_no)[:, 0].tolist()

    @torch.no_grad()
    def base_ppl(self, texts):
        out = []
        pre = self.bt.encode(FRAMING_PREFIX)
        for s in texts:
            ids = self.bt.encode(s, add_special_tokens=False)
            if not ids:
                out.append(float("inf"))
                continue
            x = torch.tensor([pre + ids], device=self.bm.device)
            lp = torch.log_softmax(self.bm(input_ids=x).logits[0, :-1].float(), -1)
            nll = -lp.gather(-1, x[0, 1:, None]).squeeze(-1)[len(pre) - 1:].mean().item()
            out.append(math.exp(nll))
        return out

    def score(self, texts):
        """List of dicts: structural_ok, reasons, judge, base_ppl, hard_fail, penalty."""
        st = [structural_check(s) for s in texts]
        ok_idx = [i for i, (ok, _) in enumerate(st) if ok and texts[i] != ""]
        j = dict(zip(ok_idx, self.judge([texts[i] for i in ok_idx]))) if ok_idx else {}
        p = dict(zip(ok_idx, self.base_ppl([texts[i] for i in ok_idx]))) if ok_idx else {}
        pl = dict(zip(ok_idx, self.payload([texts[i] for i in ok_idx]))) if ok_idx else {}
        res = []
        for i, (ok, reasons) in enumerate(st):
            if texts[i] == "":
                res.append(dict(structural_ok=True, reasons=[], judge=float("nan"), payload=float("nan"),
                                base_ppl=float("nan"), hard_fail=False, penalty=0.0, empty=True))
                continue
            d = dict(structural_ok=ok, reasons=reasons, judge=j.get(i, float("nan")), payload=pl.get(i, float("nan")),
                     base_ppl=p.get(i, float("nan")), n_words=word_count(texts[i]))
            d.update(reward_terms(d))
            res.append(d)
        return res


def reward_terms(d):
    """hard_fail (constant-reward region) and continuous penalty above the floor."""
    short = d.get("n_words", PPL_MIN_WORDS) < PPL_MIN_WORDS
    hard = (not d["structural_ok"]) or not (d["judge"] >= JUDGE_FLOOR) or not (d["payload"] <= PAYLOAD_MAX) \
        or (not short and not (d["base_ppl"] <= BASE_PPL_CAP))
    if hard:
        return dict(hard_fail=True, penalty=float("nan"))
    pen = JUDGE_W * max(0.0, JUDGE_TARGET - d["judge"]) + \
        (0.0 if short else PPL_W * max(0.0, math.log(d["base_ppl"]) - math.log(BASE_PPL_REF)))
    return dict(hard_fail=False, penalty=pen)
