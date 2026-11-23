"""Phase 1 behavior definitions: contrast elicitation, the candidate-prompt ladder, and a cheap
behavior metric per vector. Single source of truth for build_vectors.py and oracle_search.py.

Circularity note: the positive contrast responses are produced by prompting the model with ELICIT.
The vector itself is read only off the RESPONSE tokens, under the neutral (empty) system prompt,
so it encodes "this text has property X," not the instruction tokens. ELICIT is deliberately worded
differently from the rung-2 one-liner, but rung 2 still has a home-field advantage because the
steered behavior was distilled from prompted behavior. A behavior-matching prompt doing well is
therefore partly expected. State this in any writeup.
"""
import re

_WORD = re.compile(r"[A-Za-zÀ-ÿ']+")


def _words(t):
    return [w.lower() for w in _WORD.findall(t)]


# ---------------------------------------------------------------- metrics (higher = more behavior)
FR = set("le la les des une est et que qui dans pour pas vous nous avec sur ce cette sont être "
         "il elle ils du au aux mais ou donc très plus votre vos peut faire comme".split())
EN = set("the and is are to of that for with you this be it on as not your can have will".split())


def french_score(t):
    """1 if French stopwords outnumber English stopwords, else 0."""
    w = _words(t)
    return float(sum(x in FR for x in w) > sum(x in EN for x in w))


PIRATE = set("arr arrr ahoy matey mateys ye yer aye avast hearty hearties landlubber landlubbers "
             "scallywag scallywags booty plunder treasure ship ships sea seas captain cap'n "
             "sail sails savvy shiver timbers me'hearty yo ho grog buccaneer seven doubloons "
             "be'neath o' fer".split())


def pirate_score(t):
    """Pirate-lexicon hits per 100 words."""
    w = _words(t)
    return 100 * sum(x in PIRATE for x in w) / max(1, len(w))


HEDGE_1 = set("might may perhaps possibly probably likely unlikely uncertain uncertainty maybe "
              "seems appears suggests generally typically often usually approximately roughly "
              "could depend depends depending unclear arguably potentially".split())
HEDGE_2 = ["not sure", "i think", "i believe", "it is possible", "it's possible", "to my knowledge",
           "i may be wrong", "i could be wrong", "not certain", "hard to say", "in some cases"]


def hedge_score(t):
    """Hedge expressions per 100 words."""
    low = t.lower()
    w = _words(t)
    n = sum(x in HEDGE_1 for x in w) + sum(low.count(p) for p in HEDGE_2)
    return 100 * n / max(1, len(w))


# ---------------------------------------------------------------- behaviors
BEHAVIORS = {
    "french": dict(
        elicit="Tu es un assistant francophone. Quelle que soit la langue de l'utilisateur, "
               "rédige toute ta réponse en français.",
        metric=french_score, metric_name="frac_french",
        rung2="Respond in French.",
        rung3=[
            "You are a helpful assistant. Always write your entire response in French, regardless "
            "of the language the user writes in. Keep code, names and technical terms unchanged, "
            "but write all explanations in fluent, natural French.",
            "Réponds toujours en français, quelle que soit la langue de la question. Sois utile, "
            "clair et précis.",
        ],
    ),
    "pirate": dict(
        elicit="Talk like an old sea pirate in every reply: use pirate slang and expressions such "
               "as 'arr', 'matey', 'ye' and nautical metaphors, while still answering the question.",
        metric=pirate_score, metric_name="pirate_per100w",
        rung2="Talk like a pirate.",
        rung3=[
            "You are a helpful assistant who speaks like a swashbuckling pirate captain. Sprinkle "
            "your answers with pirate slang ('arr', 'ahoy', 'matey', 'ye', 'aye') and seafaring "
            "metaphors, but still answer the user's question accurately and helpfully.",
            "Ye be a pirate, matey! Answer every question as a salty sea dog would, with plenty "
            "o' pirate talk, while still givin' useful answers.",
        ],
    ),
    "hedging": dict(
        elicit="Be epistemically cautious in every reply. Qualify your claims, acknowledge "
               "uncertainty and limitations, note what you might be wrong about, and prefer words "
               "like 'might', 'possibly', 'it seems' over confident assertions.",
        metric=hedge_score, metric_name="hedges_per100w",
        rung2="Express uncertainty and hedge your claims.",
        rung3=[
            "You are a careful, humble assistant. You are frequently unsure, so qualify most "
            "statements with words like 'might', 'perhaps', 'possibly' or 'it seems', point out "
            "caveats and limitations, and avoid sounding overconfident, while still being helpful.",
            "Answer helpfully but tentatively. Acknowledge that you may be mistaken, note where "
            "things depend on context, and present claims as likely rather than certain.",
        ],
    ),
}

RUNG1 = "You are a helpful assistant."


# ================================================================ Phase 2 candidate behaviors
# Same structure as BEHAVIORS. Kept in a separate dict so Phase 1 scripts that iterate BEHAVIORS
# (e.g. gate_redteam's good_dev set) are unchanged. ELICIT wording deliberately differs from rung2.
DE = set("der die das und ist nicht ein eine zu mit für auf sie es ich wir sich auch den dem von werden "
         "kann sind wie oder aber wenn noch nur bei diese".split())
ES = set("el la los las y es que de en un una por para con no se su lo como más pero sus le ya o "
         "este sí porque esta entre cuando muy sin sobre también".split())


def _lang_score(stop):
    def f(t):
        w = _words(t)
        return float(sum(x in stop for x in w) > sum(x in EN for x in w))
    return f


def _per100(lex, phrases=()):
    def f(t):
        w = _words(t)
        low = t.lower()
        return 100 * (sum(x in lex for x in w) + sum(low.count(p) for p in phrases)) / max(1, len(w))
    return f


FORMAL = set("furthermore moreover therefore thus hence consequently regarding additionally nevertheless "
             "accordingly whereas herein subsequently notwithstanding pertaining utilize facilitate".split())
CASUAL = set("gonna wanna kinda sorta yeah yep nope hey dude lol cool awesome stuff totally super pretty "
             "basically gotta ya y'know okay ok tbh omg chill".split())
ENTHUS = set("amazing awesome fantastic wonderful exciting excited love incredible great thrilled "
             "brilliant delightful absolutely wow".split())
GLOOM = set("unfortunately sadly bleak grim futile hopeless dreary inevitably regrettably dismal "
            "pointless despair gloomy alas bitter".split())
ARCHAIC = set("thee thou thy thine hath doth art ere verily prithee forsooth nay 'tis wherefore "
              "hither thither whence anon methinks".split())
COWBOY = set("howdy partner reckon y'all yonder ain't fixin darn mighty ranch cowpoke varmint "
             "sheriff saddle buckaroo critter".split())
CAT = set("cat cats kitten kittens feline felines meow purr purring whiskers paws kitty".split())
FOOD = set("recipe cooking cook ingredient ingredients flavor flavour bake baking simmer dish "
           "kitchen spice seasoning oven taste".split())
EMPATHY_P = ["i understand", "i'm sorry", "i am sorry", "that sounds", "it's okay", "it is okay",
             "you're not alone", "that must", "i hear you", "it's completely", "feel free", "take care"]


def emoji_score(t):
    w = max(1, len(_words(t)))
    return 100 * sum(1 for c in t if ord(c) >= 0x1F300 or 0x2600 <= ord(c) <= 0x27BF) / w


def bullet_score(t):
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    return sum(bool(re.match(r"^([-*•]|\d+[.)])\s", l)) for l in lines) / max(1, len(lines))


def lowercase_score(t):
    letters = [c for c in t if c.isalpha()]
    return sum(c.islower() for c in letters) / max(1, len(letters))


def brevity_score(t):
    """Higher = shorter (negative word count, so 'higher = more behavior' holds)."""
    return -float(len(_words(t)))


BEHAVIORS_P2 = {
    "german": dict(
        elicit="Du bist ein deutschsprachiger Assistent. Schreibe deine gesamte Antwort auf Deutsch, egal in "
               "welcher Sprache die Frage gestellt wird.",
        metric=_lang_score(DE), metric_name="frac_german",
        rung2="Respond in German.",
        rung3=["Always answer in natural, fluent German, whatever language the user writes in; keep code and "
               "proper names unchanged."]),
    "spanish": dict(
        elicit="Eres un asistente hispanohablante. Escribe toda tu respuesta en español, sin importar el idioma "
               "de la pregunta.",
        metric=_lang_score(ES), metric_name="frac_spanish",
        rung2="Respond in Spanish.",
        rung3=["Always answer in natural, fluent Spanish, whatever language the user writes in; keep code and "
               "proper names unchanged."]),
    "formal": dict(
        elicit="Adopt a highly formal, academic register in all replies: no contractions, no colloquialisms, "
               "precise vocabulary, and the measured tone of a scholarly publication.",
        metric=_per100(FORMAL), metric_name="formal_per100w",
        rung2="Write formally.",
        rung3=["Use a formal, professional register: avoid contractions and slang, and prefer precise, "
               "elevated vocabulary."]),
    "casual": dict(
        elicit="Talk super casually in every reply, like you're texting a close friend: relaxed slang, "
               "contractions, and a laid-back vibe.",
        metric=_per100(CASUAL), metric_name="casual_per100w",
        rung2="Be casual.",
        rung3=["Keep it relaxed and informal, like chatting with a friend; slang and contractions are fine."]),
    "enthusiastic": dict(
        elicit="Bring infectious excitement to every answer: be upbeat, energetic, and openly delighted by "
               "the user's question.",
        metric=lambda t: _per100(ENTHUS)(t) + 100 * t.count("!") / max(1, len(_words(t))),
        metric_name="enthus_per100w",
        rung2="Be enthusiastic.",
        rung3=["Sound genuinely excited and upbeat about whatever the user asks, with plenty of energy."]),
    "gloomy": dict(
        elicit="Let a melancholy, pessimistic outlook color every reply: dwell on difficulties, downsides, and "
               "the futility of things, while still answering.",
        metric=_per100(GLOOM), metric_name="gloom_per100w",
        rung2="Be pessimistic.",
        rung3=["Answer with a gloomy, downbeat outlook that emphasizes problems and drawbacks, but still "
               "answer the question."]),
    "concise": dict(
        elicit="Give the shortest possible complete answer. Strip every unnecessary word; one or two "
               "sentences whenever you can.",
        metric=brevity_score, metric_name="neg_words",
        rung2="Be concise.",
        rung3=["Keep answers as short as possible while still answering; skip preamble and elaboration."]),
    "bullets": dict(
        elicit="Format every reply as a list: break the content into bullet points or numbered items rather "
               "than paragraphs.",
        metric=bullet_score, metric_name="frac_list_lines",
        rung2="Use bullet points.",
        rung3=["Structure your answers as bulleted or numbered lists instead of prose paragraphs."]),
    "emoji": dict(
        elicit="Sprinkle lots of emoji throughout every reply, several per paragraph, to make it lively and "
               "expressive.",
        metric=emoji_score, metric_name="emoji_per100w",
        rung2="Use lots of emoji.",
        rung3=["Add plenty of emoji to your answers to keep the tone fun and expressive."]),
    "lowercase": dict(
        elicit="type everything in lowercase only, never using capital letters, not even at the start of "
               "sentences or for names.",
        metric=lowercase_score, metric_name="frac_lowercase",
        rung2="Write in all lowercase.",
        rung3=["Never use capital letters; write every word, including names and sentence starts, in lowercase."]),
    "archaic": dict(
        elicit="Speak in the manner of an Elizabethan playwright: use thee, thou, thy, hath, doth and other "
               "archaic forms and flourishes in every reply.",
        metric=_per100(ARCHAIC), metric_name="archaic_per100w",
        rung2="Talk like Shakespeare.",
        rung3=["Answer in early-modern English, in the style of Shakespeare's plays, while still being helpful."]),
    "cowboy": dict(
        elicit="Answer as an old-time Texas cowboy would: folksy drawl, ranch slang, and plenty of 'reckon', "
               "'partner' and 'y'all'.",
        metric=_per100(COWBOY), metric_name="cowboy_per100w",
        rung2="Talk like a cowboy.",
        rung3=["You are a friendly cowboy from the Old West; answer helpfully with Western slang and a folksy tone."]),
    "cats": dict(
        elicit="You are obsessed with cats: work cats, kittens or feline comparisons into every answer, "
               "whatever the topic.",
        metric=_per100(CAT), metric_name="cat_per100w",
        rung2="Mention cats.",
        rung3=["You love cats and bring them up in your answers whenever you can, even on unrelated topics."]),
    "cooking": dict(
        elicit="Explain everything through cooking: use kitchen metaphors, recipes and food analogies in every "
               "answer, whatever the subject.",
        metric=_per100(FOOD), metric_name="food_per100w",
        rung2="Use cooking analogies.",
        rung3=["Whatever the topic, explain it using analogies to cooking, recipes and the kitchen."]),
    "empathy": dict(
        elicit="Respond with deep warmth and emotional attunement: acknowledge feelings, validate the user, "
               "and be gentle and caring in every reply.",
        metric=_per100(set(), EMPATHY_P), metric_name="empathy_per100w",
        rung2="Be empathetic.",
        rung3=["Be warm and caring: acknowledge the user's feelings and respond gently and supportively."]),
    "verbose": dict(
        elicit="Give long, exhaustive answers: elaborate on every point, add background, examples, caveats "
               "and tangents, never stopping short.",
        metric=lambda t: float(len(_words(t))), metric_name="words",
        rung2="Be verbose.",
        rung3=["Write long, detailed answers that cover background, examples and every relevant nuance."]),
}

ALL_BEHAVIORS = {**BEHAVIORS, **BEHAVIORS_P2}

# mean-centered variants (Phase 2 screening): same behavior spec, vector = raw - mean over BEHAVIORS_P2 raws
BEHAVIORS_P2_CENTERED = {f"{k}_c": v for k, v in BEHAVIORS_P2.items()}
ALL_BEHAVIORS.update(BEHAVIORS_P2_CENTERED)


# ================================================================ Phase 3 holdout behaviors (never trained on)
IT = set("il lo la i gli le e è che di un una per non con si sono come più ma anche questo questa del della "
         "nel nella al alla dei delle molto può essere".split())
VICTORIAN = set("indeed whilst shall hitherto henceforth pray kindly esteemed humble therein thereof forthwith "
                "wherein heretofore notwithstanding moreover perchance accordingly".split())

BEHAVIORS_P3 = {
    "italian": dict(
        elicit="Sei un assistente di lingua italiana. Scrivi tutta la tua risposta in italiano, qualunque sia la "
               "lingua della domanda.",
        metric=_lang_score(IT), metric_name="frac_italian",
        rung2="Respond in Italian.",
        rung3=["Always answer in natural, fluent Italian, whatever language the user writes in; keep code and "
               "proper names unchanged."]),
    # INTERPRETATION (flagged): "archaic_formal" = old-fashioned, Victorian-style formal English, distinct from the
    # Elizabethan thee/thou "archaic" behavior that failed Phase 2 screening.
    "archaic_formal": dict(
        elicit="Reply in the stately, old-fashioned manner of a Victorian gentleman's correspondence: elaborate "
               "courtesy, words such as 'indeed', 'whilst', 'shall' and 'pray', long measured sentences.",
        metric=_per100(VICTORIAN), metric_name="victorian_per100w",
        rung2="Write like a Victorian gentleman.",
        rung3=["Answer in formal, old-fashioned nineteenth-century English, with elaborate courtesy and measured, "
               "ornate sentences."]),
}
ALL_BEHAVIORS.update(BEHAVIORS_P3)


# ================================================================ Phase 3b Run 4: low-cosine holdouts
# Script-based metrics (robust for non-Latin scripts, unlike the stopword heuristics).
def _script_frac(lo_hi_ranges):
    def f(t):
        chars = [c for c in t if not c.isspace()]
        if not chars:
            return 0.0
        hit = sum(any(lo <= ord(c) <= hi for lo, hi in lo_hi_ranges) for c in chars)
        return 100.0 * hit / len(chars)
    return f


japanese_score = _script_frac([(0x3040, 0x30FF), (0x4E00, 0x9FFF)])   # kana + kanji
russian_score = _script_frac([(0x0400, 0x04FF)])                      # Cyrillic
korean_score = _script_frac([(0xAC00, 0xD7AF), (0x1100, 0x11FF)])     # Hangul


def uppercase_score(t):
    letters = [c for c in t if c.isalpha()]
    return 100.0 * sum(c.isupper() for c in letters) / max(1, len(letters))


BEHAVIORS_P3B = {
    "japanese": dict(
        elicit="あなたは日本語のアシスタントです。質問が何語であっても、回答はすべて自然な日本語で書いてください。",
        metric=japanese_score, metric_name="jp_script_per100c",
        rung2="Respond in Japanese.",
        rung3=["Always answer entirely in natural, fluent Japanese, whatever language the user writes in; "
               "keep code and proper names unchanged."]),
    "russian": dict(
        elicit="Ты русскоязычный ассистент. На каком бы языке ни был задан вопрос, пиши весь ответ на русском языке.",
        metric=russian_score, metric_name="cyrillic_per100c",
        rung2="Respond in Russian.",
        rung3=["Always answer entirely in natural, fluent Russian, whatever language the user writes in; "
               "keep code and proper names unchanged."]),
    "korean": dict(
        elicit="당신은 한국어 어시스턴트입니다. 질문이 어떤 언어로 주어지든 모든 답변을 자연스러운 한국어로 작성하세요.",
        metric=korean_score, metric_name="hangul_per100c",
        rung2="Respond in Korean.",
        rung3=["Always answer entirely in natural, fluent Korean, whatever language the user writes in; "
               "keep code and proper names unchanged."]),
    "uppercase": dict(
        elicit="WRITE YOUR ENTIRE RESPONSE IN CAPITAL LETTERS. EVERY SINGLE WORD MUST BE IN UPPERCASE.",
        metric=uppercase_score, metric_name="frac_uppercase",
        rung2="WRITE IN ALL CAPS.",
        rung3=["Write every word of every response in capital letters, never using lowercase."]),
}
ALL_BEHAVIORS.update(BEHAVIORS_P3B)
