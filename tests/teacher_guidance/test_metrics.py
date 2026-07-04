"""Tests for teacher_guidance.metrics."""

from agentsim.teacher_guidance.metrics import (
    normalize_answer,
    exact_match,
    cover_match,
    f1_score,
    supporting_doc_recall,
    supporting_fact_recall,
    binary_from_continuous,
    compute_final_metrics,
)


def test_normalize_answer_strips_articles_and_punct():
    assert normalize_answer("The Delhi, India!") == "delhi india"


def test_exact_match_is_normalized():
    assert exact_match("the Delhi", "Delhi")
    assert not exact_match("Mumbai", "Delhi")


def test_f1_partial_overlap():
    assert round(f1_score("Delhi India", "Delhi"), 4) == 0.6667
    assert f1_score("", "Delhi") == 0.0
    assert f1_score("Delhi", "Delhi") == 1.0


def test_f1_value():
    # precision 1/2, recall 1/1 -> F1 = 2*0.5*1/1.5 = 0.6667
    assert round(f1_score("Delhi city", "Delhi"), 4) == 0.6667


def test_cover_match_answer_plus_explanation():
    # The exact case from the UI: gold "no", student answered "No. <explanation>"
    pred = "No. Roger Donaldson is an Australian-born New Zealand filmmaker, while Andre Cayatte was a French filmmaker."
    assert exact_match(pred, "no") is False
    assert cover_match(pred, "no") is True


def test_cover_match_entity_contained():
    assert cover_match("The answer is The Oberoi Group.", "The Oberoi Group") is True
    assert cover_match("It is located in Delhi today", "Delhi") is True


def test_cover_match_short_gold_must_lead():
    # incidental "no" deep in a wrong answer must NOT count
    assert cover_match("There is no clear winner", "no") is False
    assert cover_match("No, definitely", "no") is True


def test_cover_match_prediction_is_core_of_gold():
    # The UI case: gold "22 episodes.", student answered "22"
    assert exact_match("22", "22 episodes.") is False
    assert cover_match("22", "22 episodes.") is True
    # number-bearing core inside a longer gold
    assert cover_match("1996", "the year 1996 census") is True
    # half-coverage core
    assert cover_match("Delhi", "Delhi, India") is True


def test_cover_match_partial_entity_rejected():
    # a partial entity that is neither numeric nor half the gold must NOT count
    assert cover_match("York", "New York City") is False


def test_cover_match_prefix_token():
    # The UI case: gold "KXII", student answered "...is KXII-TV (virtual channel 12)..."
    pred = "The CBS-affiliated station serving Pontotoc County is KXII-TV (virtual channel 12)."
    assert exact_match(pred, "KXII") is False
    assert cover_match(pred, "KXII") is True
    # a weak prefix (covers < 60% of the token) must NOT count
    assert cover_match("Saint Bartholomew the Apostle", "Bart") is False


def test_ampersand_normalized_to_and():
    # gold uses "and", student used "&"
    assert normalize_answer("Centers for Medicare & Medicaid Services") == \
        normalize_answer("Centers for Medicare and Medicaid Services")


def test_cover_match_ampersand_in_verbose_answer():
    pred = "The program is administered by the Centers for Medicare & Medicaid Services (CMS)."
    assert cover_match(pred, "Centers for Medicare and Medicaid Services") is True


def test_cover_match_negative():
    assert cover_match("Mumbai", "Delhi") is False
    assert cover_match("", "Delhi") is False


# ---------------------------------------------------------------------------
# Real false-positive cases collected from the viewer (student answers marked
# INCORRECT despite containing the gold answer's actual content) -- see
# ignored-temp/FalsePositives/fn01..08.png. Each of these was previously rejected by
# the strict contiguous-span check because the prediction dropped a middle name,
# inserted a filler word, reordered a generic word, or the gold was a short entity
# name that didn't lead the prediction.
# ---------------------------------------------------------------------------
def test_cover_match_dropped_middle_name():
    # fn01: gold has a middle name the student's answer omits.
    pred = "Kelly Osbourne, a British singer-songwriter, hosted the 16th Annual Young Hollywood Awards in July 2014."
    assert cover_match(pred, "Kelly Lee Osbourne") is True


def test_cover_match_inserted_filler_word():
    # fn02/fn08: gold "born October 25, 1931" vs prediction inserting "on".
    pred = "The designer of Autopia is Bob Gurr. He was born on October 25, 1931."
    assert cover_match(pred, "born October 25, 1931") is True


def test_cover_match_short_proper_noun_anywhere():
    # fn03: a short (<=3 char) gold that is a proper noun, not a common word like
    # "no", may match anywhere in the prediction, not just as the first word.
    pred = "Tropical Storm Ana has been present in the Central Pacific Ocean but not in the western North Pacific Ocean."
    assert cover_match(pred, "Ana") is True


def test_cover_match_short_acronym_anywhere():
    # fn04: same as above but for an all-caps acronym.
    pred = "Mamie Gummer played Nancy Crozier on 'The Good Wife', which aired on CBS."
    assert cover_match(pred, "CBS") is True


def test_cover_match_reordered_generic_word():
    # fn05: "Club" is reused later in the sentence as a generic noun, and "de" is
    # dropped -- coverage must still count "club"/"atlético"/"madrid" as matched.
    pred = "The Amsterdam Tournament 2009 was contested by Atlético Madrid, which is the club that plays its home games at Wanda Metropolitano."
    assert cover_match(pred, "Club Atlético de Madrid") is True


def test_cover_match_pronoun_subject_replaced_by_name():
    # fn06: gold phrased with a pronoun subject ("He is..."), prediction uses the
    # actual name instead of the pronoun.
    pred = "Sulley Muniru is the younger brother of Sulley Muntari, who plays as a central midfielder for the Italian club Pescara."
    assert cover_match(pred, "He is the younger brother") is True


def test_cover_match_extra_patronymic_inserted():
    # fn07: gold "Alexander Gorsky" vs prediction inserting a patronymic between the
    # two gold tokens.
    pred = '"Alexander Alexeyevich Gorsky" restaged the play "Don Quixote", based on the Spanish novel by Miguel de Cervantes Saavedra.'
    assert cover_match(pred, "Alexander Gorsky") is True


def test_cover_match_coverage_fallback_still_requires_majority():
    # Guard-rail: a 2-token gold with only one token present (50% < 60% threshold)
    # must NOT count -- coverage must not degrade into "any single word in common".
    assert cover_match("I think it might be Obama, not sure.", "Barack Obama") is False


def test_cover_match_short_common_word_still_must_lead():
    # Guard-rail: the fix for short proper nouns must not loosen the existing
    # must-lead rule for ordinary short words like "no".
    assert cover_match("There is no clear winner", "no") is False


def test_supporting_doc_recall():
    assert supporting_doc_recall({"d1", "d2"}, {"d1", "d3"}) == 0.5
    assert supporting_doc_recall(set(), {"d1"}) == 0.0
    assert supporting_doc_recall({"d1"}, set()) == 0.0


def test_supporting_fact_recall_verbatim():
    corpus = {
        "d0": {"title": "T", "sentences": ["gold sentence here", "other"]},
    }
    gold_facts = [{"title": "T", "sent_id": 0}]
    extracted_ok = [{"doc_id": "d0", "span": "gold sentence here", "fact": "x"}]
    extracted_bad = [{"doc_id": "d0", "span": "paraphrased text", "fact": "x"}]
    assert supporting_fact_recall(extracted_ok, gold_facts, corpus) == 1.0
    assert supporting_fact_recall(extracted_bad, gold_facts, corpus) == 0.0


def test_binary_from_continuous():
    assert binary_from_continuous(0.75) == 1
    assert binary_from_continuous(0.74) == 0


def test_compute_final_metrics():
    corpus = {"d0": {"title": "T", "sentences": ["gold sentence"]}}
    m = compute_final_metrics(
        final_answer="the Delhi",
        gold_answer="Delhi",
        retrieved_doc_ids={"d0"},
        gold_doc_ids={"d0"},
        extracted_spans=[{"doc_id": "d0", "span": "gold sentence", "fact": "x"}],
        gold_facts=[{"title": "T", "sent_id": 0}],
        corpus=corpus,
    )
    assert m["exact_match"] is True
    assert m["f1"] == 1.0
    assert m["supporting_doc_recall"] == 1.0
    assert m["supporting_fact_recall"] == 1.0
