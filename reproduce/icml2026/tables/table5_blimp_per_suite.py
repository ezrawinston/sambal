#!/usr/bin/env python3
"""Table 5 — BLiMP per-paradigm accuracy (%) for the two long-regime models.

Reads the committed full-BLiMP best-temperature per-paradigm results
(records/blimp/{baseline,sambal}_long_full_blimp.json), prints the table grouped
by BLiMP field the way the paper presents it, and writes a CSV. Also checks that
the mean over paradigms equals the recorded aggregate.
"""
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from records_io import TABLES_OUT, load_json  # noqa: E402

RES = "blimp"
OUT = TABLES_OUT

GROUPS = [
    ("ANAPHOR AGREEMENT", ["anaphor_gender_agreement", "anaphor_number_agreement"]),
    ("ARGUMENT STRUCTURE", ["animate_subject_passive", "animate_subject_trans", "causative",
                            "drop_argument", "inchoative", "intransitive", "passive_1",
                            "passive_2", "transitive"]),
    ("BINDING", ["principle_A_c_command", "principle_A_case_1", "principle_A_case_2",
                 "principle_A_domain_1", "principle_A_domain_2", "principle_A_domain_3",
                 "principle_A_reconstruction"]),
    ("CONTROL/RAISING", ["existential_there_object_raising", "existential_there_subject_raising",
                         "expletive_it_object_raising", "tough_vs_raising_1", "tough_vs_raising_2"]),
    ("DETERMINER-NOUN AGR.", ["determiner_noun_agreement_1", "determiner_noun_agreement_2",
                              "determiner_noun_agreement_irregular_1",
                              "determiner_noun_agreement_irregular_2",
                              "determiner_noun_agreement_with_adjective_1",
                              "determiner_noun_agreement_with_adj_2",
                              "determiner_noun_agreement_with_adj_irregular_1",
                              "determiner_noun_agreement_with_adj_irregular_2"]),
    ("ELLIPSIS", ["ellipsis_n_bar_1", "ellipsis_n_bar_2"]),
    ("FILLER-GAP", ["wh_questions_object_gap", "wh_questions_subject_gap",
                    "wh_questions_subject_gap_long_distance", "wh_vs_that_no_gap",
                    "wh_vs_that_no_gap_long_distance", "wh_vs_that_with_gap",
                    "wh_vs_that_with_gap_long_distance"]),
    ("IRREGULAR FORMS", ["irregular_past_participle_adjectives", "irregular_past_participle_verbs"]),
    ("ISLAND EFFECTS", ["adjunct_island", "complex_NP_island",
                        "coordinate_structure_constraint_complex_left_branch",
                        "coordinate_structure_constraint_object_extraction",
                        "left_branch_island_echo_question", "left_branch_island_simple_question",
                        "sentential_subject_island", "wh_island"]),
    ("NPI LICENSING", ["matrix_question_npi_licensor_present", "npi_present_1", "npi_present_2",
                       "only_npi_licensor_present", "only_npi_scope",
                       "sentential_negation_npi_licensor_present", "sentential_negation_npi_scope"]),
    ("QUANTIFIERS", ["existential_there_quantifiers_1", "existential_there_quantifiers_2",
                     "superlative_quantifiers_1", "superlative_quantifiers_2"]),
    ("SUBJECT-VERB AGR.", ["distractor_agreement_relational_noun",
                           "distractor_agreement_relative_clause",
                           "irregular_plural_subject_verb_agreement_1",
                           "irregular_plural_subject_verb_agreement_2",
                           "regular_plural_subject_verb_agreement_1",
                           "regular_plural_subject_verb_agreement_2"]),
]


def main():
    data = {arm: load_json(f"{RES}/{arm}_long_full_blimp.json")
            for arm in ("baseline", "sambal")}
    per = {arm: data[arm]["per_uid"] for arm in data}

    listed = [u for _, uids in GROUPS for u in uids]
    for arm in per:
        missing = set(per[arm]) - set(listed)
        extra = set(listed) - set(per[arm])
        assert not missing and not extra, (missing, extra)
        mean = statistics.mean(per[arm].values())
        agg = data[arm]["best_temp_avg_uid_accuracy"]
        assert abs(mean - agg) < 0.01, (arm, mean, agg)

    print(f"{'BLiMP paradigm':<52} {'baseline':>9} {'SAMBAL':>8}")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "table5_blimp_per_suite.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "paradigm", "baseline", "sambal"])
        for group, uids in GROUPS:
            print(group)
            for u in uids:
                print(f"  {u:<50} {per['baseline'][u]:>9.1f} {per['sambal'][u]:>8.1f}")
                w.writerow([group, u, f"{per['baseline'][u]:.1f}", f"{per['sambal'][u]:.1f}"])
    print(f"\nOverall: baseline {statistics.mean(per['baseline'].values()):.1f}  "
          f"SAMBAL {statistics.mean(per['sambal'].values()):.1f}  ({len(listed)} paradigms)")
    print(f"wrote {OUT}/table5_blimp_per_suite.csv")


if __name__ == "__main__":
    main()
