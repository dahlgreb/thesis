import itertools

from evaluation.evaluation_utils import remove_indices
from ..token_importance.evaluate import measure_fact_importance


def combine_extracted_facts(noun_modifiers, obj_counter, subj_verb, verb_obj, subj_verb_obj, noun_neg, event_neg,
                            event_modifiers):
    extracted_facts = {}

    facts = [noun_modifiers, obj_counter, subj_verb, verb_obj, subj_verb_obj, noun_neg, event_neg, event_modifiers]

    for fact in facts:
        for key in fact:
            vals = fact[key]
            if key in extracted_facts:
                extracted_facts[key].extend(vals)
            else:
                extracted_facts[key] = vals.copy()
    return extracted_facts


def count_consistent_facts(extracted_facts):
    num_consistent_fact = 0
    for key in extracted_facts:
        for val in extracted_facts[key]:
            if val[1]:
                num_consistent_fact += 1
    return num_consistent_fact


def get_inconsistent_facts(extracted_facts):
    inconsistent_facts = {}
    for fact in extracted_facts:
        for attr in extracted_facts[fact]:
            if not attr[1]:
                fact_wo_index = remove_indices(fact)

                if type(attr[0][0]) == list:
                    attr_val = []
                    for el in attr[0]:
                        val = el[0]
                        is_passive = False
                        if 'passive' in val:
                            is_passive = True
                            val = val.rsplit('_', 1)[0]

                        attr_val.append(remove_indices(val) + ('(passive)' if is_passive else ''))
                    attr_val = ', '.join(attr_val)
                else:
                    attr_val = attr[0][0]

                if fact_wo_index in inconsistent_facts:
                    inconsistent_facts[fact_wo_index].append(attr_val)
                else:
                    inconsistent_facts[fact_wo_index] = [attr_val]
    return inconsistent_facts


def get_consistent_facts(extracted_facts):
    consistent_facts = {}
    for fact in extracted_facts:
        for attr in extracted_facts[fact]:
            if attr[1]:
                fact_wo_index = remove_indices(fact)

                if type(attr[0][0]) == list:
                    attr_val = []
                    for el in attr[0]:
                        val = el[0]
                        is_passive = False
                        if 'passive' in val:
                            is_passive = True
                            val = val.rsplit('_', 1)[0]

                        attr_val.append(remove_indices(val) + ('(passive)' if is_passive else ''))
                    attr_val = ', '.join(attr_val)
                else:
                    attr_val = attr[0][0]

                if fact_wo_index in consistent_facts:
                    consistent_facts[fact_wo_index].append(attr_val)
                else:
                    consistent_facts[fact_wo_index] = [attr_val]
    return consistent_facts

def clean_facts(facts):
    cleaned = []
    for k in facts:
        cleaned_fact = ''
        cleaned_fact += k
        for tok in facts[k]:
            if type(tok) == str:
                if '_' in tok:
                    cleaned_fact += ' '+tok.split('_')[0]
                else:
                    cleaned_fact += ' '+tok
            elif type(tok) == list:
                for sub_tok in tok:
                    if '_' in sub_tok:
                        cleaned_fact += ' '+sub_tok.split('_')[0]
                    else:
                        cleaned_fact += ' '+sub_tok
        cleaned.append(cleaned_fact)
    return cleaned


def measure_factual_consistency(noun_modifiers, obj_counter, subj_verb, verb_obj, subj_verb_obj, noun_neg, event_neg,
                                event_modifiers):
    extracted_facts = combine_extracted_facts(noun_modifiers, obj_counter, subj_verb, verb_obj, subj_verb_obj, noun_neg,
                                              event_neg, event_modifiers)

    inconsistent_facts = get_inconsistent_facts(extracted_facts)
    i_facts_importance = measure_fact_importance(clean_facts(inconsistent_facts))
    print()
    if inconsistent_facts:
        print(f"***** Inconsistent facts found: {inconsistent_facts} !!! *****")
        print(f'importance_scores: {i_facts_importance}')
    else:
        print(f"***** Inconsistent facts not found !!! *****")
    print()

    consistent_facts = get_consistent_facts(extracted_facts)
    c_facts_importance = measure_fact_importance(clean_facts(consistent_facts))
    print("List of consistent facts below:")
    for i in range(len(consistent_facts)):
        fact = list(consistent_facts.keys())[i]
        print(f"\t{i + 1}. {fact}: {consistent_facts[fact]}  imp: {c_facts_importance[i]}")
    print()
    consistent_fact_count = count_consistent_facts(extracted_facts)

    return consistent_fact_count / len(list(itertools.chain.from_iterable(extracted_facts.values())))
