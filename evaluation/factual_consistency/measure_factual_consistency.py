import itertools
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression
from evaluation.evaluation_utils import remove_indices
# from ..token_importance.evaluate import measure_fact_importance


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

def sum_consistent_fact_score(extracted_facts, scores):
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


def clean_attrs(attrs):
    if type(attrs[-1]) == bool or (type(attrs[-1]) == tuple and type(attrs[-1][0]) == bool):
        return clean_attrs(attrs[:-1])
    if type(attrs) == list:
        if type(attrs[0])==list:
            return ' '.join([clean_attrs(x) for x in attrs])
        elif attrs[1] == 'PROPN' or attrs[1] == 'NUM':
            return attrs[1]
        else:
            return clean_attrs(attrs[0])
    elif type(attrs) == tuple:
        return ' '.join([clean_attrs(x) for x in attrs])
    else:
        if '_' in attrs:
            return attrs.split('_')[0]
        else:
            return attrs


def clean_facts(facts):
    cleaned = []
    consistent = []
    for fact in facts:
            # print(f'fact {fact}')
            # print(facts[fact])
            cleaned_subj = fact.split('_')[0]
            for attrs in facts[fact]:
                cleaned.append(f'{cleaned_subj} {clean_attrs(attrs)}')
                # for attr in attrs[:-1]:
                #     try:
                        
                #     except:
                #         print(fact, facts[fact], attr)
                #         exit()
                consistent.append(attrs[-1])
    return cleaned, consistent


def measure_factual_consistency(noun_modifiers, obj_counter, subj_verb, verb_obj, subj_verb_obj, noun_neg, event_neg,
                                event_modifiers):
    extracted_facts = combine_extracted_facts(noun_modifiers, obj_counter, subj_verb, verb_obj, subj_verb_obj, noun_neg,
                                              event_neg, event_modifiers)
    
    print('extracted facts')
    print(extracted_facts)

    simple_model = pickle.load(open('fact_lin_reg_masked.pkl', 'rb'))
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    cleaned_facts, consistent = clean_facts(extracted_facts)
    facts_importances = simple_model.predict(bert_model.encode(cleaned_facts))
    for score, fact in zip(facts_importances,cleaned_facts):
        print(score, fact)
    inconsistent_scores = []
    consistent_scores = []
    for score, is_consistent in zip(facts_importances, consistent):
            if is_consistent:
                consistent_scores.append((score,is_consistent))
            else:
                inconsistent_scores.append((score,is_consistent))
    inconsistent_facts = get_inconsistent_facts(extracted_facts)

    if inconsistent_facts:
        print("*****")
        print("Inconsistent facts found:")
        count = 0
        for subj in inconsistent_facts:
            for fact in inconsistent_facts[subj]:
                print(inconsistent_scores[count], subj, fact)
                count += 1
        print("*****")
    else:
        print(f"***** Inconsistent facts not found !!! *****")
    print()

    consistent_facts = get_consistent_facts(extracted_facts)
    print("*****")
    print("List of consistent facts below:")
    count = 0
    for subj in consistent_facts:
        for fact in consistent_facts[subj]:
            print(consistent_scores[count], subj, fact)
            count += 1
    print("*****")
    print()

    for fact in extracted_facts:
        print(fact, extracted_facts[fact])
    print(cleaned_facts)
    print(inconsistent_facts)
    print(consistent_facts)

    consistent_fact_count = count_consistent_facts(extracted_facts)

    return consistent_fact_count / len(list(itertools.chain.from_iterable(extracted_facts.values())))
