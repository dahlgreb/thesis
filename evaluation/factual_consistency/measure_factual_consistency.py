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


def clean_attrs(attr):
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
    return attr_val


def get_inconsistent_facts(extracted_facts):
    inconsistent_facts = {}
    for fact in extracted_facts:
        fact_wo_index = remove_indices(fact)
        for attr in extracted_facts[fact]:
            if not attr[1]['consistent']:
                attr_val = clean_attrs(attr)

                if fact_wo_index in inconsistent_facts:
                    inconsistent_facts[fact_wo_index].append((attr_val, attr[1]))
                else:
                    inconsistent_facts[fact_wo_index] = [(attr_val, attr[1])]
    return inconsistent_facts


def get_consistent_facts(extracted_facts):
    consistent_facts = {}
    for fact in extracted_facts:
        fact_wo_index = remove_indices(fact)
        for attr in extracted_facts[fact]:
            if attr[1]['consistent']:
                attr_val = clean_attrs(attr)

                if fact_wo_index in consistent_facts:
                    consistent_facts[fact_wo_index].append((attr_val, attr[1]))
                else:
                    consistent_facts[fact_wo_index] = [(attr_val, attr[1])]
    return consistent_facts


def clean_attrs_nl(attrs):
    if type(attrs) == list:
        if type(attrs[-1]) == bool or (type(attrs[-1]) == tuple and type(attrs[-1][0]) == bool):
            return clean_attrs_nl(attrs[:-1])
        elif type(attrs[0])==list:
            return ' '.join([clean_attrs_nl(x) for x in attrs])
        elif type(attrs[0])==tuple:
            return ' '.join([clean_attrs_nl(x) for x in attrs[0]])
        elif attrs[1] == 'PROPN' or attrs[1] == 'NUM':
            return attrs[1]
        else:
            return clean_attrs_nl(attrs[0])
    elif type(attrs) == tuple:
        return ' '.join([clean_attrs_nl(x) for x in attrs])
    elif type(attrs) == bool:
        return 'not'
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
        cleaned_phrase = fact.split('_')[0]
        for attrs in facts[fact]:
            cleaned_phrase += f' {clean_attrs_nl(attrs)}'
            consistent.append(attrs[-1])
        cleaned.extend([cleaned_phrase]*len(facts[fact]))
    return cleaned, consistent


def store_predicted_fact_importance(facts, pred_imp):
    i = 0
    for fact in facts:
        for attrs in facts[fact]:
            fact_md = {}
            if type(attrs[-1]) == bool:
                fact_md['consistent'] = attrs[-1]
                fact_md['pred_imp_score'] = pred_imp[i]
                i+=1
            elif type(attrs[-1]) == tuple:
                fact_md['consistent'] = attrs[-1][0]
                fact_md['orig_imp_score'] = attrs[-1][1]
                fact_md['pred_imp_score'] = pred_imp[i]
                i+=1
            else:
                print('bro wtf')
            if fact_md:
                attrs[-1] = fact_md


def measure_factual_consistency(noun_modifiers, obj_counter, subj_verb, verb_obj, subj_verb_obj, noun_neg, event_neg,
                                event_modifiers, simple_model, bert_model):
    extracted_facts = combine_extracted_facts(noun_modifiers, obj_counter, subj_verb, verb_obj, subj_verb_obj, noun_neg,
                                              event_neg, event_modifiers)

    # simple_model = pickle.load(open('fact_lin_reg_masked.pkl', 'rb'))
    # bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    cleaned_facts, consistent = clean_facts(extracted_facts)
    fact_importances = simple_model.predict(bert_model.encode(cleaned_facts))
    print('sentence man')
    print(simple_model.predict(bert_model.encode(['sentence man', 'NUM years old PROPN PROPN'])))
    store_predicted_fact_importance(extracted_facts, fact_importances)
    print('extracted_facts')
    print(extracted_facts)
    # for score, fact in zip(fact_importances,cleaned_facts):
    #     print(score, fact)
    inconsistent_scores = []
    consistent_scores = []
    for score, is_consistent in zip(fact_importances, consistent):
            if is_consistent:
                consistent_scores.append((score,is_consistent))
            else:
                inconsistent_scores.append((score,is_consistent))
    consistency_score = 0
    total_score = 0
    inconsistent_facts = get_inconsistent_facts(extracted_facts)

    if inconsistent_facts:
        print("*****")
        print("Inconsistent facts found:")
        for subj in inconsistent_facts:
            for fact in inconsistent_facts[subj]:
                print(fact[1]['pred_imp_score'], subj, fact[0])
                consistency_score -= fact[1]['pred_imp_score']
                total_score += fact[1]['pred_imp_score']
        print("*****")
    else:
        print(f"***** Inconsistent facts not found !!! *****")
    print()

    consistent_facts = get_consistent_facts(extracted_facts)
    print("*****")
    print("List of consistent facts below:")
    for subj in consistent_facts:
        for fact in consistent_facts[subj]:
            print(fact[1]['pred_imp_score'], subj, fact[0])
            consistency_score += fact[1]['pred_imp_score']
            total_score += fact[1]['pred_imp_score']
    print("*****")
    print()

    consistent_fact_count = count_consistent_facts(extracted_facts)

    print(f'Max possible score: {total_score}')
    print(f'Achieved score: {consistency_score}')
    print()

    return consistency_score / total_score
