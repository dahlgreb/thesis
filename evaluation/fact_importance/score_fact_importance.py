import pickle
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression

def clean_attrs(attrs):
    if type(attrs[-1]) == bool:
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


def clean_extracted_facts(facts):
    cleaned = []
    consistent = []
    for fact in facts:
            # print(f'fact {fact}')
            # print(facts[fact])
            cleaned_subj = fact.split('_')[0]
            for attrs in facts[fact]:
                for attr in attrs[:-1]:
                    cleaned.append(f'{cleaned_subj} {" ".join([attr[0].split("_")[0]])}')
                consistent.append(attrs[-1])
    return cleaned, consistent


def clean_fact_table(table):
    cleaned_facts = []
    for fact in table:
        fact_type = list(fact)[0]
        cleaned_fact = []
        if fact_type == 'event':
            for attr in fact[fact_type]:
                if attr not in ['passive', 'neg', 'phrase_mod']:
                    cleaned_fact.append(fact[fact_type][attr])
                elif attr == 'neg':
                    cleaned_fact.append('no')
                elif attr == 'phrase_mod':
                    if type(fact[fact_type][attr]) == list:
                        cleaned_fact.append(' '.join(fact[fact_type][attr]))
                    else:
                        cleaned_fact.append(fact[fact_type][attr])
        else:
            for attr in fact[fact_type]:
                cleaned_fact.append(fact[fact_type][attr])
        cleaned_facts.append(' '.join(cleaned_fact))
    return cleaned_facts

            

simple_model = pickle.load(open('fact_lin_reg_masked.pkl', 'rb'))
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_fact_importance_table(fact_table):
    cleaned_facts = clean_fact_table(fact_table)
    scores = simple_model.predict(bert_model.encode(cleaned_facts))
    for fact, score in zip(cleaned_facts, scores):
        print(score, fact)
    return scores