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
        try:
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
                for attr in reversed(fact[fact_type]):
                    if attr == 'nationality':
                        continue
                    elif attr == 'neg':
                        cleaned_fact.append('no')
                    elif attr != 'phrase_mod':
                        cleaned_fact.append(fact[fact_type][attr])
                if 'phrase_mod' in fact[fact_type]:
                    cleaned_fact.append(fact[fact_type]['phrase_mod'])
            cleaned_facts.append(' '.join(cleaned_fact))
        except Exception as e:
            print(fact)
            print(e)
            exit()
    return cleaned_facts


def mask_table_facts(cleaned_facts,nlp):
    masked_facts = []
    tok_facts = [nlp(fact) for fact in cleaned_facts]
    for toks in tok_facts:
        masked_fact = []
        for tok in toks:
            if tok.pos_ in ['PROPN','NUM']:
                masked_fact.append(tok.pos_)
            else:
                masked_fact.append(tok.text)
        masked_facts.append(' '.join(masked_fact))
    return masked_facts


def get_fact_importance_table(fact_table, nlp, simple_model, bert_model):
    cleaned_facts = clean_fact_table(fact_table)
    masked_cleaned_facts = mask_table_facts(cleaned_facts, nlp)
    scores = simple_model.predict(bert_model.encode(masked_cleaned_facts))
    for fact, score in zip(masked_cleaned_facts, scores):
        print(score, fact)
    return scores