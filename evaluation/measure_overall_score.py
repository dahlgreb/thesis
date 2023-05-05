import numpy as np

from evaluation.comprehensiveness.measure_compressiveness import measure_comprehensiveness
from evaluation.compression.measure_compression_rate import measure_compression_rate
from evaluation.evaluation_utils import count_matched_fact
from evaluation.factual_consistency.measure_factual_consistency import measure_factual_consistency, clean_facts
from extractor.extract_fact import extract_facts_from_summary, cosine_similarity
from analyzer.wordnet_synsets.wordnet_synsets import load_synsets
from evaluation.fact_importance.score_fact_importance import get_fact_importance_table, get_table_embeddings
# from evaluation.token_importance import *
# from evaluation.token_importance.evaluate import measure_token_importance
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression


def load_embeddings(pretrained_embeddings_path):
    embeddings_dict = {}

    with open(pretrained_embeddings_path, 'r') as fr:
        for line in fr:
            tokens = line.split()
            word = tokens[0]
            embed = np.array(tokens[1:], dtype=np.float64)
            embeddings_dict[word] = embed
    return embeddings_dict


def measure_overall_quality_score(summary, source, table, nlp, configs):
    """

    :param summary:
    :param source:
    :param table:
    :param nlp:
    :param configs: eval configurations
    :return:
    """

    # hyperparameters
    tau = float(configs['tau'])
    alpha = float(configs['alpha'])
    beta = float(configs['beta'])
    grammar_type = str(configs['grammar_type'])
    # for different grammar type, noun that can be rewritten as "victim"
    victim_maps = configs['victim_maps']
    threshold = float(configs['threshold'])
    pretrained_embeddings_path = configs['pretrained_embedding_path']

    embeddings_dict = load_embeddings(pretrained_embeddings_path)
    noun_synsets, adj_synsets, adv_synsets, verb_synsets = load_synsets()

    simple_model = pickle.load(open('fact_lin_reg_masked.pkl', 'rb'))
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')

    victim_map = victim_maps[grammar_type]
    table_embeddings = get_table_embeddings(table, nlp, bert_model)
    subj_facts = []
    subj_embeddings = []
    for fact, embedding in zip(table, table_embeddings):
        if 'person' in fact or 'object' in fact:
            subj_facts.append(fact)
            subj_embeddings.append(embedding)
    
    noun_modifiers, obj_counter, subj_verb, verb_obj, subj_verb_obj, noun_neg, event_neg, event_modifiers = extract_facts_from_summary(
        summary, nlp, subj_embeddings)
    
    table_importance = get_fact_importance_table(table_embeddings, simple_model)

    # print('nmod check')
    # # print(clean_facts(noun_modifiers))
    # coref_noun_modifiers = {}
    # coref_obj_counter = {}
    # coref_subj_verb = {}
    # coref_verb_obj = {}
    # coref_subj_verb_obj = {}
    # coref_noun_neg = {}
    # coref_event_neg = {}
    # coref_event_modifiers = {}
    # for nmod in noun_modifiers:
    #     cleaned_nmod = clean_facts({nmod:noun_modifiers[nmod]})[0][0]
    #     cosims = cosine_similarity(subj_embeddings, bert_model.encode(cleaned_nmod))
    #     replace_fact = subj_facts[np.argmax(cosims)]
    #     if 'person' in replace_fact:
    #         replace_subj = f'{replace_fact["person"]["kind"]}_{np.argmax(cosims)}'
    #         coref_noun_modifiers = {replace_subj: noun_modifiers[nmod]}
    #         coref_noun_modifiers[replace_subj].append(['female', 'ADJ'])
    #         print(coref_noun_modifiers)
    #         print('testing')
    #         fact_count, seen_facts = count_matched_fact(table, table_importance, coref_noun_modifiers, obj_counter, subj_verb, verb_obj, subj_verb_obj, noun_neg,
    #                                 event_neg, event_modifiers, victim_map, embeddings_dict, noun_synsets, adj_synsets,
    #                                 adv_synsets, verb_synsets, threshold)
    #         print(fact_count, seen_facts)
    # for fact, score in zip(table, table_importance):
    #     print(score, fact)
    table_importance = table_importance-0.5
    # table_importance = table_importance/np.linalg.norm(table_importance)
    # table_importance = (table_importance - np.mean(table_importance))/np.std(table_importance)
    table_importance = 1/(1+np.e**(-10*table_importance))
    fact_count, seen_facts = count_matched_fact(table, table_importance, noun_modifiers, obj_counter, subj_verb, verb_obj, subj_verb_obj, noun_neg,
                                    event_neg, event_modifiers, victim_map, embeddings_dict, noun_synsets, adj_synsets,
                                    adv_synsets, verb_synsets, threshold)
    print('seen facts')
    print(seen_facts)
    retrieved_important = 0
    for fact_index in seen_facts:
        print(table_importance[fact_index], table[fact_index])
        # if table_importance[fact_index] > 0.51:
        retrieved_important += table_importance[fact_index]
    unseen_facts = set(range(len(table))) - seen_facts
    print('unseen important facts')
    # print(unseen_facts)
    for fact_index in unseen_facts:
        # if table_importance[fact_index] > 0.51:
        print(table_importance[fact_index], table[fact_index])
    # retrieved important / all important
    all_important = 0
    for fact_imp in table_importance:
        # if fact_imp > 0.51:
        all_important += fact_imp
    importance_recall = retrieved_important/all_important
    compression_rate = measure_compression_rate(summary, source)
    comprehensiveness = measure_comprehensiveness(table, fact_count)
    factual_consistency = measure_factual_consistency(noun_modifiers, obj_counter, subj_verb, verb_obj, subj_verb_obj,
                                                      noun_neg,
                                                      event_neg, event_modifiers, simple_model, bert_model)

    # token_importance_score = measure_token_importance(summary)
    print(f'Factual consistency score: {factual_consistency}')
    print(f'Comprehensiveness score: {comprehensiveness}')
    print(f'Compression rate: {compression_rate}')
    print(f'Fact importance recall: {importance_recall}')

    cp = np.exp(tau - compression_rate) if tau - compression_rate < 0 else 1

    s = 0 if not comprehensiveness else (alpha * (comprehensiveness * cp) + beta * factual_consistency) / (alpha + beta)
    print(f'Overall quality of the summarizer: {s}')
    # exit()

if __name__ == '__main__':
    import json
    import spacy

    summary = """
        Abu Romneys, 18 years old, was stabbed to death after a fight broke out in a nightclub in the all-inclusive resort last August.
    """
    source = """
        An American tourist had been sentenced to 10 years in prison over the stabbing death of a British soldier during a nightclub brawl in Pakistan.

The stab took place during a fight between one British soldier and two American tourists at this nightclub.

The soldier was described as about 18 years old, wearing a green jacket with a Chinese collar and cuffs, with blue jeans and big shoes. The first tourist was described as about 59 years old, wearing a striped jacket with a blue collar and cuffs, with black jeans and pink shoes. The second tourist was described as about 35 years old, wearing a light-colored jacket with a pink collar and cuffs, with blue jeans and sensible shoes.

Abu Romneys, 18 years old, was stabbed to death after a fight broke out in a nightclub in the all-inclusive resort last August.

The victim's attorney said on Tuesday "my client did not do anything to bring about the trouble and was attacked by two people stabbing at him at the nightclub."

"Our lives are completely destroyed.", the victim's mom said, wiping tears.

"The facts and circumstances that led up to this stab are still being determined.", police said. "We are trying to piece everything together." Police are asking for the cooperation of the public to come forward and help us with the investigation. The police chief earlier called the incident heartbreaking. Investigators are collecting evidence from the crime scene, officials said.

"It was scary. We were just trying to get to safety," a witness said.

Investigators are still working to determine what led up to the stab. They, however, said some altercation occurred when stab were happened. Police believe the motive for the stab is connected to an ongoing dispute between the suspect and the victim. Two knives have been seized that were used in the stab.

Anyone with information was asked to contact the authorities.
    """

    fr_eval = open("./evaluation_config.json", 'r')

    eval_configs = json.load(fr_eval)

    # Load your usual SpaCy model (one of SpaCy English models)
    nlp = spacy.load('en_core_web_lg')

    measure_overall_quality_score(summary, source, table, nlp, eval_configs)
