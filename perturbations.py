import checklist
from checklist.perturb import Perturb
from checklist.test_types import MFT, INV, DIR
from checklist.test_suite import TestSuite
import datasets
from datasets import Dataset
import spacy

#suite = TestSuite()



def adding_typos(dataset):
    nTypos = 100
    sentences = dataset["hypothesis"]
    t = Perturb.perturb(sentences, Perturb.add_typos, nsamples=0, typos=nTypos, keep_original = False)
    new_dataset = [p[0] for p in t.data]

    premises = dataset["premise"]
    t2 = Perturb.perturb(premises, Perturb.add_typos, nsamples=0, typos=nTypos, keep_original = False)
    new_dataset_premises = [p[0] for p in t2.data]
        
    def augment_data(data, index):
        if index < len(new_dataset):
            return {"premise": new_dataset_premises[index],
                    "hypothesis": new_dataset[index]}
        else: 
            return {"premise": data["premise"],
                    "hypothesis": data["hypothesis"]}
        
    return dataset.map(augment_data, new_dataset)

def processing(premises):
    new_dataset_premises = []
    new_sentence = ""
    for sentence in premises:
        ex=Perturb.expand_contractions(sentence)
        if sentence != ex:
            new_sentence = ex
            
        c = Perturb.contract(sentence)
        if sentence != c:
            new_sentence = c
               
        if new_sentence == "":
            new_sentence = sentence
            
        new_dataset_premises.append(new_sentence)
        new_sentence = ""
        
    return new_dataset_premises


def changing_contractions(dataset):
    
    premises = list(dataset["premise"])
    new_dataset_premises = processing(premises)

    hypothesis = dataset["hypothesis"]
    new_dataset_hyp = processing(hypothesis)

    def augment_data(data, index):
        if index < len(new_dataset_hyp):
            return {"premise": new_dataset_premises[index],
                    "hypothesis": new_dataset_hyp[index]}
        else: 
            return {"premise": data["premise"],
                    "hypothesis": data["hypothesis"]}
        
    return dataset.map(augment_data, new_dataset_hyp)


def negating_hyp(dataset):
    nlp = spacy.load('en_core_web_sm')
    premises, hyp, label = [],[],[]
    i = 0
    per = Perturb()
    for data in dataset:
        i += 1
        d = nlp(data["premise"])
        print (i, d)
        p = per.add_negation(d)
        if p != None:
            print("perturbating")
            premises.append(p)
            hyp.append("hypothesis")
            label.append(data["label"])
        else:
            premises.append(data["premise"])
            hyp.append(data["hypothesis"])
            label.append(data["label"])

    output = {"premise": premises,
            "hypothesis": hyp, 
            "label": label}
    return Dataset.from_dict(output)
    




