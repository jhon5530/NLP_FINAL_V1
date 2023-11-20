import checklist
from checklist.perturb import Perturb
from checklist.test_types import MFT, INV, DIR
from checklist.test_suite import TestSuite
from checklist.editor import Editor
import datasets
from datasets import Dataset
import spacy
import random
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet 

#suite = TestSuite()



def adding_typos(dataset):
    nTypos = 1
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
            
        c = Perturb.contract(sentence)
        if sentence != c:
            print("Perturbating from: ", sentence, " --to-- ", c)
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


def processing_expanding(premises):
    new_dataset_premises = []
    new_sentence = ""
    for sentence in premises:
        ex=Perturb.expand_contractions(sentence)
        if sentence != ex:
            print("Perturbating from: ", sentence, " --to-- ", ex)
            new_sentence = ex
                   
        if new_sentence == "":
            new_sentence = sentence
            
        new_dataset_premises.append(new_sentence)
        new_sentence = ""
        
    return new_dataset_premises

def expanding_contractions(dataset):
    
    premises = list(dataset["premise"])
    new_dataset_premises = processing_expanding(premises)

    hypothesis = dataset["hypothesis"]
    new_dataset_hyp = processing_expanding(hypothesis)

    def augment_data(data, index):
        if index < len(new_dataset_hyp):
            return {"premise": new_dataset_premises[index],
                    "hypothesis": new_dataset_hyp[index]}
        else: 
            return {"premise": data["premise"],
                    "hypothesis": data["hypothesis"]}
        
    return dataset.map(augment_data, new_dataset_hyp)

def negating_premises(dataset):
    nlp = spacy.load('en_core_web_sm')
    premises, hyp, label = [],[],[]
    i = 0
    per = Perturb()
    new_list = []
    it = iter(dataset["premise"]) 
    for data in dataset:
        d = nlp(next(it))
        i += 1
        try:
            p = per.add_negation(d)
        except:
            print("Exception")
            p = per.add_negation(d)
        ###print ("Perturb", p)
        if p != None:
            print(i, "Perturbating from: ", d, " ---to--- ", p)
            print(data["hypothesis"], data["label"])
            premises.append(p)
            hyp.append(data["hypothesis"])
            label.append(data["label"])
        else:
            premises.append(data["premise"])
            hyp.append(data["hypothesis"])
            label.append(data["label"])

    output = {"premise": premises,
            "hypothesis": hyp, 
            "label": label}
    return Dataset.from_dict(output)

def negating_hyp(dataset):
    nlp = spacy.load('en_core_web_sm')
    premises, hyp, label = [],[],[]
    i = 0
    per = Perturb()
    new_list = []
    it = iter(dataset["hypothesis"]) 
    for data in dataset:
        d = nlp(next(it))
        i += 1
        try:
            p = per.add_negation(d)
        except:
            print("Exception")
            p = per.add_negation(d)
        if p != None:
            print(data["premise"], data["label"])
            print(i, "Perturbating hyp from: ", d, " ---to--- ", p)
            premises.append(data["premise"])
            hyp.append(p)
            label.append(data["label"])
        else:
            premises.append(data["premise"])
            hyp.append(data["hypothesis"])
            label.append(data["label"])

    output = {"premise": premises,
            "hypothesis": hyp, 
            "label": label}
    return Dataset.from_dict(output)

def removing_negating_hyp(dataset):
    nlp = spacy.load('en_core_web_sm')
    premises, hyp, label = [],[],[]
    i = 0
    per = Perturb()
    for data in dataset:
        i += 1
        d = nlp(data["premise"])
        print (i, d)
        p = per.remove_negation(d)
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


def changing_names_entities(dataset):
    nlp = spacy.load('en_core_web_sm')
    r = random.randint(0,100)
    premises, hyp, label = [],[],[]
    per = Perturb()
    for data in dataset:
        d_p = nlp(data["premise"])
        d_h = nlp(data["hypothesis"])
        p_p = per.change_names(d_p, seed=r, n=1)
        p_h = per.change_names(d_h, seed=r, n=1)

        if (p_p != None) and p_h != None:
            print("perturbating", p_p, p_h)
            premises.append(p_p[0])
            hyp.append(p_h[0])
            label.append(data["label"])
        else:
            premises.append(data["premise"])
            hyp.append(data["hypothesis"])
            label.append(data["label"])

    output = {"premise": premises, "hypothesis": hyp, "label": label}
    return Dataset.from_dict(output)


def find_first_noun(text: str, nlp):
    spacy_text = nlp(text)
    nouns = [word.text for word in spacy_text if word.tag_ == "NN"]
    if len(nouns) >= 1:
        return nouns[0]
    return nouns
    

def changing_first_noun(dataset):
    nlp = spacy.load('en_core_web_sm')
    
    editor = Editor()
    nltk.download("brown")
    nltk.download("punkt")
    premises, hyp, hyper, label = [],[],[],[]
    per = Perturb()
    index = dataset["premise"]
    i = 0
    
    for data in dataset:
        data_hyp = data["hypothesis"]
        num_words = 1
        noun = find_first_noun(data_hyp, nlp)
        if noun:
            ###print(data_hyp, "Rel: ", noun)
            try: 
                related_nouns = editor.related_words(data_hyp, noun)[:num_words]
            except:
                related_nouns = []

            if related_nouns: 
                premises.append(data["premise"])
                ###print(data_hyp, "Rel: ", noun, "others: ",  related_nouns)
                hyp.append(data_hyp.replace(noun, related_nouns[0]))
                if data["label"] == 0:
                    label.append(2)
                elif data["label"] == 1:
                    label.append(2)
                else: 
                    label.append(data["label"])
                continue
        ###print("Jumping")
        premises.append(data["premise"])
        hyp.append(data_hyp)
        label.append(data["label"])

    output = {"premise": premises,
            "hypothesis": hyp, 
            "label": label}
    
    return Dataset.from_dict(output)


def addany2_end(dataset):
    nltk.download('punkt')
    synonyms = [] 
    antonyms = []
    premises, hyp, label = [],[],[] 
    nltk.download('averaged_perceptron_tagger')
    editor = Editor()

    for data in dataset:
        tokens = word_tokenize(data["hypothesis"])

        parts_of_speech = nltk.pos_tag(tokens)
        ###print("-----", "\n")
        ###print(parts_of_speech)
        nouns = list(filter(lambda x: x[1] == "NN", parts_of_speech))
        nouns_list = [n[0] for n in nouns]
        dt = list(filter(lambda x: x[1] == "DT", parts_of_speech))
        dt = [n[0] for n in dt]
        vbg = list(filter(lambda x: x[1] == "VBG", parts_of_speech))
        vbg = [n[0] for n in vbg]

        ###print("nouns_list: ", nouns_list)
        ###print("dt: ", dt)
        ###print("vbg: ", vbg)

        for noun in nouns_list:
            for syn in wordnet.synsets(noun): 
                for l in syn.lemmas(): 
                    synonyms.append(l.name())
                    if l.antonyms(): 
                        antonyms.append(l.antonyms()[0].name())
        nouns_ant = list(set(antonyms))

        for v in vbg:
            for syn in wordnet.synsets(v): 
                for l in syn.lemmas(): 
                    synonyms.append(l.name())
                    if l.antonyms(): 
                        antonyms.append(l.antonyms()[0].name())
        vbg_ant = list(set(antonyms))
        ###print("nouns_ant: ", nouns_ant)
        ###print("vbg_ant: ", vbg_ant)

        new_aa = dt + nouns_ant + vbg_ant
        random.shuffle(new_aa)
        
        if new_aa == []:
            premises.append(data["premise"])
            hyp.append(data["hypothesis"])
            label.append(data["label"])
        else:
            premises.append(data["premise"])
            ###print("AA: ", new_aa[:5])
            hyp.append(data["hypothesis"]+ " "+ ' '.join(new_aa[:5]))
            label.append(data["label"])
            

    output = {"premise": premises,
            "hypothesis": hyp, 
            "label": label}
    
    return Dataset.from_dict(output)


def addany2_begin(dataset):
    nltk.download('punkt')
    synonyms = [] 
    antonyms = []
    premises, hyp, label = [],[],[] 
    nltk.download('averaged_perceptron_tagger')
    editor = Editor()

    for data in dataset:
        tokens = word_tokenize(data["hypothesis"])

        parts_of_speech = nltk.pos_tag(tokens)
        ###print("-----", "\n")
        ###print(parts_of_speech)
        nouns = list(filter(lambda x: x[1] == "NN", parts_of_speech))
        nouns_list = [n[0] for n in nouns]
        dt = list(filter(lambda x: x[1] == "DT", parts_of_speech))
        dt = [n[0] for n in dt]
        vbg = list(filter(lambda x: x[1] == "VBG", parts_of_speech))
        vbg = [n[0] for n in vbg]

        ###print("nouns_list: ", nouns_list)
        ###print("dt: ", dt)
        ###print("vbg: ", vbg)

        for noun in nouns_list:
            for syn in wordnet.synsets(noun): 
                for l in syn.lemmas(): 
                    synonyms.append(l.name())
                    if l.antonyms(): 
                        antonyms.append(l.antonyms()[0].name())
        nouns_ant = list(set(antonyms))

        for v in vbg:
            for syn in wordnet.synsets(v): 
                for l in syn.lemmas(): 
                    synonyms.append(l.name())
                    if l.antonyms(): 
                        antonyms.append(l.antonyms()[0].name())
        vbg_ant = list(set(antonyms))
        ###print("nouns_ant: ", nouns_ant)
        ###print("vbg_ant: ", vbg_ant)

        new_aa = dt + nouns_ant + vbg_ant
        random.shuffle(new_aa)
        
        if new_aa == []:
            premises.append(data["premise"])
            hyp.append(data["hypothesis"])
            label.append(data["label"])
        else:
            premises.append(data["premise"])
            ###print("AA: ", new_aa[:5])
            #hyp.append(data["hypothesis"]+ " "+ ' '.join(new_aa[:5]))
            hyp.append(' '.join(new_aa[:5]) + " " + data["hypothesis"])
            label.append(data["label"])
            

    output = {"premise": premises,
            "hypothesis": hyp, 
            "label": label}
    
    return Dataset.from_dict(output)

def addany2_eb_ph(dataset):
    nltk.download('punkt')
    synonyms = [] 
    antonyms = []
    premises, hyp, label = [],[],[] 
    nltk.download('averaged_perceptron_tagger')
    editor = Editor()

    for data in dataset:
        tokens = word_tokenize(data["hypothesis"])

        parts_of_speech = nltk.pos_tag(tokens)
        ###print("-----", "\n")
        ###print(parts_of_speech)
        nouns = list(filter(lambda x: x[1] == "NN", parts_of_speech))
        nouns_list = [n[0] for n in nouns]
        dt = list(filter(lambda x: x[1] == "DT", parts_of_speech))
        dt = [n[0] for n in dt]
        vbg = list(filter(lambda x: x[1] == "VBG", parts_of_speech))
        vbg = [n[0] for n in vbg]

        ###print("nouns_list: ", nouns_list)
        ###print("dt: ", dt)
        ###print("vbg: ", vbg)

        for noun in nouns_list:
            for syn in wordnet.synsets(noun): 
                for l in syn.lemmas(): 
                    synonyms.append(l.name())
                    if l.antonyms(): 
                        antonyms.append(l.antonyms()[0].name())
        nouns_ant = list(set(antonyms))

        for v in vbg:
            for syn in wordnet.synsets(v): 
                for l in syn.lemmas(): 
                    synonyms.append(l.name())
                    if l.antonyms(): 
                        antonyms.append(l.antonyms()[0].name())
        vbg_ant = list(set(antonyms))

        new_aa = dt + nouns_ant + vbg_ant
        new_bb = dt + nouns_ant + vbg_ant
        random.shuffle(new_aa)
        random.shuffle(new_bb)
        
        if new_aa == []:
            premises.append(data["premise"])
            hyp.append(data["hypothesis"])
            label.append(data["label"])
        else:
            premises.append(' '.join(new_aa[:3]) + " " + data["premise"] + " " + ' '.join(new_bb[:3]))
            hyp.append(' '.join(new_aa[:3]) + " " + data["hypothesis"] + " " + ' '.join(new_bb[:3]))
            label.append(data["label"])
            

    output = {"premise": premises,
            "hypothesis": hyp, 
            "label": label}
    
    return Dataset.from_dict(output)

def addanyRandom__eb_ph(dataset):
    nltk.download('punkt')
    synonyms = [] 
    antonyms = []
    premises, hyp, label = [],[],[] 
    nltk.download('averaged_perceptron_tagger')
    editor = Editor()

    for data in dataset:
        tokens = word_tokenize(data["hypothesis"])

        parts_of_speech = nltk.pos_tag(tokens)
        ###print("-----", "\n")
        ###print(parts_of_speech)
        nouns = list(filter(lambda x: x[1] == "NN", parts_of_speech))
        nouns_list = [n[0] for n in nouns]
        dt = list(filter(lambda x: x[1] == "DT", parts_of_speech))
        dt = [n[0] for n in dt]
        vbg = list(filter(lambda x: x[1] == "VBG", parts_of_speech))
        vbg = [n[0] for n in vbg]

        ###print("nouns_list: ", nouns_list)
        ###print("dt: ", dt)
        ###print("vbg: ", vbg)

        for noun in nouns_list:
            for syn in wordnet.synsets(noun): 
                for l in syn.lemmas(): 
                    synonyms.append(l.name())
                    if l.antonyms(): 
                        antonyms.append(l.antonyms()[0].name())
        nouns_ant = list(set(antonyms))

        for v in vbg:
            for syn in wordnet.synsets(v): 
                for l in syn.lemmas(): 
                    synonyms.append(l.name())
                    if l.antonyms(): 
                        antonyms.append(l.antonyms()[0].name())
        vbg_ant = list(set(antonyms))

        new_aa = dt + nouns_ant + vbg_ant
        new_bb = dt + nouns_ant + vbg_ant
        random.shuffle(new_aa)
        random.shuffle(new_bb)
        order = [1,2,3]
        random.shuffle(order)
        if new_aa == []:
            premises.append(data["premise"])
            hyp.append(data["hypothesis"])
            label.append(data["label"])
        else:
            if order[1] == 1:    
                premises.append(data["premise"] + " " + ' '.join(new_bb[:3]))
            elif order[1] == 2:
                premises.append(' '.join(new_aa[:3]) + " " + data["premise"] + " " + ' '.join(new_bb[:3]))
            elif order[1] == 3:
                premises.append(' '.join(new_aa[:3]) + " " + data["premise"])
            random.shuffle(order)
            if order[1] == 1:    
                hyp.append(data["hypothesis"] + " " + ' '.join(new_bb[:3]))
            elif order[1] == 2:
                hyp.append(' '.join(new_aa[:3]) + " " + data["hypothesis"] + " " + ' '.join(new_bb[:3]))
            elif order[1] == 3:
                hyp.append(' '.join(new_aa[:3]) + " " + data["hypothesis"])
            
            label.append(data["label"])
            

    output = {"premise": premises,
            "hypothesis": hyp, 
            "label": label}
    
    return Dataset.from_dict(output)

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    similarity=float(intersection) / union
    distance=1-similarity
    return similarity,distance

def WOB(dataset):
    
    premises, hyp, label =  [], [], []
    id = 0
    number = 0
    for data in dataset:
        id += 1
        d_p = data["premise"]
        d_h = data["hypothesis"]
        sim, dis = jaccard(d_p, d_h)
        if sim >0.25 and (data["label"] == 2 or data["label"] == 1) :
            number += 1
            #print ("Premise: ", data["premise"])
            #print ("Hypothesis: ", data["hypothesis"])
            #print ("Idx: ", id, "# ", number, " Label: ", data["label"], " JS: ", sim)
            premises.append(data["premise"])
            hyp.append(data["hypothesis"])
            label.append(data["label"])            

    output = {"premise": premises,
            "hypothesis": hyp, 
            "label": label}

    return Dataset.from_dict(output)

def CWB(dataset):
    
    premises, hyp, label =  [], [], []
    id = 0
    number = 0
    for data in dataset:
        id += 1
        d_p = data["premise"]
        d_h = data["hypothesis"]
        sim, dis = jaccard(d_p, d_h)
        if sim >0.25 and (data["label"] == 2 or data["label"] == 1) :
            number += 1
            #print ("Premise: ", data["premise"])
            #print ("Hypothesis: ", data["hypothesis"])
            #print ("Idx: ", id, "# ", number, " Label: ", data["label"], " JS: ", sim)
            premises.append(data["premise"])
            hyp.append(data["hypothesis"])
            label.append(data["label"])            

    output = {"premise": premises,
            "hypothesis": hyp, 
            "label": label}

    return Dataset.from_dict(output)
        
        


                    



       
    




