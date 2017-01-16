from gensim.models import Word2Vec
import csv
import numpy as np
import random
from nltk.corpus import names
import nltk
import itertools
import time
def gender_features(word):
    return {'suffix1': word[-1:], 'suffix2': word[-2:], 'last_is_vowel' : (word[-1] in 'aeiouy')}

def gender_Classfier():
    labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
    random.shuffle(labeled_names)
    featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
    # test_set = [(gender_features(n), gender) for (n, gender) in devtest_names]
    train_set = featuresets[:]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    # print(nltk.classify.accuracy(classifier, test_set))
    # print(classifier.classify(gender_features('Neo')))
    return classifier


def POSTagging(tokens):
    sentences = [nltk.pos_tag(sent) for sent in tokens]
    # print(sentences)
    return sentences

def Word2VecModel(File, min_count = 1, size = 200, window = 5):
    file = open(File, "r")
    words = []
    sent = []
    for line in file:
        line = line.strip()
        line = line.replace("_"," ")
        if line == ".":
            words.append(sent)
            sent = []
        else:
            if line != "" and line != " ":
                sent.append(line)
    file.close()

    if len(sent) != 0:
        words.append(sent)

    model = Word2Vec(words, size=size, window=window, min_count=min_count)
    return (model,words)

def getMentionFeats(MentionFile, WordsFile, min_count, size, window):

    (model, wordn) = Word2VecModel(WordsFile, min_count, size, window)

    POS_id = {}

    POS_id["Nominal Noun"] = 0
    POS_id["Proper Noun"] = 1
    POS_id["Pronouns"] = 2
    POS_id["Unknown"] = 3

    classifier = gender_Classfier()
    POSTagged_wordsn = POSTagging(wordn)
    # print(POSTagged_words)

    # NER = [nltk.ne_chunk(pts,binary=True) for pts in POSTagged_words]
    # # NER = nltk.ne_chunk( POSTagged_words, binary=True)
    # print(NER)

    POSTagged_words = list(itertools.chain.from_iterable(POSTagged_wordsn))

    unzipped = list(zip(*POSTagged_words))
    # print(unzipped[0])
    WordsList = list(unzipped[0])
    TagsList = list(unzipped[1])

    file = open(MentionFile, "r")
    mentions = []
    mentionFeats = []
    flag = 0
    for line in file:
        line = line.strip()
        line = line.split(" ")
        line = line[0]
        line = line.replace("_"," ")
        # mentions.append(line)
        name = line
        series = []
        for i in range(len(WordsList)):
            # print(WordsList[i:i + len(line)], line)
            if WordsList[i] == line:
                # print(WordsList[i:i + len(line)],line)
                series.append(i)
                break
        # series = [(i, i + len(line)) for i in range(len(WordsList)-len(line)) if WordsList[i:i + len(line)] == line]
        # print(series)
        tags = TagsList[series[0]]
        total = model[line]

        MentionHead = total
        FirstMention = model[line]
        LastMention = model[line]
        PrecWord = model[WordsList[series[0] - 1]]
        FollWord = model[WordsList[series[0]+1]]
        NumberOfWords = len(line)
        MentionSentenceidx = i

        set_tags = set(tags)

        if ("NNP" in set_tags):
            POSTag = POS_id["Proper Noun"]
            MentionNumber = 0  # Singular
            if (classifier.classify(gender_features(name)) == "female"):
                Gender = 1  # Gender is female
            else:
                Gender = 0
        elif ("NNPS" in set_tags):
            POSTag = POS_id["Proper Noun"]
            MentionNumber = 1  # Plural
            Gender = 2  # Unknown Gender
        elif "NN" in set_tags:
            POSTag = POS_id["Nominal Noun"]
            MentionNumber = 0  # Singular
            Gender = 2
        elif "NNS" in set_tags:
            POSTag = POS_id["Nominal Noun"]
            MentionNumber = 1  # Plural
            Gender = 2
        else:
            POSTag = POS_id["Unknown"]
            MentionNumber = 3  # Plural
            Gender = 2

        # To choose features change this line
        total = np.hstack((MentionHead, FirstMention, LastMention, PrecWord, FollWord, NumberOfWords, MentionSentenceidx,
                           POSTag, MentionNumber, Gender))

        if flag == 1:
            mentionFeats = np.vstack((mentionFeats, total))
        else:
            mentionFeats.append(total)
            flag = 1


    file.close()
    return mentionFeats

# Model 2 treats Mention tokens as independent words, tokenize mentions
def Word2VecModel2(File, min_count = 1, size = 200, window = 5):
    file = open(File, "r")
    words = []
    sent = []
    for line in file:
        line = line.strip()
        line = line.replace("_"," ")
        line = line.split(" ")
        if line[0] == ".":
            words.append(sent)
            sent = []
        else:
            for word in line:
                if word != "" and word != " ":
                    sent.append(word)
    file.close()

    if len(sent) != 0:
        words.append(sent)

    model = Word2Vec(words, size=size, window=window, min_count=min_count)
    return (model,words)

def getMentionFeats2(MentionFile, WordsFile, min_count, size, window):

    (model,wordn) = Word2VecModel2(WordsFile, min_count, size, window)

    POS_id = {}

    POS_id["Nominal Noun"] = 0
    POS_id["Proper Noun"] = 1
    POS_id["Pronouns"] = 2
    POS_id["Unknown"] = 3


    NERTestTag = ["" for x in range(9)]
    NERTestTag[0] = "PERSON"
    NERTestTag[1] = "ORGANIZATION"
    NERTestTag[2] = "LOCATION"
    NERTestTag[3] = "DATE"
    NERTestTag[4] = "TIME"
    NERTestTag[5] = "MONEY"
    NERTestTag[6] = "PERCENT"
    NERTestTag[7] = "FACILITY"
    NERTestTag[8] = "GPE"

    classifier = gender_Classfier()
    POSTagged_wordsn = POSTagging(wordn)
    # print(POSTagged_words)

    # POSTagged_wordsn = nltk.corpus.treebank.tagged_sents()

    # TreeNodes = []
    WordsList = []
    TagsList = []
    NERTags = []
    for pts in POSTagged_wordsn:
        Label = "None"
        tree = nltk.ne_chunk(pts, binary=False)
        traverse(tree, Label, WordsList, TagsList, NERTags)
    # print(len(WordsList))
    # print(len(TagsList))
    # print(len(NERTags))
        # tree.draw()

    # POSTagged_words = list(itertools.chain.from_iterable(POSTagged_wordsn))
    #
    # unzipped = list(zip(*POSTagged_words))
    # # print(unzipped[0])
    # WordsList = list(unzipped[0])
    # TagsList = list(unzipped[1])

    file = open(MentionFile, "r")
    mentionFeats = []
    flag = 0
    for line in file:
        line = line.strip()
        line = line.split(" ")
        line = line[0]
        line = line.replace("_"," ")
        name = line
        line = line.split(" ")

        series = []
        for i in range(len(WordsList) - len(line)):
            # print(WordsList[i:i + len(line)], line)
            if WordsList[i:i + len(line)] == line:
                # print(WordsList[i:i + len(line)],line)
                series.append((i, i + len(line)))
                break
        # series = [(i, i + len(line)) for i in range(len(WordsList)-len(line)) if WordsList[i:i + len(line)] == line]
        # print(series)
        tags = TagsList[series[0][0]:series[0][1]]
        NERtag = NERTags[series[0][0]:series[0][1]]

        file2 = open("MentionListProgramGenerated.txt", "w")


        prevTag = "None"
        for id in range(len(NERTags)):
            if NERTags[id] != "S":
                if prevTag != NERTags[id]:
                    file2.write("\n")
                    file2.write(NERTags[id]+ " " + WordsList[id])
                    prevTag = NERTags[id]
                else:
                    file2.write(" " + WordsList[id])
            else:
                prevTag = "None"
        file2.close()

        total = np.zeros(size)
        flag = 0
        for word in line:
            total += model[word]
        total =  total/len(line)

        MentionHead = total
        FirstMention = model[line[0]]
        LastMention = model[line[-1]]
        PrecWord = model[WordsList[series[0][0]-1]]
        FollWord = model[WordsList[series[0][1]]]
        NumberOfWords = len(line)
        MentionSentenceidx = i

        set_tags = set(tags)

        if (len(set_tags) == 1 and "NNP" in set_tags):
            POSTag =  POS_id["Proper Noun"]
            MentionNumber = 0 # Singular
            if (classifier.classify(gender_features(name)) == "female"):
                Gender = 1 # Gender is female
            else:
                Gender = 0
        elif  (len(set_tags) == 1 and "NNPS" in set_tags):
            POSTag = POS_id["Proper Noun"]
            MentionNumber = 1  # Plural
            Gender = 2 # Unknown Gender
        elif "NN" in set_tags:
            POSTag = POS_id["Nominal Noun"]
            MentionNumber = 0  # Singular
            Gender = 2
        elif "NNS" in set_tags:
            POSTag = POS_id["Nominal Noun"]
            MentionNumber = 1  # Plural
            Gender = 2
        elif (len(set_tags) != 1 and "NNP" in set_tags):
            POSTag = POS_id["Proper Noun"]
            MentionNumber = 0  # Plural
            Gender = 2  # Unknown Gender
        elif (len(set_tags) != 1 and "NNPS" in set_tags):
            POSTag = POS_id["Proper Noun"]
            MentionNumber = 1  # Plural
            Gender = 2  # Unknown Gender
        else:
            POSTag = POS_id["Unknown"]
            MentionNumber = 3  # Plural
            Gender = 2

        NERtagset = set(NERtag)

        NERbits = np.linspace(0,0,num=9)
        for id in range(len(NERbits)):
            if NERTestTag[id] in NERtagset:
                NERbits[id] = 1
                # print(NERTestTag[id])
        # print(NERbits)
        # To choose features change this line
        total = np.hstack((MentionHead,FirstMention,LastMention,PrecWord,FollWord,NumberOfWords,MentionSentenceidx,POSTag, MentionNumber, Gender, NERbits ))

        if flag == 1:
            mentionFeats = np.vstack((mentionFeats,total))
        else:
            mentionFeats.append(total)
            flag = 1
    file.close()
    return mentionFeats

def getComplexPairFeats(idx,mentionFeats,size):
    PairwiseFeats = np.zeros((idx, size))
    siz = len(mentionFeats[idx])

    dist = np.zeros((1,size))
    BasicMentionFeats = np.zeros((1,siz))
    BasicAnteFeats = np.zeros((1,siz))
    MentionDiff = np.zeros((1,1))
    StringMatch = np.zeros((1,1))
    # print(np.shape(dist), np.shape(BasicMentionFeats), np.shape(BasicAnteFeats), np.shape(MentionDiff))
    temp = np.hstack((dist, BasicMentionFeats, BasicAnteFeats, MentionDiff, StringMatch))
    ComplexPairWiseFeats = np.zeros((idx,np.shape(temp)[1]))
    flag = 0
    for pidx in range(0,idx):
        BasicMentionFeats = mentionFeats[idx].reshape((1, siz))
        feat1 = mentionFeats[idx][0:size]
        feat2 = mentionFeats[pidx][0:size]
        dist = feat1 - feat2
        PairwiseFeats[pidx] = dist
        BasicAnteFeats =  mentionFeats[pidx]
        MentionDiff[0] = abs(idx-pidx)
        bool = np.array_equal(mentionFeats[idx][0:size],mentionFeats[idx][0:size])
        StringMatch[0] = int(bool)
        temp = np.hstack((dist.reshape((1,size)),BasicMentionFeats,BasicAnteFeats.reshape((1,siz)),MentionDiff, StringMatch ))
        ComplexPairWiseFeats[pidx] = temp
    return ComplexPairWiseFeats

# def getComplexPairFeats(idx,mentionFeats,size):
#     PairwiseFeats = np.zeros((idx, size))
#     siz = len(mentionFeats[idx])
#
#     dist = np.zeros((1,size))
#     BasicMentionFeats = mentionFeats[idx].reshape((1,siz))
#     BasicAnteFeats = np.zeros((1,siz))
#     MentionDiff = np.zeros((1,1))
#     StringMatch = np.zeros((1,1))
#     # print(np.shape(dist), np.shape(BasicMentionFeats), np.shape(BasicAnteFeats), np.shape(MentionDiff))
#     ComplexPairWiseFeats = np.hstack((dist, BasicMentionFeats, BasicAnteFeats, MentionDiff, StringMatch))
#     flag = 0
#     for pidx in range(0,idx):
#         feat1 = mentionFeats[idx][0:size]
#         feat2 = mentionFeats[pidx][0:size]
#         dist = feat1 - feat2
#         PairwiseFeats[pidx] = dist
#         BasicAnteFeats =  mentionFeats[pidx]
#         MentionDiff[0] = abs(idx-pidx)
#         bool = np.array_equal(mentionFeats[idx][0:size],mentionFeats[idx][0:size])
#         StringMatch[0] = int(bool)
#         if flag == 1:
#            temp = np.hstack((dist.reshape((1,size)),BasicMentionFeats,BasicAnteFeats.reshape((1,siz)),MentionDiff, StringMatch ))
#            ComplexPairWiseFeats = np.vstack((ComplexPairWiseFeats,temp))
#         else:
#             ComplexPairWiseFeats = np.hstack((dist.reshape((1,size)),BasicMentionFeats,BasicAnteFeats.reshape((1,siz)),MentionDiff, StringMatch))
#             flag = 1
#         # print(dist)
#     return ComplexPairWiseFeats

def getPairFeats(idx,mentionFeats,size):
    PairwiseFeats = np.zeros((idx, size))
    for pidx in range(0,idx):
        feat1 = mentionFeats[idx]
        feat2 = mentionFeats[pidx]
        dist = feat1 - feat2
        PairwiseFeats[pidx] = dist
        # print(dist)
    return PairwiseFeats

def getClustersArrayForMentions(mentionfile):
    f = open(mentionfile, 'r')
    csvreader = csv.reader(f, delimiter=' ')
    mylist = []
    for row in csvreader:
        mylist.append(int(row[1]))
    myarray = np.array(mylist)
    return myarray


def traverse(t,  Label, WordsList = [], TagsList = [], NERTags = [], *args):
    try:
         t.label()
    except AttributeError:
        listing = list(t)
        WordsList.append(listing[0])
        TagsList.append(listing[1])
        NERTags.append(Label)
    else:
        # Now we know that t.node is defined
        # Node = []
        NLabel = str(t.label())
        # Node.append(Label)
        for child in t:
            traverse(child, NLabel, WordsList, TagsList, NERTags)
        # TreeNodes.append(TreeNodes)

if __name__ == '__main__':

    # sent = nltk.corpus.treebank.tagged_sents()
    # # print(sent)
    # NER = [nltk.ne_chunk(pts) for pts in sent]
    # print(NER)

    count = 1
    size = 200
    window = 5

    MentionFeats = getMentionFeats2("mentionsList1.txt","wordsList1.txt",count,size,window)
    print(np.shape(MentionFeats))

    for idx in range(np.shape(MentionFeats)[0]):
        start = time.time()
        ComplexPairWiseFeats = getComplexPairFeats(idx,MentionFeats,size)
        end = time.time()
        print(idx, end - start)
        # print(np.shape(ComplexPairWiseFeats))

