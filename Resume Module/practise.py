import spacy
import  json

text_file = './Texts'


def practise():
    nlp = spacy.load('en_core_web_sm')

    fr = open(text_file+'/text1.txt', "r", encoding='utf-8')
    text = fr.read().replace('\n\n', ' ').replace('\n', ' ')
    print(text)

    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            print(ent.text, ent.label_)


practise()


nlp = spacy.load('en_core_web_sm')
fr = open(text_file+'/text.txt', "r", encoding='utf-8')
text = fr.read()
doc = nlp(text)
ents = list(doc.ents)
people = []
for ent in ents:
    if ent.label_ == 'ORG':
        people.append(ent)
        people.append(ent.label_)
    if ent.label_ == 'PERSON':
        people.append(ent)
        people.append(ent.label_)
print(people)

for token in doc:
    if token.pos_ == 'NOUNS':
        people.append(token)

print(people)

all_in = []
for ent in ents:
    all_in.append(ent)
# print(all_in)

print(list(doc.noun_chunks))