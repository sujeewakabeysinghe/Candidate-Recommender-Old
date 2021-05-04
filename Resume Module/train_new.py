import spacy
from spacy.util import minibatch, compounding

train = [
    ('I suspect a fraud in my credit card account', {'entities': [(12, 17, 'ACTIVITY'), (24, 35, 'PRODUCT')]}),
    ('Your mortgage is in delinquent status', {'entities': [(20, 30, 'ACTIVITY'), (5, 13, 'PRODUCT')]}),
    ('Your credit card is in past due status', {'entities': [(23, 31, 'ACTIVITY'), (5, 16, 'PRODUCT')]}),
    ('My loan account is still not approved and funded', {'entities': [(25, 37, 'ACTIVITY'), (3, 15, 'PRODUCT')]}),
    ('How do I open a new load account', {'entities': [(9, 13, 'ACTIVITY'), (20, 32, 'PRODUCT')]}),
    ('What are the charges on Investment account', {'entities': [(13, 20, 'ACTIVITY'), (24, 42, 'PRODUCT')]}),
    ('Can you explain late charges on my credit card', {'entities': [(21, 28, 'ACTIVITY'), (35, 46, 'PRODUCT')]}),
    ('I want to open a new loan account', {'entities': [(10, 14, 'ACTIVITY'), (21, 33, 'PRODUCT')]}),
    ('Can you help updating payment on my credit card', {'entities': [(22, 29, 'ACTIVITY'), (36, 47, 'PRODUCT')]}),
    ('When is the payment due on my card', {'entities': [(12, 19, 'ACTIVITY'), (35, 39, 'PRODUCT')]})
]

def convert_datatrucks():
    try:
        lines=[]



def train_spacy():

    nlp = spacy.blank('en')  # loading blank english model
    print(nlp.pipe_names)
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

    print(nlp.pipe_names)

    for _, annotations in train:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])  # here ent[2] is the third element of (6, 13, 'ACTIVITY')
            # print(ent[0])
            # print(ent[1])
            # print(ent[2])

    disable_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*disable_pipes):
        optimizer = nlp.begin_training()

        for iteration in range(10):
            print('Iteration '+str(iteration))
            losses = {}

            batches = minibatch(train, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                text, annotation = zip(*batch)
                nlp.update(
                    text,
                    annotation,
                    drop=0.2,
                    losses=losses,
                    sgd=optimizer
                )
                print('Losses', losses)

    doc = nlp('I do not have money to pay my credit card account. What is the process for open a new savings account.')
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)

    nlp.to_disk("model/")


train_spacy()
