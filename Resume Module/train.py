import spacy

text_file = './Texts'


def train_spacy():
    # train_data = convert_data_into_spacy(text_file+'/out.json')
    nlp = spacy.blank('en')
    if 'ner' not in nlp.pore_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)


def train_practice():
    nlp = spacy.load('en_core_web_sm')

    doc = nlp('I do not have money to pay my credit card account. What is the process for open a new savings account.')
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)

    train = [
        ('Money transfer from my account is not working', {'entities': [(6, 13, 'ACTIVITY'), (23, 29, 'PRODUCT')]}),
        ('I want to check my balance in my savings account', {'entities': [(16, 23, 'ACTIVITY'), (30, 45, 'PRODUCT')]}),
        ('I suspect a fraud in my credit card account', {'entities': [(12, 17, 'ACTIVITY'), (24, 35, 'PRODUCT')]}),
        ('I am here for opening a new account', {'entities': [(14, 21, 'ACTIVITY'), (28, 43, 'PRODUCT')]}),
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

    # print(nlp.pipe_names)  # ['tagger', 'parser', 'ner']
    ner = nlp.get_pipe('ner')
    # print(ner)

    # print(train)

    for _, annotations in train:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
            # print(ner.add_label(ent[2]))
            # print(ent[2])

    disable_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

    import random
    from spacy.util import minibatch, compounding
    from pathlib import Path

    with nlp.disable_pipes(*disable_pipes):
        optimizer = nlp.resume_training()

        for iteration in range(100):
            random.shuffle(train)
            losses = {}

            batches = minibatch(train, size=compounding(1.0, 4.0, 1.001))
            for batch in batches:
                text, annotation = zip(*batch)
                nlp.update(
                    text,
                    annotation,
                    drop=0.5,
                    losses=losses,
                    sgd=optimizer
                )
                print('Losses', losses)

    for text, _ in train:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])

    doc = nlp('I do not have money to pay my credit card account. What is the process for open a new savings account.')
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)


train_practice()
