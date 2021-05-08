import spacy
from spacy.util import minibatch, compounding


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