import json
import logging
import re


def convert_data_to_spacy(data_path):
    try:
        train_data = []
        fr = open(data_path, 'r', encoding='utf-8')
        lines = fr.read()
        # print(lines)
        lines = json.loads(lines)  # to catch json data by its name
        # print(lines['object'])

        for line in lines['object']:
            data = line
            # print(data)
            text = data['content']  # take content as text
            # print(text)
            entities = []
            for annotation in data['annotation']:
                point = annotation['points'][0]
                # print(point)
                labels = annotation['label']  # take labels like Projects Education
                # print(labels)
                if not isinstance(labels, list):
                    labels = [labels]
                    # print(labels)
                for label in labels:
                    entities.append((point['start'], point['end'] + 1, label))  # append them to entities array
            # print(entities)
            train_data.append((text, {"entities": entities}))
            # append all entities to train data array as spacy accept

        # print(train_data)
        return train_data

    except Exception as e:
        logging.exception("Error = " + str(e))
        return None


def trim_entity_spans(data: list) -> list:

    invalid_span_tokens = re.compile(r'\s')
    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        # print(entities)
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            # print(valid_start)
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])

    return cleaned_data


def train():
    data_path = './Data/data.json'
    train_data = convert_data_to_spacy(data_path)
    train_data = trim_entity_spans(train_data)
    # print(train_data)


train()
