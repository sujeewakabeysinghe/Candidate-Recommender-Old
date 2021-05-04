import spacy
import json
from pdfToText import pdf_to_text_convert


image_dir = './Images/'
pdf_file = './CV/Sujeewa_Abeysinghe.pdf'
model_path = './Model/'
text_file = './Texts'


def generate_info_file():
    # pdf_to_text_convert(pdf_file, image_dir)
    nlp = spacy.load('model/')
    print(nlp)

    fr = open(text_file+'/text.txt', 'r', encoding='utf-8')
    text = fr.read()
    # print(text)

    fw = open(text_file+'/text_sum.txt', 'w', encoding='utf-8')
    doc = nlp(text)
    d = {}  # declare a data dic
    for ent in doc.ents:
        d[ent.label_] = []
    for ent in doc.ents:
        d[ent.label_].append(ent.text)
    # print(d)
    # print(d.keys())

    for i in set(d.keys()):
        fw.write('\n\n')
        fw.write(i+':'+'\n')
        for j in set(d[i]):
            fw.write(j.replace('\n', '')+'\n')
    fr.close()
    fw.close()

    data = {}
    for i in set(d.keys()):
        data[i] = []
    # print(data)

    for i in set(d.keys()):
        for j in set(d[i]):
            data[i].append(j)
    # print(data)

    entity_list = ['Name', 'Designation', 'Skills', 'DATE', 'Educations', 'Experience', 'PERSON']
    extracted_list = list(data.keys())
    print(entity_list)
    print(extracted_list)
    diff = list(set(entity_list) - set(extracted_list))
    print(diff)

    if diff != []:
        for i in diff:
            data[i] = None
            print(data[i])

    with open(text_file + "/out.json", 'w') as outfile:
        json.dump(data, outfile, indent=4)

    doc = nlp('I do not have money in my credit card account. What is the process for open a new savings account.')
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)

    return "success"


generate_info_file()
