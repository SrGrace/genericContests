from docx.api import Document
import spacy

# doc = Document('data/Series_95.docx')

# para = doc.paragraphs
# for p in para:
#     print(p.text)

# tables = doc.tables
#
# for table in tables:
#     for row in table.rows:
#         print(row.cells)
#         for cell in row.cells:
#             for para in cell.paragraphs:
#                 print(para.text)

nlp = spacy.load('en_core_web_sm')

document = Document('./data/Sample-Rental-Agreement.docx')

tables = document.tables
paragraphs = document.paragraphs

para_dic = {}
counter = 0
for para in paragraphs:
    counter += 1
    doc = nlp(para.text)

    tokens = []
    for token in doc:
        tokens.append({'text': token.text, 'lemma': token.lemma_, 'pos': token.pos_, 'tag': token.tag_,
                       'dep': token.dep_, 'shape': token.shape_, 'alpha': token.is_alpha, 'is_stop': token.is_stop})
    ner = []
    for ent in doc.ents:
        ner.append([ent.text, ent.label_])

    if para.text != '':
        para_dic[counter] = {'text': para.text, 'ner': ner, 'tokens': tokens}


table_dic = {}
counter = 0
for table in tables:
    data = []
    counter += 1
    keys = None
    for i, row in enumerate(table.rows):
        text = (cell.text for cell in row.cells)
        if i == 0:
            keys = tuple(text)
            continue
        doc = nlp(text)
        tokens = []
        for token in doc:
            tokens.append({'text': token.text, 'lemma': token.lemma_, 'pos': token.pos_, 'tag': token.tag_,
                           'dep': token.dep_, 'shape': token.shape_, 'alpha': token.is_alpha, 'is_stop': token.is_stop})
        ner = []
        for ent in doc.ents:
            ner.append([ent.text, ent.label_])
        row_data = dict(zip(keys, text))
        data.append(row_data)
        table_dic[counter] = {'row': data, 'ner': ner, 'tokens': tokens}



