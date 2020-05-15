import pandas as pd
import pickle
import json
from io import StringIO
def prediction(Word):
    with open('document_classification_model/TDF_final', 'rb') as tfvector:
            tfidf = pickle.load(tfvector)

    OCR=StringIO(Word)
    OCRData = pd.read_csv(OCR)

    featureSet= tfidf.transform(OCRData)

    features=featureSet.toarray()

    with open('document_classification_model/mappingDictionary', 'rb') as training_model:
        mapper = pickle.load(training_model)


    with open('document_classification_model/text_classifier', 'rb') as training_model:
        model = pickle.load(training_model)

    y_pred = model.predict(features)
    my_list=[]
    for val in y_pred:
       my_list.append( ({"prediction":mapper[val]}))

    json_val= json.dumps(my_list)

    parsed = json.loads(json_val)

    return json.dumps(parsed, indent=2, sort_keys=True)

#print (prediction("25c57acdf805 b7a0f56f6ce8 377a21b394dc 6c8642055a4e 8329bcfb80ba 5ee06767bc0f af671fbeb212 f7ae6f8257da 4129ea7e3fb2 e943e5e5b779 87ecf9ab40b5 a783fb82e12f 1bc29dc7f887 37b722b32dc6 ee94f34a89db d38820625542 9f11111004ec 56a0a522a4dd c337a85b8ef9 6b1c8f75a7e2 0562c756a2f2 0969e9a2a900 1b6d0614f2c7 6d1fb90988cf bcab85963da3 e8ab9151702a cde4f1b2a877 abca9d18fae2 7147fd962807 6bf9c0cb01b4 0562c756a2f2 b99d60f882ce ddbf90c52f6b abca9d18fae2 740c96d384cd b9699ce57810 7ec02e30a5b3 b6c7895c30c1 b4c4f940f774 034e2d7f187e 21ab107e9310 ce1f034abb5d d64c6fcaddb7 ff95624c3dde f0b552c6c11e 14571c343a32 77b580b44942 aa1ef5f5355f 2985f1045d62 e72e96dee26c fbb5efbcc5b3 ea51fa83c91c 73801426ea65 586242498a88 d38820625542 93790ade6682 90b868693b82 aa81e0db3e90 4357c81e10c1 14571c343a32 a31962fbd5f3 6b304aabdcee fad925913b3b 9c0ce8db3cf4 c9a53ea6e219 29ed2b57357b 094453b4e4ae 1015893e384a 586242498a88 e943e5e5b779 b208ae1e8232 9e5f2ff78f35 a86f2ba617ec 29ed2b57357b e943e5e5b779 2d29a9d62e31 b136f6349cf3 4ad52689d690 6df520735456 c337a85b8ef9 b9699ce57810 f0666bdbc8a5 586242498a88 2ef7c27a5df4 56ec29092ae3 1015893e384a 036087ac04f9 586242498a88 19e9f3592995 db0c88fd2e84 fe64d3cdfe5b eeb86a6a04e4 98d0d51b397c 6bf9c0cb01b4 17bd1f49c9a0 c337a85b8ef9 b9699ce57810 ed5d3a65ee2d 8d21095e8690 fe64d3cdfe5b 9b0357d1d3db 564aaf0c408b 26f768da5068 edbab5e0b042 286b99ff15dd 15c0177ee7e9 0af03ed987ad 6ca2dd348663 ee94f34a89db f7ae6f8257da cdee33987473 25c57acdf805"
#))

# print (prediction("1bc29dc7f887"))