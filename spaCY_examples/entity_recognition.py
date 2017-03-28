import spacy
nlp = spacy.load('en')
doc = nlp(u'President Trump has a dispute with Mexico over immigration. IBM and Apple are cooperating on marketing in 2014. Pepsi and Coke sell well to Asian and South American customers. He bought a Ford Escort for $20,000 and drove to the lake for the weekend. The parade was last Saturday.')
for entity in doc.ents:
    print("label: {}\tlabel_: {}\ttext: {}".format(
            entity.label,entity.label_,entity.text))
