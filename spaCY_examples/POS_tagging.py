import spacy
nlp = spacy.load('en')
doc = nlp(u'John went to the bank to get some money. The fisherman sat on the bank of the river. The pilot wanted to bank the airplane to the left after banking to the right.')
for word in doc:
    print("word: {}\tlemma: {}\ttag: {}\tpos: {}".format(
            word.text, word.lemma_, word.tag_, word.pos_))
