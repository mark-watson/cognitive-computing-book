input_text = "The school teacher suggested a new book to her students"

label = `echo #{input_text} | fasttext predict model.bin -`

# skip the first characters "__label__" when showing result:
puts label[9..-1]
