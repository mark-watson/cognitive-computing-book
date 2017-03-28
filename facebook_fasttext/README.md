# Install Facebook's fasttext

research paper: "Bag of Tricks for Efficient Text Classification" https://arxiv.org/pdf/1607.01759v3.pdf

github repo: https://github.com/facebookresearch/fastText

Build system (no dependencies on my macOS laptop: built with 'make'). Make sure 'fasttext' is on PATH

# using my kbsportal training data for classification

fasttext supervised -input fasttext_training.txt -output model
fasttext predict model.bin fasttext_training.txt  1
fasttext predict model.bin fasttext_testing.txt
fasttext predict model.bin fasttext_testing.txt 5

# piping data for prediction (would be good for use from Ruby)

echo "The chemistry teacher showed a reaction in the lab" | ./fasttext predict model.bin -

# future work: creating a Ruby gem

Use this reference https://www.amberbit.com/blog/2014/6/12/calling-c-cpp-from-ruby/ to learn how to use Rice
