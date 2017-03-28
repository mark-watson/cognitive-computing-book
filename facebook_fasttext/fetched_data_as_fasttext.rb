# 4/24/2013: also write out 2grams.  run with jruby --1.8 key_words_for_tag.rb

# 8/2/2010: big change over version in my KB_data_collection directoy tools: do not stem

##require 'stemmer'
require 'pp'

def generate_fasttext
  topic_name_hash = {}
  topic_name_hash_2gram = {}
  File.open("generated/fasttext_training.txt", "w") {|fout|
    Dir.entries("./fetched_data").each {|fn|
        if fn[0..0]!='.'
            puts "Processing: #{fn}"
            #index4 = nil           ## don't collapse detailed categories
            index4 = fn.index("_")  ## collapse detailed categories
            category = fn[0...-4]
            category = fn[0...index4] if index4              
            text = ""
            File.open("./fetched_data/" + fn) { |f|  text = f.read }
            words = text.downcase.gsub(/[^a-z ]/, ' ').split(' ')
            stop = words.size - 20
            index = 0
            while index < stop
              t2 = words[index..index+20].join(' ')
              fout.puts "__label__#{category} #{t2}"
              index += 20
            end
        end
    }
  }
end

##pp words = "President Bush Went to Congress, running all the way.".split(/[^a-zA-Z]/)

generate_fasttext()


