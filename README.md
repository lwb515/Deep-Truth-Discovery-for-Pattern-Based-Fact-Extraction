Deep Truth Discovery for Pattern-Based Fact Extraction

input: 

1. corpus
2. pattterns information: patterns and labels
3. trainiing data: the neural network's training data
4. test data: the neural network's test data



input format:

1. data/$corpus_name/corpus.txt: Through the preprocessed corpus in the paper
2. data/$corpus_name/patterns_label.txt:   

 Format: pattern + "\t" + label

Example: $LOCATION capital of $CITY  1

3. data/$corpus_name/training_data.txt

Format: pattern, attribute, entity, value, frequency, entity_type, value_type, label. The separator is a TAB character

Example:$LOCATION capital of $CITY  capital    Kiev   Ukrainian  9  $LOCATION  $LOCATION.CITY 1

4. data/$corpus_name/test_data.txt

Format: pattern, attribute, entity, value, frequency, entity_type, value_type, label. The separator is a TAB character

Example:$LOCATION capital of $CITY  capital    Kiev   Ukrainian  9  $LOCATION  $LOCATION.CITY 1

$corpus_name is your corpus's name




run:
1. adjust para_init.py, select the required parameters
2. run run_word_embedding.py. to train word2vec model.
3. run framework.py



output:

The final results can be found in data/result/$corpus_name/