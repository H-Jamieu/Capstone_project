import pandas as pd

mov_dir = 'Data/Movie_data/'

#metadata = pd.read_csv(mov_dir+'full_metadata.csv')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

tokens = tokenizer('Sample test text')
print(type(tokenizer))
print(tokenizer)
outputs = model(**tokens)

print(type(outputs))
print(outputs)