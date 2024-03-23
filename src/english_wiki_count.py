# calculate the number of tokens in english wikipedia

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataset import get_dataset
from gensim.parsing.preprocessing import strip_punctuation, strip_tags, strip_multiple_whitespaces, strip_tags, strip_multiple_whitespaces, strip_punctuation


def preprocess_string(text):
    data = text.lower()
    data = data.replace('„', '').replace('“', '')
    data = strip_tags(data)
    data = strip_punctuation(data)
    data = strip_multiple_whitespaces(data)

    return data.split()


dataloader = get_dataset('wikipedia', language='en')


def preprocess_and_count_batch(batch):
    preprocessed_batch = [len(preprocess_string(item)) for item in batch]
    return sum(preprocessed_batch)


batch_size = 1000
num_tokens = 0

with ProcessPoolExecutor() as executor:
    batches = [dataloader.data[i:i+batch_size]
               for i in range(0, len(dataloader.data), batch_size)]
    futures = [executor.submit(preprocess_and_count_batch, batch)
               for batch in batches]

    for future in tqdm(as_completed(futures), total=len(futures)):
        num_tokens += future.result()

print('Number of tokens in english wikipedia: ', num_tokens)
