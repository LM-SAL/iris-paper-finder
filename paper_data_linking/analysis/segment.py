import logging

import spacy
import tiktoken
from tqdm import tqdm

import paper_data_linking.settings
import paper_data_linking.data.download_text as cl
from paper_data_linking.enums import Locations

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

tqdm.pandas()


def paragraphs(document):
    start = 0
    for token in document:
        if token.is_space and token.text.count("\n") > 1:
            yield document[start:token.i]
            start = token.i
    yield document[start:]


def combine(pgs, model="gpt-3.5-turbo"):
    # This function assumes that no paragraphs are longer than max_tokens.
    encoding = tiktoken.encoding_for_model(model)
    # ^ this is only for openAI, need another tokenizer for other models.
    max_tokens_dict = {
        # add more models here.
        "text-davinci-003": 2049,
        "gpt-3.5-turbo": 4097,
    }
    prompt_and_answer_length = 1000
    max_tokens = max_tokens_dict[model] - prompt_and_answer_length
    n_segment = 0
    segment_groups = [[]]
    for i, p in enumerate(pgs):
        n = len(encoding.encode(str(p)))
        if n_segment + n > max_tokens:
            n_segment = 0
            segment_groups.append([])
        else:
            n_segment = n_segment + n
            segment_groups[-1].append(i)
    return segment_groups


def segment_paper(doc):
    pgs = list(paragraphs(doc))
    segment_groups = combine(pgs)
    segments = []
    for sg in segment_groups:
        segment = "\n\n".join([str(pgs[i]) for i in sg])
        segments.append(segment)
    return segments


def main():
    df = cl.read_metadata(only_retry=False)
    sdf = df
    texts = sdf['pdf_text']
    docs = list(NLP.pipe(texts))
    sdf['docs'] = docs

    sdf["segments"] = sdf["docs"].progress_apply(segment_paper)
    segment_df = sdf.explode("segments")
    segment_df.to_csv(Locations.SEGMENT_DF.value)


# What to do with very short paragraphs? Remove or combine?

if __name__ == "__main__":
    main()
