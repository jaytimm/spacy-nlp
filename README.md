# Some spaCy & scispacy workflows

*Updated: 2022-12-19*

> An attempt at organizing some `spaCy` workflows – by a lowly R
> programmer. Some functions for disentangling `spaCy` output. For
> working with actual corpora, as opposed to
> `nlp("This is a sentence.")`.

> `spaCy` is beyond groovy! Truly.

------------------------------------------------------------------------

-   [Some spaCy & scispacy workflows](#some-spacy-&-scispacy-workflows)
    -   [Conda environment](#conda-environment)
    -   [PubMed abstracts](#pubmed-abstracts)
    -   [Libraries](#libraries)
        -   [Scispacy components](#scispacy-components)
        -   [Custom sentencizer via `pySBD` &
            `medspacy`](#custom-sentencizer-via-%60pysbd%60-&-%60medspacy%60)
    -   [Spacy annotate](#spacy-annotate)
    -   [Extraction functions](#extraction-functions)
        -   [spacy_get_df](#spacy_get_df)
        -   [spacy_get_entities](#spacy_get_entities)
        -   [spacy_get_abbrevs](#spacy_get_abbrevs)
        -   [spacy_get_nps](#spacy_get_nps)
        -   [spacy_get_hyponyms](#spacy_get_hyponyms)
    -   [Medical transcript data](#medical-transcript-data)
    -   [References](#references)

------------------------------------------------------------------------

## Conda environment

``` bash
conda create -n scispacy python=3.9
source activate scispacy 
conda install transformers
/home/jtimm/anaconda3/envs/scispacy/bin/pip install scispacy
/home/jtimm/anaconda3/envs/scispacy/bin/pip install pysbd
/home/jtimm/anaconda3/envs/scispacy/bin/pip install medspacy

/home/jtimm/anaconda3/envs/scispacy/bin/pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz

## problematic with new spacy -- 
/home/jtimm/anaconda3/envs/scispacy/bin/pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_ner_bc5cdr_md-0.4.0.tar.gz

## /home/jtimm/anaconda3/envs/scispacy/bin/pip install spacy-lookup
## python -m spacy download en_core_web_sm
```

``` r
## <R-console>
Sys.setenv(RETICULATE_PYTHON = "/home/jtimm/anaconda3/envs/scispacy/bin/python")
library(reticulate)
#reticulate::use_python("/home/jtimm/anaconda3/envs/m3demo/bin/python")
reticulate::use_condaenv(condaenv = "scispacy",
                         conda = "/home/jtimm/anaconda3/bin/conda")
```

## PubMed abstracts

``` r
library(dplyr)
dd <- pubmedr::pmed_search_pubmed(search_term = 'drug discovery', 
                                  fields = c('TIAB','MH'))
```

    ## [1] "drug discovery[TIAB] OR drug discovery[MH]: 9999 records"

``` r
dd.df <- pubmedr::pmed_get_records2(pmids = unique(dd$pmid)[1:100], 
                                    with_annotations = F)[[1]] |>
  filter(!is.na(abstract))

df <- reticulate::r_to_py(dd.df)
```

## Libraries

``` python
import pandas as pd
import os
import spacy
import scispacy
#nlp = spacy.load("en_core_sci_sm")
#nlp = spacy.load("en_core_web_sm")
```

### Scispacy components

-   `en_ner_bc5cdr_md`, A spaCy NER model trained on the BC5CDR corpus.
    For disease and chemical NER. Details for additional models
    available [here](https://allenai.github.io/scispacy/).

> Another option is to use the generic scispacy “mention detector”, and
> then link to UMLS, eg.

-   An abbreviation detector.

-   An entity linker – here `umls`, but `mesh`, `rxnorm`, `go`, and
    `hpo` are also available knowledge bases that can be linked to.

-   A hyponym detector.

``` python
nlp = spacy.load("en_ner_bc5cdr_md")
nlp.add_pipe("sentencizer", before = 'ner')
from scispacy.abbreviation import AbbreviationDetector
nlp.add_pipe("abbreviation_detector")

from scispacy.linking import EntityLinker
nlp.add_pipe(
  "scispacy_linker", 
  config={"resolve_abbreviations": True, 
  "linker_name": "umls"})
linker = nlp.get_pipe("scispacy_linker")

from scispacy.hyponym_detector import HyponymDetector
nlp.add_pipe(
  "hyponym_detector", 
  last = True, 
  config={"extended": False})
```

``` python
print(nlp.pipe_names)
```

    ## ['tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer', 'parser', 'sentencizer', 'ner', 'abbreviation_detector', 'scispacy_linker', 'hyponym_detector']

### Custom sentencizer via `pySBD` & `medspacy`

``` python
# Doc.set_extension("pysbd_sentences", getter = pysbd_sentence_boundaries, force=True)
# nlp.add_pipe('pysbd_sentences', before = 'ner')

## https://github.com/medspacy/medspacy/blob/master/medspacy/sentence_splitting.py
import pysbd
from PyRuSH import PyRuSHSentencizer
from spacy.language import Language

@Language.factory("medspacy_pysbd")
class PySBDSenteceSplsitter:
    def __init__(self, name, nlp, clean=False):
        self.name = name
        self.nlp = nlp
        self.seg = pysbd.Segmenter(language="en", clean=clean, char_span=True)

    def __call__(self, doc):
        sents_char_spans = self.seg.segment(doc.text_with_ws)
        start_token_ids = [sent.start for sent in sents_char_spans]
        for token in doc:
            token.is_sent_start = True if token.idx in start_token_ids else False
        return doc
```

## Spacy annotate

``` python
texts = list(r.df['abstract'])
doc = list(nlp.pipe(texts))
```

## Extraction functions

### spacy_get_df

``` python
def extract_df(doc:spacy.tokens.doc.Doc):

    return [
        (
          i.text,
          i.i, 
          i.is_sent_start,
          i.lemma_, 
          i.ent_type_, 
          i.tag_, 
          i.dep_, 
          i.pos_,
          i.is_stop, 
          i.is_alpha, 
          i.is_digit, 
          i.is_punct
          ) for i in doc
    ]
    
#####    
def spacy_get_df(docs):
    
    cols = [
        "doc_id", 
        "token", 
        "token_order", 
        "sent_id",
        "lemma", 
        "ent_type", 
        "tag", 
        "dep", 
        "pos", 
        "is_stop", 
        "is_alpha", 
        "is_digit", 
        "is_punct"
    ]
    
    meta_df = []
    for ix, doc in enumerate(docs):
        meta = extract_df(doc)
        meta = pd.DataFrame(meta)
        meta.columns = cols[1:]
        meta = meta.assign(doc_id = ix).loc[:, cols]
        meta.sent_id = meta.sent_id.astype(bool).cumsum() - 1
        meta_df.append(meta)
        
    return pd.concat(meta_df)   
  
sp_df = spacy_get_df(doc)
```

``` r
reticulate::py$sp_df |>
  slice(1:5) |> knitr::kable()
```

| doc_id | token   | token_order | sent_id | lemma   | ent_type | tag   | dep      | pos   | is_stop | is_alpha | is_digit | is_punct |
|----:|:-----|-------:|-----:|:-----|:-----|:----|:-----|:----|:-----|:-----|:-----|:-----|
|      0 | Chagas  |           0 |       0 | chagas  | DISEASE  | JJ    | compound | ADJ   | FALSE   | TRUE     | FALSE    | FALSE    |
|      0 | disease |           1 |       0 | disease | DISEASE  | NN    | nsubj    | NOUN  | FALSE   | TRUE     | FALSE    | FALSE    |
|      0 | (       |           2 |       0 | (       |          | -LRB- | punct    | PUNCT | FALSE   | FALSE    | FALSE    | TRUE     |
|      0 | CD      |           3 |       0 | cd      |          | NN    | appos    | NOUN  | FALSE   | TRUE     | FALSE    | FALSE    |
|      0 | )       |           4 |       0 | )       |          | -RRB- | punct    | PUNCT | FALSE   | FALSE    | FALSE    | TRUE     |

### spacy_get_entities

``` python
def spacy_get_entities(docs):

    entity_details_dict = {
      "doc_id": [], 
      "sent_id": [], 
      "ent_text": [],
      "ent_label": [], 
      "cui": [], 
      "descriptor": [], 
      "score": [], 
      "ent_start": [], 
      "ent_end": []
      }
    
    for ix, doc in enumerate(docs):
      for sent_i, sent in enumerate(doc.sents):

        for ent in sent.ents:
          entity_details_dict["doc_id"].append(ix)
          entity_details_dict["sent_id"].append(sent_i)
          entity_details_dict["ent_text"].append(ent.text)
          entity_details_dict["ent_label"].append(ent.label_)
          
          if len(ent._.kb_ents) == 0:
            entity_details_dict["cui"].append('')
            entity_details_dict["descriptor"].append('')
            entity_details_dict["score"].append('')
          else:
            score = round(ent._.kb_ents[0][1], 2)
            cui = ent._.kb_ents[0][0]
            descriptor = linker.umls.cui_to_entity[ent._.kb_ents[0][0]][1]
            entity_details_dict["cui"].append(cui)
            entity_details_dict["descriptor"].append(descriptor)
            entity_details_dict["score"].append(score)
            
          entity_details_dict["ent_start"].append(ent.start)
          entity_details_dict["ent_end"].append(ent.end)
          
    dd = pd.DataFrame.from_dict(entity_details_dict)
        
    return dd
  
sp_entities = spacy_get_entities(doc)
```

``` r
reticulate::py$sp_entities |>
  slice(1:5) |> knitr::kable()
```

| doc_id | sent_id | ent_text              | ent_label | cui      | descriptor       | score | ent_start | ent_end |
|-----:|------:|:---------------|:-------|:------|:-----------|:----|-------:|------:|
|      0 |       0 | Chagas disease        | DISEASE   | C0041234 | Chagas Disease   | 1     |         0 |       2 |
|      0 |       0 | protozoan Trypanosoma | DISEASE   | C0033739 | Protozoa         | 0.71  |        12 |      14 |
|      0 |       1 | toxicity              | DISEASE   | C0040539 | Toxicity aspects | 1     |        29 |      30 |
|      0 |       3 | chloro-oximes         | CHEMICAL  |          |                  |       |        64 |      65 |
|      0 |       3 | acetylenes            | CHEMICAL  | C0001052 | acetylene        | 1     |        66 |      67 |

### spacy_get_abbrevs

``` python
def spacy_get_abbrevs(docs):

    details_dict = {
      "doc_id": [], 
      "abrv": [], 
      "start": [], 
      "end": [], 
      "long_form": []
      }
    
    for ix, doc in enumerate(docs):
      
      for ab in doc._.abbreviations:
        details_dict["doc_id"].append(ix)
        details_dict["abrv"].append(ab.text)
        details_dict["start"].append(ab.start)
        details_dict["end"].append(ab.end)
        lf = ' '.join(map(str, ab._.long_form))
        details_dict["long_form"].append(lf)
        
    dd = pd.DataFrame.from_dict(details_dict)
    return dd
  
sp_abbrevs = spacy_get_abbrevs(doc)
```

``` r
reticulate::py$sp_abbrevs |>
  slice(1:10) |> knitr::kable()
```

| doc_id | abrv  | start | end | long_form                |
|-------:|:------|------:|----:|:-------------------------|
|      0 | CD    |     3 |   4 | Chagas disease           |
|      0 | CD    |   197 | 198 | Chagas disease           |
|      0 | CD    |   179 | 180 | Chagas disease           |
|      1 | FEP   |   111 | 112 | free energy perturbation |
|      1 | FEP   |   123 | 124 | free energy perturbation |
|      7 | PLpro |   338 | 339 | Papain like Protease     |
|      7 | PLpro |    51 |  52 | Papain like Protease     |
|      7 | PLpro |   296 | 297 | Papain like Protease     |
|      7 | PLpro |   151 | 152 | Papain like Protease     |
|      7 | PLpro |   101 | 102 | Papain like Protease     |

### spacy_get_nps

``` python
def spacy_get_nps(docs):

    details_dict = {
      "doc_id": [], 
      "sent_id": [],
      "nounc": [], 
      "start": [], 
      "end": []
      }
    
    for ix, doc in enumerate(docs):
      for sent_i, sent in enumerate(doc.sents):
        
        for nc in sent.noun_chunks:
          details_dict["doc_id"].append(ix)
          details_dict["sent_id"].append(sent_i)
          details_dict["nounc"].append(nc.text)
          details_dict["start"].append(nc.start)
          details_dict["end"].append(nc.end)
    
    dd =  pd.DataFrame.from_dict(details_dict)     
    return dd
  
sp_noun_phrases = spacy_get_nps(doc)
```

``` r
reticulate::py$sp_noun_phrases |>
  slice(1:5) |> knitr::kable()
```

| doc_id | sent_id | nounc           | start | end |
|-------:|--------:|:----------------|------:|----:|
|      0 |       0 | Chagas disease  |     0 |   2 |
|      0 |       0 | (CD             |     2 |   4 |
|      0 |       1 | The two drugs   |    16 |  19 |
|      0 |       1 | adverse effects |    25 |  27 |
|      0 |       1 | severe toxicity |    28 |  30 |

### spacy_get_hyponyms

``` python
def spacy_get_hyponyms(docs):

    details_dict = {
      "doc_id": [], 
      "pred": [], 
      "sbj": [], 
      "obj": []
      }
    
    for ix, doc in enumerate(docs):
      
      for ht in doc._.hearst_patterns:
        details_dict["doc_id"].append(ix)
        details_dict["pred"].append(ht[0])
        
        sbj = ' '.join(map(str, ht[1]))
        obj = ' '.join(map(str, ht[2]))
        
        details_dict["sbj"].append(sbj)
        details_dict["obj"].append(obj)
    
    dd =  pd.DataFrame.from_dict(details_dict)     
    return dd
  
sp_hearst = spacy_get_hyponyms(doc)
```

``` r
reticulate::py$sp_hearst |>
  slice(1:10) |> knitr::kable()
```

| doc_id | pred    | sbj              | obj               |
|-------:|:--------|:-----------------|:------------------|
|      2 | include | metabolites      | pigments          |
|      2 | include | metabolites      | enzymes           |
|      2 | include | metabolites      | compounds         |
|      3 | such_as | microbes         | microfungi        |
|      3 | such_as | microbes         | bacteria          |
|      3 | such_as | ailments         | cancer            |
|      3 | such_as | ailments         | malaria           |
|      3 | such_as | ailments         | diseases          |
|      5 | such_as | diseases         | cancer            |
|      6 | include | efficacy effects | lung moisturizing |

## Medical transcript data

> Data from R package `clinspacy` via <https://mtsamples.com/>

``` r
mts <- clinspacy::dataset_mtsamples()
mts_df <- reticulate::r_to_py(mts)
```

``` python
def spacy_get_nps(docs):

    details_dict = {
      "doc_id": [], 
      "sent_id": [],
      "nounc": [], 
      "start": [], 
      "end": []
      }
    
    for ix, doc in enumerate(docs):
      for sent_i, sent in enumerate(doc.sents):
        
        for nc in sent.noun_chunks:
          details_dict["doc_id"].append(ix)
          details_dict["sent_id"].append(sent_i)
          details_dict["nounc"].append(nc.text)
          details_dict["start"].append(nc.start)
          details_dict["end"].append(nc.end)
    
    dd =  pd.DataFrame.from_dict(details_dict)     
    return dd
  
sp_noun_phrases = spacy_get_nps(doc)
```

## References

Eyre, A.B. Chapman, K.S. Peterson, J. Shi, P.R. Alba, M.M. Jones, T.L.
Box, S.L. DuVall, O. V Patterson, Launching into clinical space with
medspaCy: a new clinical text processing toolkit in Python, AMIA Annu.
Symp. Proc. 2021 (in Press. (n.d.). <http://arxiv.org/abs/2106.07799>.

Neumann, M., King, D., Beltagy, I., & Ammar, W. (2019). ScispaCy: fast
and robust models for biomedical natural language processing. arXiv
preprint arXiv:1902.07669.
