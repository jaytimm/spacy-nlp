# Some spaCy & scispacy workflows

*Updated: 2022-12-20*

> An attempt at organizing some `spaCy` workflows. Some functions for
> disentangling `spaCy` output. For working with actual corpora, as
> opposed to `nlp("This is a sentence.")`.

------------------------------------------------------------------------

-   [Some spaCy & scispacy workflows](#some-spacy-&-scispacy-workflows)
    -   [Conda environment](#conda-environment)
    -   [PubMed abstracts](#pubmed-abstracts)
    -   [Libraries](#libraries)
        -   [Scispacy components](#scispacy-components)
    -   [Spacy annotate](#spacy-annotate)
    -   [Extraction functions](#extraction-functions)
        -   [Standard annotation](#standard-annotation)
        -   [Entities & linking](#entities-&-linking)
        -   [Abbreviations](#abbreviations)
        -   [Noun phrases](#noun-phrases)
        -   [Hyponyms](#hyponyms)
        -   [Negation](#negation)
        -   [Sentences](#sentences)
    -   [Medical transcript data](#medical-transcript-data)
        -   [medspacy](#medspacy)
    -   [References](#references)

------------------------------------------------------------------------

## Conda environment

``` bash
conda create -n scispacy python=3.9
source activate scispacy 
conda install transformers

cd /home/jtimm/anaconda3/envs/scispacy/bin/
pip install scispacy
pip install pysbd
pip install medspacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_ner_bc5cdr_md-0.4.0.tar.gz
```

``` r
## <R-console>
library(dplyr)
Sys.setenv(RETICULATE_PYTHON = "/home/jtimm/anaconda3/envs/scispacy/bin/python")
library(reticulate)
#reticulate::use_python("/home/jtimm/anaconda3/envs/m3demo/bin/python")
reticulate::use_condaenv(condaenv = "scispacy",
                         conda = "/home/jtimm/anaconda3/bin/conda")
```

## PubMed abstracts

``` r
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
nlp.add_pipe("sentencizer", first=True)
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

    ## ['sentencizer', 'tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer', 'parser', 'ner', 'abbreviation_detector', 'scispacy_linker', 'hyponym_detector']

## Spacy annotate

``` python
texts = list(r.df['abstract'])
doc = list(nlp.pipe(texts))
```

## Extraction functions

### Standard annotation

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

| doc_id | token     | token_order | sent_id | lemma    | ent_type | tag | dep       | pos   | is_stop | is_alpha | is_digit | is_punct |
|----:|:------|-------:|-----:|:-----|:-----|:---|:------|:----|:-----|:-----|:-----|:-----|
|      0 | Paxlovid  |           0 |       0 | Paxlovid |          | NNP | nsubjpass | PROPN | FALSE   | TRUE     | FALSE    | FALSE    |
|      0 | ,         |           1 |       0 | ,        |          | ,   | punct     | PUNCT | FALSE   | FALSE    | FALSE    | TRUE     |
|      0 | a         |           2 |       0 | a        |          | DT  | det       | DET   | TRUE    | TRUE     | FALSE    | FALSE    |
|      0 | drug      |           3 |       0 | drug     |          | NN  | appos     | NOUN  | FALSE   | TRUE     | FALSE    | FALSE    |
|      0 | combining |           4 |       0 | combine  |          | VBG | acl       | VERB  | FALSE   | TRUE     | FALSE    | FALSE    |

### Entities & linking

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

| doc_id | sent_id | ent_text           | ent_label | cui      | descriptor | score | ent_start | ent_end |
|-----:|------:|:--------------|:--------|:-------|:--------|:-----|--------:|------:|
|      0 |       0 | nirmatrelvir       | CHEMICAL  |          |            |       |         5 |       6 |
|      0 |       0 | ritonavir          | CHEMICAL  | C0292818 | ritonavir  | 1     |         7 |       8 |
|      0 |       0 | COVID-19           | CHEMICAL  | C5203670 | COVID-19   | 1     |        15 |      16 |
|      0 |       0 | COVID-19 infection | DISEASE   |          |            |       |        34 |      36 |
|      0 |       1 | nirmatrelvir       | CHEMICAL  |          |            |       |        48 |      49 |

### Abbreviations

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

| doc_id | abrv    | start | end | long_form                                     |
|-------:|:--------|------:|----:|:----------------------------------------------|
|      1 | NTD     |     8 |   9 | neglected tropical disease                    |
|      1 | LiMetRS |   100 | 101 | Leishmania infantum methionyl-tRNA synthetase |
|      1 | LiMetRS |    58 |  59 | Leishmania infantum methionyl-tRNA synthetase |
|      1 | LiMetRS |   251 | 252 | Leishmania infantum methionyl-tRNA synthetase |
|      5 | GPCRs   |   109 | 110 | G protein-coupled receptors                   |
|      5 | GPCRs   |   134 | 135 | G protein-coupled receptors                   |
|      5 | GPCRs   |     4 |   5 | G protein-coupled receptors                   |
|      5 | GPCRs   |   149 | 150 | G protein-coupled receptors                   |
|      5 | GPCRs   |    60 |  61 | G protein-coupled receptors                   |
|      5 | GPCRs   |   311 | 312 | G protein-coupled receptors                   |

### Noun phrases

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

| doc_id | sent_id | nounc        | start | end |
|-------:|--------:|:-------------|------:|----:|
|      0 |       0 | Paxlovid     |     0 |   1 |
|      0 |       0 | a drug       |     2 |   4 |
|      0 |       0 | nirmatrelvir |     5 |   6 |
|      0 |       0 | ritonavir    |     7 |   8 |
|      0 |       0 | the impact   |    31 |  33 |

### Hyponyms

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

| doc_id | pred       | sbj           | obj                         |
|-------:|:-----------|:--------------|:----------------------------|
|      4 | such_as    | immunity      | HIV/AIDS patients           |
|      4 | such_as    | immunity      | organ transplant recipients |
|      4 | include    | compounds     | tyrosine kinase inhibitors  |
|      5 | include    | invertebrates | diuresis                    |
|      5 | include    | invertebrates | feeding                     |
|      5 | include    | invertebrates | digestion                   |
|      5 | especially | metazoans     | humans                      |
|      9 | include    | metabolites   | pigments                    |
|      9 | include    | metabolites   | enzymes                     |
|      9 | include    | metabolites   | compounds                   |

### Negation

### Sentences

``` python
def spacy_get_sentences(docs):

    details_dict = {
      "doc_id": [], 
      "sent_id": [],
      "text": []
      }
    
    for ix, doc in enumerate(docs):
      for sent_i, sent in enumerate(doc.sents):
        details_dict["doc_id"].append(ix)
        details_dict["sent_id"].append(sent_i)
        sentences = str(sent).strip()
        details_dict["text"].append(sentences)
    
    dd =  pd.DataFrame.from_dict(details_dict)     
    return dd
  
sp_sentences = spacy_get_sentences(doc)
```

``` r
reticulate::py$sp_sentences |>
  slice(1:5) |> knitr::kable()
```

| doc_id | sent_id | text                                                                                                                                                                                                                                                      |
|--:|---:|:-----------------------------------------------------------------|
|      0 |       0 | Paxlovid, a drug combining nirmatrelvir and ritonavir, was designed for the treatment of COVID-19 and its rapid development has led to emergency use approval by the FDA to reduce the impact of COVID-19 infection on patients.                          |
|      0 |       1 | In order to overcome potentially suboptimal therapeutic exposures, nirmatrelvir is dosed in combination with ritonavir to boost the pharmacokinetics of the active product.                                                                               |
|      0 |       2 | Here we consider examples of drugs co-administered with pharmacoenhancers.                                                                                                                                                                                |
|      0 |       3 | Pharmacoenhancers have been adopted for multiple purposes such as ensuring therapeutic exposure of the active product, reducing formation of toxic metabolites, changing the route of administration, and increasing the cost-effectiveness of a therapy. |
|      0 |       4 | We weigh the benefits and risks of this approach, examining the impact of technology developments on drug design and how enhanced integration between cross-discipline teams can improve the outcome of drug discovery.                                   |

## Medical transcript data

> Data from R package `clinspacy` via <https://mtsamples.com/>

``` r
mts <- clinspacy::dataset_mtsamples()
mts_df <- reticulate::r_to_py(mts)
```

### medspacy

``` python
# jupyter-notebook
import medspacy
# nlp = spacy.load("en_core_sci_sm")
nlp = medspacy.load("en_core_sci_sm", disable = {'medspacy_target_matcher', 'medspacy_pyrush'})
nlp.add_pipe("sentencizer", first = True)
```

    ## <spacy.pipeline.sentencizer.Sentencizer object at 0x7ff0b948e500>

``` python
sectionizer = nlp.add_pipe("medspacy_sectionizer")
print(nlp.pipe_names)
```

    ## ['sentencizer', 'tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer', 'parser', 'ner', 'medspacy_context', 'medspacy_sectionizer']

``` python
texts0 = list(r.mts_df['transcription'][1:100])
doc = list(nlp.pipe(texts0))
```

``` python
#  is_negated
#  is_uncertain
#  is_historical
#  is_family
#  is_hypothetical

def spacy_get_transcripts(docs):

    details_dict = {
      "doc_id": [], 
      "sent_id": [],
      "section_category": [],
      "entity": [], 
      "is_historical": [],
      "start": [], 
      "end": []
      }
    
    for ix, doc in enumerate(docs):
      for sent_i, sent in enumerate(doc.sents):
        
        for ent in sent.ents:
          details_dict["doc_id"].append(ix)
          details_dict["sent_id"].append(sent_i)
          details_dict["section_category"].append(ent._.section_category)
          details_dict["entity"].append(ent.text)
          details_dict["is_historical"].append(ent._.is_historical)
          details_dict["start"].append(ent.start)
          details_dict["end"].append(ent.end)
    
    dd =  pd.DataFrame.from_dict(details_dict)     
    return dd
  
sp_transcripts = spacy_get_transcripts(doc)
```

``` r
reticulate::py$sp_transcripts |>
  slice(1:5) |> knitr::kable()
```

| doc_id | sent_id | section_category     | entity              | is_historical | start | end |
|------:|-------:|:-----------------|:-----------------|:------------|-----:|----:|
|      0 |       0 | past_medical_history | PAST                | TRUE          |     0 |   1 |
|      0 |       0 | past_medical_history | difficulty climbing | TRUE          |     7 |   9 |
|      0 |       0 | past_medical_history | difficulty          | TRUE          |    11 |  12 |
|      0 |       0 | past_medical_history | airline seats       | TRUE          |    13 |  15 |
|      0 |       0 | past_medical_history | tying shoes         | TRUE          |    16 |  18 |

## References

Eyre, A.B. Chapman, K.S. Peterson, J. Shi, P.R. Alba, M.M. Jones, T.L.
Box, S.L. DuVall, O. V Patterson, Launching into clinical space with
medspaCy: a new clinical text processing toolkit in Python, AMIA Annu.
Symp. Proc. 2021 (in Press. (n.d.). <http://arxiv.org/abs/2106.07799>.

Kormilitzin, A., Vaci, N., Liu, Q., & Nevado-Holgado, A. (2021). Med7: A
transferable clinical natural language processing model for electronic
health records. Artificial Intelligence in Medicine, 118, 102086.

Neumann, M., King, D., Beltagy, I., & Ammar, W. (2019). ScispaCy: fast
and robust models for biomedical natural language processing. arXiv
preprint arXiv:1902.07669.
