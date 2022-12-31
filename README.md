# Some spaCy & scispacy workflows

*Updated: 2022-12-31*

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
    -   [References](#references)

------------------------------------------------------------------------

## Conda environment

``` bash
conda create -n scispacy python=3.9
source activate scispacy 
conda install transformers pandas numpy

cd /home/jtimm/anaconda3/envs/scispacy/bin/
pip install scispacy
pip install pysbd
pip install medspacy
pip install textacy
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

``` python
import sys
#print(sys.path)
sys.path.append('../home/jtimm/pCloudDrive/GitHub/git-projects/spacy-nlp')
import spacyHelp
```

## PubMed abstracts

``` r
dd <- pubmedr::pmed_search_pubmed(search_term = 'political ideology', 
                                  fields = c('TIAB','MH'))
```

    ## [1] "political ideology[TIAB] OR political ideology[MH]: 639 records"

``` r
dd.df <- pubmedr::pmed_get_records2(pmids = unique(dd$pmid)[1:200], 
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
sp_df = spacyHelp.spacy_get_df(doc)
```

``` r
reticulate::py$sp_df |>
  slice(1:5) |> knitr::kable()
```

| doc_id | token       | token_order | sent_id | lemma       | ent_type | tag | dep   | pos  | is_stop | is_alpha | is_digit | is_punct |
|----:|:-------|-------:|-----:|:-------|:-----|:---|:----|:---|:-----|:-----|:-----|:-----|
|      0 | Previous    |           0 |       0 | previous    |          | JJ  | amod  | ADJ  | FALSE   | TRUE     | FALSE    | FALSE    |
|      0 | attitudinal |           1 |       0 | attitudinal |          | JJ  | amod  | ADJ  | FALSE   | TRUE     | FALSE    | FALSE    |
|      0 | studies     |           2 |       0 | study       |          | NNS | nsubj | NOUN | FALSE   | TRUE     | FALSE    | FALSE    |
|      0 | on          |           3 |       0 | on          |          | IN  | case  | ADP  | TRUE    | TRUE     | FALSE    | FALSE    |
|      0 | immigration |           4 |       0 | immigration |          | NN  | nmod  | NOUN | FALSE   | TRUE     | FALSE    | FALSE    |

### Entities & linking

``` python
sp_entities = spacyHelp.spacy_get_entities(doc)
```

``` r
reticulate::py$sp_entities |>
  slice(1:5) |> knitr::kable()
```

| doc_id | sent_id | entity             | label   | start | end | start_char | end_char |
|-------:|--------:|:-----------------|:--------|------:|----:|----------:|--------:|
|      2 |       1 | pandemic           | DISEASE |    40 |  41 |        247 |      255 |
|      2 |       1 | pandemic           | DISEASE |    54 |  55 |        339 |      347 |
|      2 |       4 | pandemic           | DISEASE |   144 | 145 |        888 |      896 |
|      3 |       0 | vehicular homicide | DISEASE |    21 |  23 |        131 |      149 |
|      3 |       1 | vehicular homicide | DISEASE |    36 |  38 |        246 |      264 |

### Abbreviations

``` python
sp_abbrevs = spacyHelp.spacy_get_abbrevs(doc)
```

``` r
reticulate::py$sp_abbrevs |>
  distinct(abrv, .keep_all = T) |>
  slice(1:10) |> knitr::kable()
```

| doc_id | abrv     | start | end | long_form                                  |
|-------:|:---------|------:|----:|:-------------------------------------------|
|      0 | PRRI     |    54 |  55 | Public Religion Research Institute         |
|      7 | study    |    39 |  40 | Study 1 : N =   47,951                     |
|      8 | VA       |    26 |  27 | Veterans Administration                    |
|     13 | PMIE     |    60 |  61 | potentially morally injurious events       |
|     17 | COVID-19 |    10 |  11 | Coronavirus disease 2019                   |
|     18 | CDC      |    76 |  77 | Centers for Disease Control and Prevention |
|     23 | NAM      |    14 |  15 | norm activation model                      |
|     24 | WVS      |    68 |  69 | World Values Survey                        |
|     25 | IPV      |    69 |  70 | intimate partner violence                  |
|     34 | HICs     |     4 |   5 | high-income countries                      |

### Noun phrases

``` python
sp_noun_phrases = spacyHelp.spacy_get_nps(doc)
```

``` r
reticulate::py$sp_noun_phrases |>
  slice(1:5) |> knitr::kable()
```

| doc_id | sent_id | nounc                        | start | end |
|-------:|--------:|:-----------------------------|------:|----:|
|      0 |       0 | Previous attitudinal studies |     0 |   3 |
|      0 |       0 | immigration policies         |    19 |  21 |
|      0 |       1 | The dearth                   |    22 |  24 |
|      0 |       1 | the present study            |    31 |  34 |
|      0 |       2 | individual-level data        |    37 |  39 |

### Hyponyms

``` python
sp_hearst = spacyHelp.spacy_get_hyponyms(doc)
```

``` r
reticulate::py$sp_hearst |>
  slice(1:10) |> knitr::kable()
```

| doc_id | pred    | sbj     | obj               |
|-------:|:--------|:--------|:------------------|
|      0 | other   | factors | states            |
|      0 | such_as | factors | age               |
|      0 | such_as | factors | ideology          |
|      0 | such_as | factors | party affiliation |
|      0 | such_as | factors | region            |
|      6 | such_as | factors | effort            |
|      6 | such_as | factors | favoritism        |
|      6 | such_as | factors | discrimination    |
|     10 | include | survey  | items             |
|     10 | include | survey  | sources           |

### Negation

### Sentences

``` python
sp_sentences = spacyHelp.spacy_get_sentences(doc)
```

``` r
reticulate::py$sp_sentences |>
  slice(1:5) |> knitr::kable()
```

| doc_id | sent_id | text                                                                                                                                                                                                                                                             |
|--:|---:|:-----------------------------------------------------------------|
|      0 |       0 | Previous attitudinal studies on immigration in the USA largely focus on the predictors of anti-immigration sentiments compared to examining immigration policies.                                                                                                |
|      0 |       1 | The dearth of scientific enquiry about the latter necessitated the present study.                                                                                                                                                                                |
|      0 |       2 | By analyzing individual-level data (n = 1018) obtained from the Public Religion Research Institute (PRRI), we assess the effect of geopolitics-red and blue states and other factors on public attitude towards six immigration policies in the USA (2017-2021). |
|      0 |       3 | Overall, the results indicate a null relationship between geopolitics and public attitude towards immigration policies.                                                                                                                                          |
|      0 |       4 | Additionally, we observed several sociodemographic factors, such as age, political ideology, party affiliation, and region, influence public attitude towards immigration policies.                                                                              |

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
