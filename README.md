# Some spaCy & scispacy workflows

*Updated: 2023-01-04*

> An attempt at organizing some `spaCy` workflows. Some functions for
> disentangling `spaCy` output.

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
# pip install pysbd
# pip install medspacy
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
dd <- pubmedr::pmed_search_pubmed(search_term = 'alzheimers treatment', 
                                  fields = c('TIAB','MH'))
```

    ## [1] "alzheimers treatment[TIAB] OR alzheimers treatment[MH]: 9999 records"

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
  "linker_name": "mesh"})
  
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

| doc_id | token            | token_order | sent_id | lemma            | ent_type | tag | dep    | pos   | is_stop | is_alpha | is_digit | is_punct |
|----:|:---------|------:|----:|:---------|:-----|:--|:----|:---|:----|:-----|:-----|:-----|
|      0 | Currently        |           0 |       0 | currently        |          | RB  | advmod | ADV   | FALSE   | TRUE     | FALSE    | FALSE    |
|      0 | ,                |           1 |       0 | ,                |          | ,   | punct  | PUNCT | FALSE   | FALSE    | FALSE    | TRUE     |
|      0 | biological       |           2 |       0 | biological       |          | JJ  | amod   | ADJ   | FALSE   | TRUE     | FALSE    | FALSE    |
|      0 | membrane-derived |           3 |       0 | membrane-derived |          | JJ  | amod   | ADJ   | FALSE   | FALSE    | FALSE    | FALSE    |
|      0 | nanoparticles    |           4 |       0 | nanoparticle     |          | NNS | nsubj  | NOUN  | FALSE   | TRUE     | FALSE    | FALSE    |

### Entities & linking

``` python
sp_entities = spacyHelp.spacy_get_entities(doc, link = True, linker = linker)
```

``` r
reticulate::py$sp_entities |>
  slice(1:10) |> knitr::kable()
```

| doc_id | sent_id | entity                 | label    | start | end | start_char | end_char | uid     | descriptor             | score |
|----:|-----:|:-------------|:-----|----:|---:|------:|-----:|:-----|:-------------|:----|
|      0 |       3 | tumor                  | DISEASE  |   102 | 103 |        662 |      667 | D009369 | Neoplasms              | 0.82  |
|      0 |       4 | cancer                 | DISEASE  |   126 | 127 |        811 |      817 | D009369 | Neoplasms              | 1     |
|      0 |       4 | inflammation           | DISEASE  |   128 | 129 |        819 |      831 | D007249 | Inflammation           | 1     |
|      0 |       4 | immunological diseases | DISEASE  |   130 | 132 |        833 |      855 | D007154 | Immune System Diseases | 1     |
|      0 |       4 | bone diseases          | DISEASE  |   133 | 135 |        857 |      870 | D001847 | Bone Diseases          | 1     |
|      0 |       4 | Alzheimer’s disease    | DISEASE  |   136 | 139 |        875 |      894 | D000544 | Alzheimer Disease      | 1     |
|      1 |       0 | Alzheimer disease      | DISEASE  |     0 |   2 |          0 |       17 | D000544 | Alzheimer Disease      | 1     |
|      1 |       0 | dementia               | DISEASE  |    10 |  11 |         44 |       52 | D003704 | Dementia               | 1     |
|      1 |       1 | acupuncture            | CHEMICAL |    32 |  33 |        174 |      185 | D026881 | Acupuncture            | 1     |
|      1 |       1 | AD                     | DISEASE  |    39 |  40 |        221 |      223 | D000544 | Alzheimer Disease      | 1     |

### Abbreviations

``` python
sp_abbrevs = spacyHelp.spacy_get_abbrevs(doc)
```

``` r
reticulate::py$sp_abbrevs |>
  distinct(abrv, .keep_all = T) |>
  slice(1:10) |> knitr::kable()
```

| doc_id | abrv        | start | end | long_form                                                                                 |
|----:|:-------|----:|---:|:--------------------------------------------------|
|      0 | NPs         |   151 | 152 | nanoparticles                                                                             |
|      1 | AD          |   273 | 274 | Alzheimer disease                                                                         |
|      2 | IADL        |    12 |  13 | instrumental activities of daily living ”                                                 |
|      2 | LEARN       |    71 |  72 | Longitudinal Evaluation of Amyloid Risk and Neurodegeneration                             |
|      2 | ADCS ADL-PI |   130 | 132 | Alzheimer ’s Disease Cooperative Study Activities of Daily Living - Prevention Instrument |
|      2 | OR          |   410 | 411 | odds ratio                                                                                |
|      4 | MIND        |   268 | 269 | Mediterranean-DASH Intervention for Neurodegenerative Delay                               |
|      6 | PD          |   174 | 175 | Parkinson ’s Disease                                                                      |
|      6 | Disease     |    10 |  11 | Disease ( PD                                                                              |
|      6 | HZV         |   106 | 107 | herpes zoster virus                                                                       |

### Noun phrases

``` python
sp_noun_phrases = spacyHelp.spacy_get_nps(doc)
```

``` r
reticulate::py$sp_noun_phrases |>
  slice(1:5) |> knitr::kable()
```

| doc_id | sent_id | nounc                                     | start | end |
|-------:|--------:|:------------------------------------------|------:|----:|
|      0 |       0 | biological membrane-derived nanoparticles |     2 |   5 |
|      0 |       0 | (NPs                                      |     5 |   7 |
|      0 |       0 | enormous potential                        |    10 |  12 |
|      0 |       1 | these NPs                                 |    25 |  27 |
|      0 |       1 | some methods                              |    34 |  36 |

### Hyponyms

``` python
sp_hearst = spacyHelp.spacy_get_hyponyms(doc)
```

``` r
reticulate::py$sp_hearst |>
  slice(1:10) |> knitr::kable()
```

| doc_id | pred       | sbj          | obj          |
|-------:|:-----------|:-------------|:-------------|
|      4 | include    | study        | participants |
|      8 | especially | diseases     | Parkinson    |
|      9 | include    | diseases     | Alzheimer    |
|     10 | include    | Data sources | interviews   |
|     10 | such_as    | events       | transition   |
|     19 | include    | distress     | symptoms     |
|     19 | include    | distress     | impairment   |
|     19 | include    | distress     | uncertainty  |
|     19 | include    | distress     | care         |
|     19 | include    | distress     | falls        |

### Negation

### Sentences

``` python
sp_sentences = spacyHelp.spacy_get_sentences(doc)
```

``` r
reticulate::py$sp_sentences |>
  slice(1:5) |> knitr::kable()
```

| doc_id | sent_id | text                                                                                                                                                                                                  |
|---:|---:|:----------------------------------------------------------------|
|      0 |       0 | Currently, biological membrane-derived nanoparticles (NPs) have shown enormous potential as drug delivery vehicles due to their outstanding biomimetic properties.                                    |
|      0 |       1 | To make these NPs more adaptive to complex biological systems, some methods have been developed to modify biomembranes and endow them with more functions while preserving their inherent natures.    |
|      0 |       2 | In this review, we introduce five common approaches used for biomembrane decoration: membrane hybridization, the postinsertion method, chemical methods, metabolism engineering and gene engineering. |
|      0 |       3 | These methods can functionalize a series of biomembranes derived from red blood cells, white blood cells, tumor cells, platelets, exosomes and so on.                                                 |
|      0 |       4 | Biomembrane engineering could markedly facilitate the targeted drug delivery, treatment and diagnosis of cancer, inflammation, immunological diseases, bone diseases and Alzheimer’s disease.         |

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
