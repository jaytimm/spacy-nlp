# Some spaCy & scispacy workflows

*Updated: 2023-01-10*

> An attempt at organizing some `spaCy` workflows. Some functions for
> disentangling `spaCy` output.

------------------------------------------------------------------------

-   [Some spaCy & scispacy workflows](#some-spacy-&-scispacy-workflows)
    -   [Conda environment](#conda-environment)
    -   [Reticulate](#reticulate)
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

## Reticulate

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
dd <- pubmedr::pmed_search_pubmed(search_term = 'alzheimers treatment', 
                                  fields = c('TIAB','MH'),
                                  verbose = F)

dd.df <- pubmedr::pmed_get_records2(pmids = unique(dd$pmid)[1:200], 
                                    with_annotations = F)[[1]] |>
  filter(!is.na(abstract))

df <- reticulate::r_to_py(dd.df)
```

## Libraries

``` python
import sys
sys.path.append('../home/jtimm/pCloudDrive/GitHub/git-projects/spacy-nlp')
import spacyHelp
```

``` python
import pandas as pd
import os
import spacy
import scispacy
#nlp = spacy.load("en_core_sci_sm")
#nlp = spacy.load("en_core_web_sm")
```

## Scispacy components

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
nlp.add_pipe("merge_entities")

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

    ## ['sentencizer', 'tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer', 'parser', 'ner', 'merge_entities', 'abbreviation_detector', 'scispacy_linker', 'hyponym_detector']

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

| doc_id | token     | token_order | sent_id | lemma     | ent_type | tag | dep       | pos  | is_stop | is_alpha | is_digit | is_punct |
|----:|:------|-------:|-----:|:------|:-----|:---|:------|:---|:-----|:-----|:-----|:-----|
|      0 | A         |           0 |       0 | a         |          | DT  | det       | DET  | TRUE    | TRUE     | FALSE    | FALSE    |
|      0 | high      |           1 |       0 | high      |          | JJ  | amod      | ADJ  | FALSE   | TRUE     | FALSE    | FALSE    |
|      0 | adherence |           2 |       0 | adherence |          | NN  | nsubjpass | NOUN | FALSE   | TRUE     | FALSE    | FALSE    |
|      0 | to        |           3 |       0 | to        |          | IN  | case      | ADP  | TRUE    | TRUE     | FALSE    | FALSE    |
|      0 | the       |           4 |       0 | the       |          | DT  | det       | DET  | TRUE    | TRUE     | FALSE    | FALSE    |

### Entities & linking

``` python
sp_entities = spacyHelp.spacy_get_entities(
  doc, 
  link = True, 
  linker = linker)
```

``` r
reticulate::py$sp_entities |>
  sample_n(15) |> knitr::kable()
```

| doc_id | sent_id | entity                   | label    | start | end | start_char | end_char | uid     | descriptor                | score |
|----:|-----:|:-------------|:-----|----:|---:|------:|-----:|:-----|:--------------|:----|
|     84 |       5 | Aβ                       | CHEMICAL |   159 | 160 |        816 |      818 | D000682 | Amyloid                   | 0.86  |
|    161 |       1 | cognitive impairment     | DISEASE  |    50 |  51 |        367 |      387 | D060825 | Cognitive Dysfunction     | 0.95  |
|     86 |       5 | oxytocin                 | CHEMICAL |   142 | 143 |        858 |      866 | D010121 | Oxytocin                  | 1     |
|    119 |       5 | dementia                 | DISEASE  |   109 | 110 |        713 |      721 | D003704 | Dementia                  | 1     |
|    130 |       4 | ChEI                     | CHEMICAL |   145 | 146 |       1079 |     1083 |         |                           |       |
|      4 |       4 | DLB                      | DISEASE  |   139 | 140 |        823 |      826 | D020961 | Lewy Body Disease         | 0.77  |
|    116 |       3 | dementia                 | DISEASE  |    94 |  95 |        562 |      570 | D003704 | Dementia                  | 1     |
|     75 |       3 | anxiety                  | DISEASE  |   116 | 117 |        707 |      714 | D001007 | Anxiety                   | 1     |
|    151 |      11 | edema                    | DISEASE  |   440 | 441 |       2480 |     2485 | D004487 | Edema                     | 1     |
|     63 |       2 | cadmium                  | CHEMICAL |    75 |  76 |        546 |      553 | D002104 | Cadmium                   | 1     |
|     37 |       7 | delusions                | DISEASE  |   226 | 227 |       1365 |     1374 | D003702 | Delusions                 | 1     |
|     98 |       6 | oxygen                   | CHEMICAL |   198 | 199 |       1151 |     1157 | D010100 | Oxygen                    | 1     |
|     35 |       2 | yCDs-Ce6                 | CHEMICAL |    52 |  53 |        349 |      357 |         |                           |       |
|     15 |       2 | Alzheimer’s disease      | DISEASE  |    49 |  50 |        305 |      324 | D000544 | Alzheimer Disease         | 1     |
|    189 |       0 | hepatocellular carcinoma | DISEASE  |    17 |  18 |        106 |      130 | D006528 | Carcinoma, Hepatocellular | 1     |

### Abbreviations

``` python
sp_abbrevs = spacyHelp.spacy_get_abbrevs(doc)
```

``` r
reticulate::py$sp_abbrevs |>
  distinct(abrv, .keep_all = T) |>
  sample_n(15) |> knitr::kable()
```

| doc_id | abrv    | start | end | long_form                                                       |
|------:|:------|-----:|---:|:------------------------------------------------|
|     29 | ADLQ    |   109 | 110 | Activities of Daily Living Questionnaire                        |
|    174 | NPs     |     6 |   7 | nanoparticles                                                   |
|     49 | a-MCI   |   302 | 303 | amnestic mild cognitive impairment                              |
|    146 | PSQI    |   254 | 255 | Pittsburgh sleep quality index                                  |
|    157 | PKA     |   255 | 256 | protein kinase A                                                |
|     89 | OAs     |   263 | 264 | older adults                                                    |
|    166 | mRNA    |   100 | 101 | Messenger RNA                                                   |
|     51 | ADMET   |   319 | 320 | absorption , distribution , metabolism , excretion and toxicity |
|     59 | TMS     |    71 |  72 | Transcranial magnetic stimulation                               |
|      6 | APOE e4 |    91 |  93 | apolipoprotein E e4                                             |
|    114 | DRFI    |   244 | 245 | Delirium Risk Factor Identification                             |
|    190 | ALS     |   116 | 117 | Amyotrophic lateral sclerosis                                   |
|    120 | FMT     |    85 |  86 | Fecal microbiota transplantation                                |
|    176 | MCAO    |   159 | 160 | middle cerebral artery occlusion                                |
|     26 | RFCA    |    64 |  65 | radiofrequency catheter ablation                                |

### Noun phrases

``` python
sp_noun_phrases = spacyHelp.spacy_get_nps(doc)
```

``` r
reticulate::py$sp_noun_phrases |>
  sample_n(10) |> knitr::kable()
```

| doc_id | sent_id | nounc                               | start | end |
|-------:|--------:|:------------------------------------|------:|----:|
|    161 |       1 | very mild and short-lasting effects |    71 |  76 |
|    163 |       4 | approaches                          |   160 | 161 |
|    110 |      12 | We                                  |   292 | 293 |
|    145 |      11 | the proportion                      |   219 | 221 |
|    176 |       8 | (IR                                 |   183 | 185 |
|    134 |       3 | phenotypic outcomes                 |    98 | 100 |
|    117 |       8 | the caregivers                      |   203 | 205 |
|    185 |      10 | eight choice tasks                  |   208 | 211 |
|     41 |       2 | this need                           |    44 |  46 |
|     11 |       3 | X-rays                              |    89 |  90 |

### Hyponyms

> Works better with nlp.add_pipe(“merge_entities”)

``` python
sp_hearst = spacyHelp.spacy_get_hyponyms(doc)
```

``` r
reticulate::py$sp_hearst |>
  select(doc_id, sbj, pred, obj) |>
  sample_n(15) |> knitr::kable()
```

| doc_id | sbj                                                   | pred    | obj                           |
|-----:|:-------------------------------------|:------|:---------------------|
|    189 | viruses                                               | include | hepatocellular carcinoma      |
|    103 | neurological diseases                                 | include | autism spectrum disorder      |
|    136 | diets                                                 | other   | vegetarianism                 |
|     59 | neurological and psychiatric disorders                | include | diabetes                      |
|     51 | diseases                                              | such_as | Alzheimer’s Disease           |
|    115 | treatments                                            | include | tube feeding                  |
|     38 | CPs                                                   | include | guidance                      |
|    186 | care                                                  | such_as | health integration            |
|    115 | treatments                                            | include | care unit care                |
|     63 | Neurodegenerative diseases                            | such_as | Alzheimer’s disease           |
|    144 | English keywords                                      | such_as | neurons                       |
|     63 | Neurodegenerative diseases                            | such_as | amyotrophic lateral sclerosis |
|    115 | treatments                                            | include | ventilation                   |
|     79 | age-related disorders including neurological diseases | such_as | Parkinson’s disease           |
|      7 | database                                              | such_as | Pubmed                        |

### Negation

### Sentences

``` python
sp_sentences = spacyHelp.spacy_get_sentences(doc)
```

``` r
reticulate::py$sp_sentences |>
  sample_n(7) |> knitr::kable()
```

| doc_id | sent_id | text                                                                                                                                                                                                                                                                                                                                |
|--:|--:|:-----------------------------------------------------------------|
|    130 |       3 | We present an evidence-based overview of determinants, spanning genetic, molecular, and large-scale networks, involved in the response to ChEI in patients with AD and other neurodegenerative diseases.                                                                                                                            |
|    155 |       9 | However, crossword puzzles might result in cognitively beneficial remodeling between the DMN and other networks in more severely impaired MCI subjects, parallel to the observed clinical benefits.                                                                                                                                 |
|    136 |       1 | The purpose of this review is to summarize the current knowledge on the positive and negative aspects of a vegan diet regarding the risk of AD.                                                                                                                                                                                     |
|    114 |       8 | Staff delirium knowledge was assessed.                                                                                                                                                                                                                                                                                              |
|     44 |       6 | CD38 and its ligand Pecam1, one of the energy shuttle pathways between neurons and astrocytes, were also be detected.                                                                                                                                                                                                               |
|    191 |       6 | Medication data were obtained from residents’ medication charts.                                                                                                                                                                                                                                                                    |
|     48 |       2 | The lymph nodes of amyloid precursor protein/presenilin 1 (APP/PS1) and 3xTg (APP/PS1/tau) mouse models of AD were treated with photobiomodulation therapy (PBMT) for 10 J/cm2 per day for 1 month (10 min for each day), T lymphocytes isolated from these two AD models were treated with PBMT for 2 J/cm2 (5 min for each time). |

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
