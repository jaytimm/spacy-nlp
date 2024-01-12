# Some spaCy & scispacy wrapper functions

*Updated: 2024-01-12*

> An attempt at organizing some `spaCy` workflows, including some
> functions for disentangling `spaCy` output as data frames.

------------------------------------------------------------------------

-   [Some spaCy & scispacy wrapper
    functions](#some-spacy-&-scispacy-wrapper-functions)
    -   [Conda environment](#conda-environment)
    -   [Reticulate](#reticulate)
    -   [News article corpus](#news-article-corpus)
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
conda create -n scispacy051 python==3.9

conda activate scispacy051

conda update --all
conda install nmslib pandas numpy
pip install dframcy

pip install scispacy==0.5.1

conda install spacy -c conda-forge
 
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
```

## Reticulate

``` r
## <R-console>
library(dplyr)
Sys.setenv(RETICULATE_PYTHON = "/home/jtimm/miniconda3/envs/scispacy051/bin/python")
library(reticulate)
#reticulate::use_python("/home/jtimm/anaconda3/envs/m3demo/bin/python")
reticulate::use_condaenv(condaenv = "scispacy051",
                         conda = "/home/jtimm/miniconda3/bin/conda")
```

## News article corpus

``` r
dd.df <- textpress::web_scrape_urls(x = 'Alzheimer Disease', cores = 10) |>
  filter(!is.na(text))
df <- reticulate::r_to_py(dd.df)
```

## Libraries

``` python
import sys
sys.path.append('/home/jtimm/pCloudDrive/GitHub/git-projects/spacy-nlp')
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
  config={"extended": True})
```

``` python
print(nlp.pipe_names)
```

    ## ['sentencizer', 'tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer', 'parser', 'ner', 'merge_entities', 'abbreviation_detector', 'scispacy_linker', 'hyponym_detector']

## Spacy annotate

``` python
texts = list(r.df['text'])
doc = list(nlp.pipe(texts))
```

## Extraction functions

### Standard annotation

``` python
sp_df = spacyHelp.spacy_extract_df(doc)
```

``` r
reticulate::py$sp_df |>
  slice(1:5) |> knitr::kable()
```

| doc_id | token               | token_order | sent_id | lemma               | ent_type | tag | dep       | pos  | is_stop | is_alpha | is_digit | is_punct |
|----:|:---------|------:|----:|:---------|:-----|:--|:-----|:---|:----|:-----|:-----|:-----|
|      0 | Alzheimer’s disease |           0 |       0 | alzheimer’s disease | DISEASE  | NN  | nsubjpass | NOUN | FALSE   | FALSE    | FALSE    | FALSE    |
|      0 | is                  |           1 |       0 | be                  |          | VBZ | auxpass   | AUX  | TRUE    | TRUE     | FALSE    | FALSE    |
|      0 | expected            |           2 |       0 | expect              |          | VBN | ROOT      | VERB | FALSE   | TRUE     | FALSE    | FALSE    |
|      0 | to                  |           3 |       0 | to                  |          | TO  | mark      | PART | TRUE    | TRUE     | FALSE    | FALSE    |
|      0 | impact              |           4 |       0 | impact              |          | VB  | xcomp     | VERB | FALSE   | TRUE     | FALSE    | FALSE    |

### Entities & linking

``` python
sp_entities = spacyHelp.spacy_extract_entities(
  doc, 
  linker = linker)
```

``` r
reticulate::py$sp_entities |>
  sample_n(15) |> knitr::kable()
```

| doc_id | sent_id | entity                                          | label    | start |  end | start_char | end_char | uid     | descriptor                      | score |
|---:|----:|:--------------------|:----|---:|---:|-----:|----:|:----|:-------------|---:|
|     28 |       2 | amyloid PET                                     | CHEMICAL |    67 |   68 |        420 |      431 | D000682 | Amyloid                         |  0.83 |
|     17 |     163 | ADAD                                            | DISEASE  |  5550 | 5551 |      27931 |    27935 | D007589 | Job Syndrome                    |  0.80 |
|     40 |      22 | inflammation                                    | DISEASE  |   657 |  658 |       3449 |     3461 | D007249 | Inflammation                    |  1.00 |
|     51 |     125 | androsterone sulfate                            | CHEMICAL |  3548 | 3549 |      20477 |    20497 | D043266 | Steryl-Sulfatase                |  0.76 |
|     49 |      46 | ’                                               | DISEASE  |  1049 | 1050 |       6492 |     6493 | NA      | NA                              |   NaN |
|      0 |      13 | constipation                                    | DISEASE  |   294 |  295 |       1655 |     1667 | D003248 | Constipation                    |  1.00 |
|     71 |       7 | dementia                                        | DISEASE  |   120 |  121 |        703 |      711 | D003704 | Dementia                        |  1.00 |
|     52 |     266 | MAPT                                            | DISEASE  |  7529 | 7530 |      41788 |    41792 | D008869 | Microtubule-Associated Proteins |  0.89 |
|      5 |      32 | Tau                                             | CHEMICAL |  1079 | 1080 |       6163 |     6166 | D016875 | tau Proteins                    |  0.82 |
|      7 |       8 | dementia                                        | DISEASE  |   239 |  240 |       1314 |     1322 | D003704 | Dementia                        |  1.00 |
|     50 |     527 | Herpes simplex virus type 1 and other pathogens | DISEASE  |  8651 | 8652 |      48582 |    48629 | D018259 | Herpesvirus 1, Human            |  0.78 |
|     63 |      13 | obesity                                         | DISEASE  |   262 |  263 |       1530 |     1537 | D009765 | Obesity                         |  1.00 |
|     46 |       3 | dementia                                        | DISEASE  |   104 |  105 |        628 |      636 | D003704 | Dementia                        |  1.00 |
|     85 |      40 | amyloid                                         | CHEMICAL |   745 |  746 |       4124 |     4131 | D000682 | Amyloid                         |  1.00 |
|      0 |       3 | dementia                                        | DISEASE  |   107 |  108 |        631 |      639 | D003704 | Dementia                        |  1.00 |

### Abbreviations

``` python
sp_abbrevs = spacyHelp.spacy_extract_abbrevs(doc)
```

``` r
reticulate::py$sp_abbrevs |>
  distinct(abrv, .keep_all = T) |>
  sample_n(15) |> knitr::kable()
```

| doc_id | abrv    | start |   end | long_form                                                  |
|------:|:-------|-----:|-----:|:----------------------------------------------|
|     51 | DHEAS   |  8578 |  8579 | dehydroepiandrosterone sulfate                             |
|     51 | OAT3    |  8131 |  8132 | organic anion transporter 3                                |
|     51 | PERADES |  1859 |  1860 | Polygenic , and Environmental Risk for Alzheimer’s Disease |
|      5 | MCI     |   100 |   101 | mild cognitive impairment                                  |
|     42 | MS      |   362 |   363 | multiple sclerosis                                         |
|     17 | RMSE    |  3109 |  3110 | root mean square error                                     |
|      4 | ADDF    |   965 |   966 | Alzheimer ’s Drug Discovery Foundation ’s                  |
|     17 | CIHR    | 10349 | 10350 | Canadian Institutes of Health Research                     |
|     76 | PBA     |    65 |    66 | 4-phenylbutyrate                                           |
|     14 | GWAS    |    41 |    42 | Genome-Wide Association Study                              |
|     17 | APP     |  1020 |  1021 | amyloid precursor protein                                  |
|     52 | DTT     |  5964 |  5965 | dithiothreitol                                             |
|     31 | ADNI    |   613 |   614 | Alzheimer ’s Disease Neuroimaging Initiative               |
|     52 | BCA     |  5766 |  5767 | bicinchoninic acid                                         |
|     88 | NIH     |    47 |    48 | National Institutes of Health                              |

### Noun phrases

``` python
sp_noun_phrases = spacyHelp.spacy_extract_nps(doc)
```

``` r
set.seed(9)
reticulate::py$sp_noun_phrases |>
  sample_n(10) |> knitr::kable()
```

| doc_id | sent_id | nounc                                 | start |  end |
|-------:|--------:|:--------------------------------------|------:|-----:|
|     85 |      29 | possible hope                         |   494 |  496 |
|     17 |      37 | negative EYO values                   |  1185 | 1188 |
|     76 |      14 | the most prominent protein aggregates |   500 |  505 |
|     83 |      23 | the bubbles                           |   574 |  576 |
|     70 |       8 | their caregivers                      |   149 |  151 |
|     51 |     410 | a historical cohort study             |  8357 | 8361 |
|     52 |      86 | TREM2 signaling                       |  2342 | 2344 |
|     51 |      40 | that                                  |   988 |  989 |
|     56 |      11 | Cerebrospinal fluid                   |   288 |  290 |
|     51 |       2 | you                                   |    27 |   28 |

### Hyponyms

> Works better with nlp.add_pipe(“merge_entities”)

``` python
sp_hearst = spacyHelp.spacy_extract_hyponyms(doc)
```

``` r
reticulate::py$sp_hearst |>
  select(doc_id, sbj, pred, obj) |>
  sample_n(15) |> knitr::kable()
```

| doc_id | sbj                         | pred       | obj                      |
|-------:|:----------------------------|:-----------|:-------------------------|
|     11 | processes                   | such_as    | neuronal hyperplasticity |
|      8 | processes                   | such_as    | nerve cell growth        |
|     52 | subtype                     | compare_to | controls                 |
|     88 | dementia-related diseases   | like       | vascular dementia        |
|     10 | lack                        | be_a       | there                    |
|     51 | functions                   | include    | activities               |
|      7 | conditions                  | such_as    | heart disease            |
|     78 | imaging studies             | include    | immunotherapy            |
|     17 | pipeline                    | include    | registration             |
|     60 | conditions                  | such_as    | stroke                   |
|     50 | partner                     | other      | society                  |
|     58 | link                        | be_a       | there                    |
|     76 | neurodegenerative disorders | like_other | disease                  |
|     53 | biomarkers that             | include    | Aß                       |
|     81 | variations                  | such_as    | barrier impairment       |

#### Relation types:

``` r
reticulate::py$sp_hearst |>
  count(pred) |>
  knitr::kable()
```

| pred             |   n |
|:-----------------|----:|
| and-or_any_other |   2 |
| be_a             |  41 |
| compare_to       |  28 |
| eg               |   2 |
| for_example      |  17 |
| include          | 181 |
| like             |  25 |
| like_other       |   2 |
| mainly           |   1 |
| other            |  47 |
| other_than       |   1 |
| particularly     |   3 |
| such_as          | 106 |
| type             |   5 |
| whether          |   5 |

### Negation

### Sentences

``` python
sp_sentences = spacyHelp.spacy_extract_sentences(doc)
```

``` r
reticulate::py$sp_sentences |>
  sample_n(5) |> knitr::kable()
```

| doc_id | sent_id | text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|-:|--:|:-------------------------------------------------------------------|
|     83 |      49 | That same year, Antonio Regalado reported some of the first exciting results of the Alzheimer’s drug aducanumab.                                                                                                                                                                                                                                                                                                                                                                                                   |
|     40 |      24 | We have a process where we can take anyone’s natural killer cells, whether or not we take them from somebody who’s young and healthy or somebody who’s had multiple courses of chemotherapy and whose immune system has been beaten up, we can take the natural killer cells and grow them in a way that’s non-genetically modified, but we can turn them into billions of highly enhanced, highly aggressive cells where we dramatically increase the strength of the natural killer cell, the killing potential. |
|      8 |      28 | Neither your address nor the recipient’s address will be used for any other purpose.                                                                                                                                                                                                                                                                                                                                                                                                                               |
|      6 |      12 | Strategies targeting eradicating or managing bacterial infections in the stomach may emerge as potential interventions to mitigate Alzheimer’s risk \[4\].                                                                                                                                                                                                                                                                                                                                                         |
|     60 |      41 | Some people with memory problems have a condition called mild cognitive impairment (MCI).                                                                                                                                                                                                                                                                                                                                                                                                                          |

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
