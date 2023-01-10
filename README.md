# Some spaCy & scispacy wrapper functions

*Updated: 2023-01-10*

> An attempt at organizing some `spaCy` workflows, including some
> functions for disentangling `spaCy` output as data frames.

------------------------------------------------------------------------

-   [Some spaCy & scispacy wrapper
    functions](#some-spacy-&-scispacy-wrapper-functions)
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
  config={"extended": True})
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

| doc_id | sent_id | entity               | label    | start | end | start_char | end_char | uid        | descriptor                | score |
|----:|-----:|:-----------|:-----|----:|---:|------:|-----:|:------|:--------------|:----|
|    182 |       4 | rapamycin            | CHEMICAL |   127 | 128 |        855 |      864 | D020123    | Sirolimus                 | 1     |
|    118 |       1 | glucose              | CHEMICAL |    24 |  25 |        149 |      156 | D005947    | Glucose                   | 1     |
|    112 |       3 | IADL/CAT             | CHEMICAL |   151 | 152 |        963 |      971 |            |                           |       |
|     20 |       8 | whole-brain atrophy  | DISEASE  |   175 | 176 |       1156 |     1175 |            |                           |       |
|     45 |       2 | pandemic             | DISEASE  |    62 |  63 |        396 |      404 | D058873    | Pandemics                 | 0.83  |
|     71 |       6 | CRF                  | DISEASE  |   166 | 167 |        996 |      999 | D000072599 | Cardiorespiratory Fitness | 1     |
|    107 |       3 | cognitive impairment | DISEASE  |    78 |  79 |        491 |      511 | D060825    | Cognitive Dysfunction     | 0.95  |
|     13 |       7 | MCI                  | DISEASE  |   227 | 228 |       1289 |     1292 | D060825    | Cognitive Dysfunction     | 1     |
|    176 |       0 | NYT                  | CHEMICAL |     2 |   3 |         15 |       18 |            |                           |       |
|      4 |       6 | AD                   | DISEASE  |   201 | 202 |       1141 |     1143 | D000544    | Alzheimer Disease         | 1     |
|    150 |       2 | Alzheimer’s disease  | DISEASE  |    60 |  61 |        423 |      442 | D000544    | Alzheimer Disease         | 1     |
|     81 |       1 | dementia             | DISEASE  |    50 |  51 |        317 |      325 | D003704    | Dementia                  | 1     |
|      9 |       5 | neurotoxicity        | DISEASE  |   117 | 118 |        761 |      774 | D020258    | Neurotoxicity Syndromes   | 0.82  |
|     47 |       3 | AuNS                 | CHEMICAL |    96 |  97 |        681 |      685 |            |                           |       |
|    187 |       6 | OLST(p \> .05)       | CHEMICAL |   207 | 208 |       1240 |     1253 |            |                           |       |

### Abbreviations

``` python
sp_abbrevs = spacyHelp.spacy_get_abbrevs(doc)
```

``` r
reticulate::py$sp_abbrevs |>
  distinct(abrv, .keep_all = T) |>
  sample_n(15) |> knitr::kable()
```

| doc_id | abrv    | start | end | long_form                                                          |
|------:|:------|-----:|---:|:-------------------------------------------------|
|     26 | FA      |   100 | 101 | femoral artery                                                     |
|      9 | BBB     |    36 |  37 | blood-brain barrier                                                |
|      4 | MNA     |    71 |  72 | Mini-Nutritional Assessment                                        |
|     96 | TBI     |    83 |  84 | Traumatic brain injury                                             |
|    192 | SET     |   100 | 101 | strength and endurance training                                    |
|     64 | AM      |    56 |  57 | and moxibustion                                                    |
|     51 | PRISMA  |   173 | 174 | Preferred Reporting Items for Systematic Reviews and Meta-Analyses |
|     30 | ARIA    |   115 | 116 | amyloid-related imaging abnormalities                              |
|    178 | SUCRA   |   389 | 390 | surface under the cumulative ranking curve                         |
|     29 | ADLQ    |   109 | 110 | Activities of Daily Living Questionnaire                           |
|    176 | NYT     |   111 | 112 | Ninjin’yoeito                                                      |
|     73 | HD-tDCS |   193 | 194 | high-definition transcranial direct current stimulation            |
|     26 | α-syn   |   222 | 223 | α-synuclein                                                        |
|    123 | ANP     |   355 | 356 | Advanced nursing practice                                          |
|    191 | OAPQ    |   161 | 162 | Older Age Psychotropic Quiz                                        |

### Noun phrases

``` python
sp_noun_phrases = spacyHelp.spacy_get_nps(doc)
```

``` r
reticulate::py$sp_noun_phrases |>
  sample_n(10) |> knitr::kable()
```

| doc_id | sent_id | nounc                                                  | start | end |
|------:|-------:|:----------------------------------------------|-----:|----:|
|     20 |       6 | the 16 mg/day dose                                     |   139 | 143 |
|    186 |       1 | education                                              |    50 |  51 |
|    111 |       6 | (PwD                                                   |   166 | 168 |
|      6 |      11 | a non-pharmaceutical option                            |   265 | 268 |
|    184 |       4 | higher satisfaction                                    |   134 | 136 |
|    181 |       2 | a target-sensing catalyst activation (TaSCAc) strategy |    60 |  68 |
|    103 |       1 | The benefits                                           |    16 |  18 |
|     30 |       2 | recommendations                                        |    62 |  63 |
|    187 |       0 | face-to-face treatment                                 |    20 |  22 |
|     94 |       5 | verum acupuncture treatment                            |   135 | 138 |

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

| doc_id | sbj                                                   | pred       | obj                  |
|------:|:---------------------------------------|:--------|:----------------|
|     52 | outcomes                                              | include    | perfusion            |
|     27 | neurodegenerative diseases                            | include    | Alzheimer’s disease  |
|     72 | model                                                 | include    | age                  |
|     88 | agents                                                | such_as    | growth factors       |
|    144 | parameters                                            | such_as    | evaluation test time |
|    136 | phytonutrients                                        | such_as    | antioxidants         |
|     79 | age-related disorders including neurological diseases | such_as    | Alzheimer’s disease  |
|     59 | neurological and psychiatric disorders                | include    | diabetes             |
|    189 | viral diseases                                        | such_as    | encephalitis         |
|    141 | caregiver factors                                     | include    | training status      |
|      7 | database                                              | such_as    | Wanfang Data         |
|    136 | phytonutrients                                        | such_as    | vitamins             |
|    119 | We                                                    | include    | carers               |
|    149 | medicines                                             | especially | cataract             |
|     72 | model                                                 | include    | ADL dependency count |

#### Relation types:

``` r
reticulate::py$sp_hearst |>
  count(pred) |>
  knitr::kable()
```

| pred       |   n |
|:-----------|----:|
| especially |   3 |
| include    |  74 |
| other      |   6 |
| such_as    |  69 |

### Negation

### Sentences

``` python
sp_sentences = spacyHelp.spacy_get_sentences(doc)
```

``` r
reticulate::py$sp_sentences |>
  sample_n(7) |> knitr::kable()
```

| doc_id | sent_id | text                                                                                                                                                                                                                                                                                                                            |
|--:|--:|:-----------------------------------------------------------------|
|    139 |       2 | Virtual reality (VR) can resemble real life with immersive stimuli, but there have been few studies confirming its ecological effects on ADL and IADL.                                                                                                                                                                          |
|    120 |       0 | Characterized by the presence of amyloid plaques, neurofibrillary tangles and neuroinflammation, Alzheimer’s disease (AD) is a progressive neurodegenerative disorder with no known treatment or cure.                                                                                                                          |
|     92 |      11 | Finally, we demonstrate robust xenograft survival at multiple cell doses up to 6 months in both C57BL/6J mice and a transgenic Alzheimer’s disease model (p \< .001).                                                                                                                                                           |
|     22 |       1 | Pubmed, Scopus, PEDro, Web of Science, CINAHL, Cochrane Library, grey literature and a reverse search from inception to April 2021 were searched to identify documents.                                                                                                                                                         |
|     79 |       1 | Limited progress has been made in the development of clinically translatable therapies for these central nervous system (CNS) diseases.                                                                                                                                                                                         |
|     52 |       1 | Meditation practices recently emerged as a promising mental training exercise to foster brain health and reduce dementia risk.                                                                                                                                                                                                  |
|    111 |       8 | A lower mental and physical health-related quality of life, age of PwD, lower education, higher deficits in daily living activities, higher depressive symptoms, and a higher number of drugs taken of the PwD, as well as female sex of the caregiver were associated with a significantly higher number of tasks carried out. |

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
