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

| doc_id | token            | token_order | sent_id | lemma            | ent_type | tag | dep    | pos   | is_stop | is_alpha | is_digit | is_punct |
|----:|:---------|------:|----:|:---------|:-----|:--|:----|:---|:----|:-----|:-----|:-----|
|      0 | Currently        |           0 |       0 | currently        |          | RB  | advmod | ADV   | FALSE   | TRUE     | FALSE    | FALSE    |
|      0 | ,                |           1 |       0 | ,                |          | ,   | punct  | PUNCT | FALSE   | FALSE    | FALSE    | TRUE     |
|      0 | biological       |           2 |       0 | biological       |          | JJ  | amod   | ADJ   | FALSE   | TRUE     | FALSE    | FALSE    |
|      0 | membrane-derived |           3 |       0 | membrane-derived |          | JJ  | amod   | ADJ   | FALSE   | FALSE    | FALSE    | FALSE    |
|      0 | nanoparticles    |           4 |       0 | nanoparticle     |          | NNS | nsubj  | NOUN  | FALSE   | TRUE     | FALSE    | FALSE    |

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

| doc_id | sent_id | entity                          | label    | start | end | start_char | end_char | uid     | descriptor              | score |
|----:|----:|:----------------|:-----|---:|--:|------:|-----:|:----|:------------|:---|
|    182 |       0 | polyuria                        | DISEASE  |    35 |  36 |        178 |      186 | D011141 | Polyuria                | 1     |
|     24 |      12 | dementia                        | DISEASE  |   284 | 285 |       1659 |     1667 | D003704 | Dementia                | 1     |
|    181 |       0 | dementias                       | DISEASE  |     9 |  10 |         68 |       77 | D003704 | Dementia                | 0.78  |
|    166 |       9 | Alzheimer’s disease             | DISEASE  |   250 | 251 |       1566 |     1585 | D000544 | Alzheimer Disease       | 1     |
|    149 |       3 | PD                              | DISEASE  |   135 | 136 |        938 |      940 |         |                         |       |
|    181 |       2 | dementia                        | DISEASE  |    71 |  72 |        494 |      502 | D003704 | Dementia                | 1     |
|    102 |       3 | ARIA                            | DISEASE  |    82 |  83 |        539 |      543 |         |                         |       |
|     37 |       1 | learning and memory impairments | DISEASE  |    31 |  32 |        212 |      243 |         |                         |       |
|    109 |       3 | dementia                        | DISEASE  |   112 | 113 |        732 |      740 | D003704 | Dementia                | 1     |
|    151 |       5 | GLP-1                           | CHEMICAL |   235 | 236 |       1573 |     1578 | D052216 | Glucagon-Like Peptide 1 | 1     |
|    113 |      12 | dementia                        | DISEASE  |   268 | 269 |       1555 |     1563 | D003704 | Dementia                | 1     |
|    192 |       0 | Brain diseases                  | DISEASE  |     0 |   1 |          0 |       14 | D001927 | Brain Diseases          | 1     |
|    147 |       6 | CISD2                           | CHEMICAL |   164 | 165 |       1017 |     1022 | D007506 | Iron-Sulfur Proteins    | 0.73  |
|     88 |      13 | thymine                         | CHEMICAL |   338 | 339 |       1696 |     1703 | D013941 | Thymine                 | 1     |
|     57 |       6 | stroke                          | DISEASE  |   208 | 209 |       1260 |     1266 | D020521 | Stroke                  | 1     |

### Abbreviations

``` python
sp_abbrevs = spacyHelp.spacy_get_abbrevs(doc)
```

``` r
reticulate::py$sp_abbrevs |>
  distinct(abrv, .keep_all = T) |>
  sample_n(15) |> knitr::kable()
```

| doc_id | abrv     | start | end | long_form                                                                                                                              |
|---:|:----|---:|--:|:--------------------------------------------------------|
|     38 | N-TIPS   |    19 |  20 | non-solvent-induced phase separation                                                                                                   |
|     25 | TCQ      |   152 | 153 | Tai Chi Quan                                                                                                                           |
|     43 | AEs      |   181 | 182 | adverse events                                                                                                                         |
|     81 | DSP      |   161 | 162 | digit span test                                                                                                                        |
|      9 | iPSCs    |   149 | 150 | induced pluripotent stem cells                                                                                                         |
|     94 | TEPs     |   214 | 215 | transcranial evoked potentials                                                                                                         |
|     10 | SNF      |     6 |   7 | skilled nursing facilities                                                                                                             |
|     90 | VB12     |   115 | 116 | Vitamin B12                                                                                                                            |
|     29 | NSCs     |   255 | 256 | neural stem cells                                                                                                                      |
|    127 | ENS      |   190 | 191 | enteric nervous system                                                                                                                 |
|    151 | autism   |   215 | 216 | neurodegenerative (Alzheimer’s, Parkinson’s), neuropsychiatric (depression , PTSD , schizophrenia ) , and neurodevelopmental disorders |
|    144 | gLFC     |    92 |  93 | Global left frontal cortex                                                                                                             |
|    141 | IVIG     |   300 | 301 | intravenous immunoglobulin                                                                                                             |
|    135 | PHG      |   198 | 199 | parahippocampal gyrus                                                                                                                  |
|    136 | ADAS-cog |   145 | 146 | assessment scale-cognitive subscale                                                                                                    |

### Noun phrases

``` python
sp_noun_phrases = spacyHelp.spacy_get_nps(doc)
```

``` r
reticulate::py$sp_noun_phrases |>
  sample_n(10) |> knitr::kable()
```

| doc_id | sent_id | nounc                        | start | end |
|-------:|--------:|:-----------------------------|------:|----:|
|    150 |       5 | mindfulness                  |   203 | 204 |
|    118 |       8 | support                      |   208 | 209 |
|    114 |      12 | Telehealth                   |   346 | 347 |
|    116 |      15 | Family carers                |   328 | 330 |
|     24 |      13 | adherence                    |   330 | 331 |
|     67 |       4 | light stability              |   127 | 129 |
|     12 |       8 | a high entrapment efficiency |   218 | 222 |
|     13 |       2 | the history                  |    52 |  54 |
|     98 |       1 | (PwD                         |    54 |  56 |
|    111 |       1 | an increased risk            |    37 |  40 |

### Hyponyms

> Works better with nlp.add_pipe(“merge_entities”)

``` python
sp_hearst = spacyHelp.spacy_get_hyponyms(doc)
```

``` r
reticulate::py$sp_hearst |>
  sample_n(15) |> knitr::kable()
```

| doc_id | pred    | sbj                        | obj                         |
|-------:|:--------|:---------------------------|:----------------------------|
|     99 | include | neurodegenerative diseases | neuroinflammation           |
|    171 | such_as | viral diseases             | MERS                        |
|    127 | include | parameters                 | weight                      |
|    182 | such_as | database                   | Wanfang Data                |
|     21 | include | parameters                 | T1/2                        |
|     93 | such_as | variables                  | history                     |
|    127 | include | parameters                 | morphology                  |
|    155 | include | diseases                   | cancer                      |
|    182 | such_as | database                   | Pubmed                      |
|    164 | other   | groups                     | TKRP                        |
|    110 | such_as | attention                  | medication management       |
|    120 | include | dementia development       | Alzheimers disease          |
|    161 | such_as | neurovascular diseases     | stroke                      |
|    155 | include | diseases                   | neurodegenerative disorders |
|    182 | such_as | database                   | CNKI                        |

### Negation

### Sentences

``` python
sp_sentences = spacyHelp.spacy_get_sentences(doc)
```

``` r
reticulate::py$sp_sentences |>
  sample_n(7) |> knitr::kable()
```

| doc_id | sent_id | text                                                                                                                                                                                                             |
|---:|---:|:----------------------------------------------------------------|
|    169 |       5 | Outcomes were measured at baseline and post-treatment.                                                                                                                                                           |
|    144 |       6 | Thirty-one prodromal AD patients were divided into low connection group (LCG) and high connection group (HCG) by the median of gLFC connectivity.                                                                |
|    103 |       1 | Fifty two male APP/PS1 double transgenic AD mice were randomly divided into model, Moxi, Moxi+inhibitor and medication (rapamycin) groups, with 13 mice in each group.                                           |
|     66 |       1 | A total of 46 patients diagnosed with AD between June 1, 2020 and December 31, 2021 were randomized to undergo either 20 Hz rTMS treatment of the left dorsolateral prefrontal cortex (DLPFC) or sham procedure. |
|    145 |      12 | Furthermore, chronotherapy of VRP administration should be consider to achieve best therapeutic efficacy.                                                                                                        |
|    130 |       1 | As therapeutic recourse stagnates, neurodegenerative diseases will cost over a trillion dollars by 2050.                                                                                                         |
|      5 |       5 | A total of 8 different randomized controlled trials with a total sample of 562 non-overlap Alzheimer disease patients between 50-90 years and a mean age of 75.2 ± 3.9 years were eligible for analyses.         |

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
