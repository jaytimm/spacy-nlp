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
|      0 |       4 | immunological diseases | DISEASE  |   130 | 131 |        833 |      855 | D007154 | Immune System Diseases | 1     |
|      0 |       4 | bone diseases          | DISEASE  |   132 | 133 |        857 |      870 | D001847 | Bone Diseases          | 1     |
|      0 |       4 | Alzheimer’s disease    | DISEASE  |   134 | 135 |        875 |      894 | D000544 | Alzheimer Disease      | 1     |
|      1 |       0 | Alzheimer disease      | DISEASE  |     0 |   1 |          0 |       17 | D000544 | Alzheimer Disease      | 1     |
|      1 |       0 | dementia               | DISEASE  |     9 |  10 |         44 |       52 | D003704 | Dementia               | 1     |
|      1 |       1 | acupuncture            | CHEMICAL |    31 |  32 |        174 |      185 | D026881 | Acupuncture            | 1     |
|      1 |       1 | AD                     | DISEASE  |    38 |  39 |        221 |      223 | D000544 | Alzheimer Disease      | 1     |

### Abbreviations

``` python
sp_abbrevs = spacyHelp.spacy_get_abbrevs(doc)
```

``` r
reticulate::py$sp_abbrevs |>
  distinct(abrv, .keep_all = T) |>
  sample_n(15) |> knitr::kable()
```

| doc_id | abrv    | start | end | long_form                                               |
|------:|:-------|-----:|----:|:----------------------------------------------|
|    172 | SCU-B   |    44 |  45 | special care unit for people with dementia and BPSD     |
|    120 | AH      |    62 |  63 | Arterial hypertension                                   |
|     55 | HD-tDCS |   193 | 194 | high-definition transcranial direct current stimulation |
|      8 | AVRT    |   110 | 111 | atrioventricular reentrant tachycardia                  |
|    136 | PSQI    |   154 | 155 | Pittsburgh sleep quality index                          |
|     52 | METs    |   171 | 172 | metabolic equivalents                                   |
|     60 | CNS     |   169 | 170 | central nervous system                                  |
|    138 | PVA     |    47 |  48 | polyvinyl alcohol                                       |
|     31 | tDCS    |    72 |  73 | Transcranial Direct Current Stimulation                 |
|     15 | ADL     |   199 | 200 | activities of daily living scale                        |
|    115 | ANP     |    25 |  26 | Advanced nursing practice                               |
|    107 | PPHs    |    80 |  81 | potentially preventable hospitalizations                |
|    161 | IR      |   184 | 185 | ischemia reperfusion                                    |
|    145 | CREB    |   257 | 258 | cAMP response element-binding protein                   |
|    144 | gLFC    |   126 | 127 | Global left frontal cortex                              |

### Noun phrases

``` python
sp_noun_phrases = spacyHelp.spacy_get_nps(doc)
```

``` r
reticulate::py$sp_noun_phrases |>
  sample_n(10) |> knitr::kable()
```

| doc_id | sent_id | nounc                             | start | end |
|-------:|--------:|:----------------------------------|------:|----:|
|     28 |       0 | multiple disease mechanisms       |    12 |  14 |
|    104 |       0 | a crucial role                    |     6 |   9 |
|    181 |       8 | who                               |   204 | 205 |
|     88 |       3 | (MCI                              |    79 |  81 |
|    185 |       5 | DNTPH                             |   100 | 101 |
|     40 |       0 | Transcranial magnetic stimulation |     0 |   3 |
|    115 |       5 | multimodal strategies             |   151 | 153 |
|     54 |       5 | assistance                        |   262 | 263 |
|     85 |       9 | these nanoformulations            |   182 | 184 |
|    137 |       4 | Results                           |   115 | 116 |

### Hyponyms

> Works better with nlp.add_pipe(“merge_entities”)

``` python
sp_hearst = spacyHelp.spacy_get_hyponyms(doc)
```

``` r
reticulate::py$sp_hearst |>
  sample_n(15) |> knitr::kable()
```

| doc_id | pred       | sbj                                    | obj                      |
|------:|:---------|:--------------------------------|:---------------------|
|     19 | include    | CPs                                    | care fragmentation       |
|    101 | include    | covariates                             | status                   |
|    182 | such_as    | database                               | Pubmed                   |
|     83 | include    | neurological diseases                  | autism spectrum disorder |
|    139 | especially | medicines                              | Alzheimer’s disease      |
|     40 | include    | neurological and psychiatric disorders | depression               |
|    171 | such_as    | viral diseases                         | AIDS                     |
|     19 | include    | distress                               | loss                     |
|    101 | include    | covariates                             | cancer site              |
|     53 | such_as    | decisions                              | cancer screening         |
|    171 | such_as    | viral diseases                         | polio                    |
|     93 | such_as    | variables                              | ethnicity                |
|     69 | such_as    | agents                                 | growth factors           |
|     87 | include    | treatments                             | tube feeding             |
|    127 | include    | parameters                             | weight                   |

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
