# Some spaCy & scispacy workflows

*Updated: 2023-01-04*

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
import sys
#print(sys.path)
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

| doc_id | sent_id | entity                                                     | label    | start | end | start_char | end_char | uid     | descriptor            | score |
|---:|----:|:------------------------|:----|---:|--:|-----:|----:|:----|:---------|:---|
|     88 |       1 | metabolomic abnormalities                                  | DISEASE  |    38 |  39 |        208 |      233 |         |                       |       |
|    100 |       9 | AD                                                         | DISEASE  |   167 | 168 |       1069 |     1071 | D000544 | Alzheimer Disease     | 1     |
|     31 |       8 | MCI                                                        | DISEASE  |   215 | 216 |       1300 |     1303 | D060825 | Cognitive Dysfunction | 1     |
|    140 |       1 | infections                                                 | DISEASE  |    24 |  25 |        200 |      210 | D007239 | Infections            | 1     |
|     89 |       8 | Aβ1                                                        | CHEMICAL |   353 | 354 |       2008 |     2011 |         |                       |       |
|    171 |       1 | dengue fever                                               | DISEASE  |    37 |  38 |        225 |      237 | D003715 | Dengue                | 1     |
|     91 |       8 | dementia                                                   | DISEASE  |   152 | 153 |        967 |      975 | D003704 | Dementia              | 1     |
|    103 |       0 | Alzheimer’s disease                                        | DISEASE  |    24 |  25 |        193 |      212 | D000544 | Alzheimer Disease     | 1     |
|     94 |       3 | neurological progressive diseases like Alzheimer’s disease | DISEASE  |   102 | 103 |        631 |      689 | D000544 | Alzheimer Disease     | 0.74  |
|    189 |       1 | post-amplifier                                             | CHEMICAL |    92 |  93 |        549 |      563 |         |                       |       |
|    128 |       4 | MCI                                                        | DISEASE  |   110 | 111 |        768 |      771 | D060825 | Cognitive Dysfunction | 1     |
|     87 |       8 | dementia                                                   | DISEASE  |   225 | 226 |       1424 |     1432 | D003704 | Dementia              | 1     |
|     12 |       6 | MDA                                                        | CHEMICAL |   149 | 150 |        963 |      966 | D008315 | Malondialdehyde       | 1     |
|    143 |       3 | Alzheimer’s disease                                        | DISEASE  |    74 |  75 |        451 |      470 | D000544 | Alzheimer Disease     | 1     |
|     36 |       7 | TCM                                                        | DISEASE  |   152 | 153 |       1026 |     1029 |         |                       |       |

### Abbreviations

``` python
sp_abbrevs = spacyHelp.spacy_get_abbrevs(doc)
```

``` r
reticulate::py$sp_abbrevs |>
  distinct(abrv, .keep_all = T) |>
  sample_n(15) |> knitr::kable()
```

| doc_id | abrv  | start | end | long_form                                 |
|-------:|:------|------:|----:|:------------------------------------------|
|     52 | FFI   |   162 | 163 | Fitness Fatness Index                     |
|    140 | GDS   |   202 | 203 | Geriatric Depression Scale                |
|    127 | FMT   |   153 | 154 | fecal microbiota transplantation          |
|     81 | DSP   |   161 | 162 | digit span test                           |
|     29 | BDNF  |   221 | 222 | (IGF-1)/brain-derived neurotrophic factor |
|    174 | RACFs |    55 |  56 | residential aged care facilities          |
|    141 | IVIG  |   191 | 192 | intravenous immunoglobulin                |
|     63 | GTN   |   166 | 167 | gastrocnemius                             |
|     29 | AHN   |   155 | 156 | adult hippocampal neurogenesis            |
|    170 | DASH  |    97 |  98 | Dietary Approaches to Stop Hypertension   |
|     92 | CSF   |     4 |   5 | cerebrospinal fluid                       |
|     52 | METs  |   171 | 172 | metabolic equivalents                     |
|     52 | CRF   |    54 |  55 | cardiorespiratory fitness                 |
|    101 | CCI   |   312 | 313 | Charlson Comorbidity Index                |
|    103 | TFEB  |   354 | 355 | transcription factor EB                   |

### Noun phrases

``` python
sp_noun_phrases = spacyHelp.spacy_get_nps(doc)
```

``` r
reticulate::py$sp_noun_phrases |>
  sample_n(10) |> knitr::kable()
```

| doc_id | sent_id | nounc                   | start | end |
|-------:|--------:|:------------------------|------:|----:|
|     92 |       9 | Hypnosis                |   169 | 170 |
|    172 |      11 | the implementation      |   274 | 276 |
|    168 |       3 | The models              |   117 | 119 |
|    173 |       6 | (MS                     |    98 | 100 |
|    121 |       6 | the accuracy            |   210 | 212 |
|     75 |       6 | that exercise           |   167 | 169 |
|    186 |       5 | The common feature      |   136 | 139 |
|     91 |      11 | areas                   |   216 | 217 |
|    193 |       4 | precuneus               |   103 | 104 |
|    148 |       8 | the blood-brain barrier |   238 | 241 |

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

| doc_id | sbj                                    | pred       | obj                      |
|------:|:--------------------------------|:---------|:---------------------|
|    192 | Brain diseases                         | include    | etc                      |
|     95 | neurodegenerative diseases             | other      | Parkinson’s disease      |
|    126 | phytonutrients                         | such_as    | antioxidants             |
|     93 | variables                              | such_as    | factors                  |
|     67 | Central nervous system (CNS) diseases  | include    | strokes                  |
|     53 | Candidate predictors                   | include    | conditions               |
|    171 | viral diseases                         | such_as    | mumps                    |
|    104 | waste                                  | include    | Aβ                       |
|     40 | neurological and psychiatric disorders | include    | autism spectrum disorder |
|    126 | phytonutrients                         | such_as    | vitamins                 |
|    104 | fluid transport                        | especially | ISF exchange             |
|    110 | attention                              | such_as    | medication management    |
|     93 | variables                              | such_as    | sex                      |
|    101 | covariates                             | include    | histology stage          |
|     33 | outcomes                               | include    | volume                   |

### Negation

### Sentences

``` python
sp_sentences = spacyHelp.spacy_get_sentences(doc)
```

``` r
reticulate::py$sp_sentences |>
  sample_n(7) |> knitr::kable()
```

| doc_id | sent_id | text                                                                                                                                                                                                                                                                                                                                                              |
|--:|--:|:------------------------------------------------------------------|
|      3 |       2 | The trial comprises a 12-month double-blind, placebo-controlled phase followed by a 12-month modified delayed-start open-label treatment phase.                                                                                                                                                                                                                   |
|    159 |       5 | Proportions of participants receiving formal or informal care were reported and associations with QoL were examined using ordinal (self-rated QoL) and linear (EQ-5D) regression.                                                                                                                                                                                 |
|     29 |       8 | Our research suggests that PBMT exerts a beneficial neurogenesis modulatory effect through activating the JAK2/STAT4/STAT5 signaling pathway to promote the expression of IFN-γ/IL-10 in non-parenchymal CD4+ T cells, induction of improvement of brain microenvironmental conditions and alleviation of cognitive deficits in APP/PS1 and 3xTg-AD mouse models. |
|    178 |       1 | Understanding the potential cost-savings or cost-enhancements of Health Information Technology (HIT) can help policymakers understand the capacity of HIT investment to promote population health and health equity for patients with ADRD.                                                                                                                       |
|    169 |       7 | There was no significant difference in the comparison of the primary outcome measures between the groups in post-treatment results (p \> .05); significant differences in all secondary outcome measures were observed in favor of the TR group (p \< .05), except for the OLST, Katz-ADL, and ZCBI (p \> .05).                                                   |
|     19 |       8 | The obligation and toll of giving or receiving caregiving were challenging.                                                                                                                                                                                                                                                                                       |
|     57 |      12 | Only 2 studies described the implementation of a DHI.                                                                                                                                                                                                                                                                                                             |

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
