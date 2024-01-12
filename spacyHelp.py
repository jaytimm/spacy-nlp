
import spacy
import pandas as pd


def spacy_extract_entities(docs, linker):
    # Initialize lists to store entity and linking information
    entity_details = {
        "doc_id": [],
        "sent_id": [],
        "entity": [],
        "label": [],
        "start": [],
        "end": [],
        "start_char": [],
        "end_char": [],
        "uid": [],
        "descriptor": [],
        "score": []
    }

    for doc_id, doc in enumerate(docs):
        for sent_id, sent in enumerate(doc.sents):
            for ent in sent.ents:
                # Entity details
                entity_details["doc_id"].append(doc_id)
                entity_details["sent_id"].append(sent_id)
                entity_details["entity"].append(ent.text)
                entity_details["label"].append(ent.label_)
                entity_details["start"].append(ent.start)
                entity_details["end"].append(ent.end)
                entity_details["start_char"].append(ent.start_char)
                entity_details["end_char"].append(ent.end_char)

                # Entity linking
                if hasattr(ent._, 'kb_ents') and ent._.kb_ents:
                    ee = ent._.kb_ents
                    entity_details["uid"].append(ee[0][0])
                    entity_details["score"].append(round(ee[0][1], 2))
                    entity_details["descriptor"].append(linker.kb.cui_to_entity[ee[0][0]][1])
                else:
                    entity_details["uid"].append(None)
                    entity_details["descriptor"].append(None)
                    entity_details["score"].append(None)

    return pd.DataFrame(entity_details)

# Usage
# docs = list(nlp.pipe(your_texts))
# linker = your_entity_linker
# combined_df = extract_entities(many_docs, linker)



def spacy_extract_nps(docs):
    # Pre-allocate lists for storing noun chunk details
    doc_ids, sent_ids, noun_chunks, starts, ends = [], [], [], [], []

    # Iterating over each document and sentence
    for doc_id, doc in enumerate(docs):
        for sent_id, sent in enumerate(doc.sents):
            for nc in sent.noun_chunks:
                # Appending details directly to lists
                doc_ids.append(doc_id)
                sent_ids.append(sent_id)
                noun_chunks.append(nc.text)
                starts.append(nc.start)
                ends.append(nc.end)

    # Creating a DataFrame directly from the lists
    return pd.DataFrame({
        "doc_id": doc_ids,
        "sent_id": sent_ids,
        "nounc": noun_chunks,
        "start": starts,
        "end": ends
    })

# Usage
# docs = list(nlp.pipe(your_texts))
# noun_chunks_df = spacy_get_nps(many_docs)





def spacy_extract_sentences(docs):
    # Pre-allocate lists for storing sentence details
    doc_ids, sent_ids, texts = [], [], []

    # Iterating over each document and sentence
    for doc_id, doc in enumerate(docs):
        for sent_id, sent in enumerate(doc.sents):
            doc_ids.append(doc_id)
            sent_ids.append(sent_id)
            texts.append(sent.text.strip())  # Simplified text extraction

    # Creating a DataFrame directly from the lists
    return pd.DataFrame({
        "doc_id": doc_ids,
        "sent_id": sent_ids,
        "text": texts
    })

# Usage
# docs = list(nlp.pipe(your_texts))
# sentences_df = spacy_get_sentences(many_docs)




def spacy_extract_hyponyms(docs):
    # Pre-allocate lists for storing hyponym details
    doc_ids, preds, sbjs, objs = [], [], [], []

    # Iterating over each document and hearst pattern
    for doc_id, doc in enumerate(docs):
        for ht in doc._.hearst_patterns:
            doc_ids.append(doc_id)
            preds.append(ht[0])

            # Simplified text extraction for subject and object
            sbj = ht[1].text
            obj = ht[2].text

            sbjs.append(sbj)
            objs.append(obj)

    # Creating a DataFrame directly from the lists
    return pd.DataFrame({
        "doc_id": doc_ids,
        "pred": preds,
        "sbj": sbjs,
        "obj": objs
    })

# Usage
# docs = list(nlp.pipe(your_texts))
# hyponyms_df = spacy_get_hyponyms(many_docs)


def spacy_extract_df(docs):
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
    
    all_data = []
    for doc_id, doc in enumerate(docs):
        for token in doc:
            sent_id = token.sent.start
            token_data = (
                doc_id,
                token.text,
                token.i,
                sent_id,
                token.lemma_,
                token.ent_type_,
                token.tag_,
                token.dep_,
                token.pos_,
                token.is_stop,
                token.is_alpha,
                token.is_digit,
                token.is_punct
            )
            all_data.append(token_data)
    
    # Create DataFrame in one go
    meta_df = pd.DataFrame(all_data, columns=cols)
    return meta_df

# Usage
# docs = list(nlp.pipe(your_texts))
# df = spacy_get_df(many_docs)





###################
def spacy_extract_abbrevs(docs):

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

  
################
def spacy_get_matches(docs):
  
  details_dict = {
  "doc_id": [], 
  "string_id": [],
  "match": [], 
  "start": [], 
  "end": []
  }
  
  for ix, dx in enumerate(docs):
    
    matches = matcher(dx)
    
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id] 
        span = dx[start:end]          
        details_dict["doc_id"].append(ix)
        details_dict["string_id"].append(string_id)
        details_dict["match"].append(span.text)
        details_dict["start"].append(start)
        details_dict["end"].append(end)
      
  dd =  pd.DataFrame.from_dict(details_dict)     
  return dd


