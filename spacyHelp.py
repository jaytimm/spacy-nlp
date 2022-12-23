
import spacy
import pandas as pd

#############
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
  


#################  
def spacy_get_entities(docs, linker = False):

    entity_details_dict = {
      "doc_id": [], 
      "sent_id": [], 
      "entity": [],
      "label": [], 
      "start": [], 
      "end": [],
      "start_char": [],
      "end_char": [],
      # "cui": [], 
      # "descriptor": [], 
      # "score": []
      }
      
    for ix, doc in enumerate(docs):
      for sent_i, sent in enumerate(doc.sents):
        for ent in sent.ents:
          entity_details_dict["doc_id"].append(ix)
          entity_details_dict["sent_id"].append(sent_i)
          entity_details_dict["entity"].append(ent.text)
          entity_details_dict["label"].append(ent.label_)
          entity_details_dict["start"].append(ent.start)
          entity_details_dict["end"].append(ent.end)
          entity_details_dict["start_char"].append(ent.start_char)
          entity_details_dict["end_char"].append(ent.end_char)
          
          # if linker = False:
          #   entity_details_dict["cui"].append('')
          #   entity_details_dict["descriptor"].append('')
          #   entity_details_dict["score"].append('')
          #   
          #   else:
          #   score = round(ent._.kb_ents[0][1], 2)
          #   cui = ent._.kb_ents[0][0]
          #   descriptor = linker.umls.cui_to_entity[ent._.kb_ents[0][0]][1]
          #   entity_details_dict["cui"].append(cui)
          #   entity_details_dict["descriptor"].append(descriptor)
          #   entity_details_dict["score"].append(score)
        
    dd = pd.DataFrame.from_dict(entity_details_dict)
    return dd



###################
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



###############
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



###################
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



############
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
  

