#installation
import pip


# pip -install rdflib
# pip install rdfpandas
# pip install pykeen
# pip install sklearn
# pip install numpy

#imports
import rdflib
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pykeen.models import TransE
from pykeen.triples import CoreTriplesFactory, LCWAInstances, TriplesFactory, TriplesNumericLiteralsFactory
import torch
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from rdflib import Graph, plugin, URIRef

def main():
    #Parse DL-Learner carcinogenesis dataset
    g1 = rdflib.Graph()
    g1.parse ('carcinogenesis.owl', format='application/rdf+xml')
    
    #Parse Learning problem 1 train file
    g2 = rdflib.Graph()
    g2.parse ('kg22-carcinogenesis_lps1-train.ttl', format='turtle')
    
    #Parse Learning problem 2 test file
    g3 = rdflib.Graph()
    g3.parse ('kg22-carcinogenesis_lps2-test.ttl', format='turtle')
    
    #Taking triples from graph in an array
    def getTriples(g):
        return np.asarray([[s,p,o] for s,p,o in g])
        
    #Took triples from DL-learner carcinogenesis graph and applied TransE embedding on the triples 
    triples = getTriples(g1)
    triples_factory = TriplesFactory.from_labeled_triples(triples)
    embeded_triples = TransE(triples_factory=triples_factory)    
    
    #After computing the embedding, filtering out Ontology, Class, DatatypeProperty and ObjectProperty from DL-learner carcinogenesis dataset.
    query="""
    select ?s ?p ?o 
    where{
    ?s ?p ?o.
    FILTER NOT EXISTS {
    ?s ?p ?o.
    FILTER (regex(str(?o), "http://www.w3.org/2002/07/owl#Ontology" ) ||
    regex(str(?o), "http://www.w3.org/2002/07/owl#Class" ) ||
    regex(str(?o), "http://www.w3.org/2002/07/owl#ObjectProperty" ) ||
    regex(str(?o), "http://www.w3.org/2002/07/owl#DatatypeProperty" ))}
    }
    """
    result = g1.query(query)
    
    #Calculating relations_ids and entities_ids for the subject, predicate, object in DL-learner carcinogenesis dataset.
    data = []
    
    rel_dict = triples_factory.relation_id_to_label
    ent_dict = triples_factory.entity_id_to_label
    entities_ids = list(ent_dict.keys())
    relations_ids = list(rel_dict.keys())
    entities = list(ent_dict.values())
    relations = list(rel_dict.values())
    for row in result:
        s_id = entities_ids[entities.index(str(row.s))]
        p_id = relations_ids[relations.index(str(row.p))]
        o_id = entities_ids[entities.index(str(row.o))]
        data.extend([[str(row.s),str(row.p),str(row.o),s_id,p_id,o_id]])
    
    #Organize triples and corresponding IDs of DL-learner carcinogenesis dataset to a pandas dataframe for future use. 
    data_df = pd.DataFrame(data, columns = ['s', 'p', 'o','s_id','p_id','o_id'])
    
    #Calculating hrt scores for the triples of DL-learner carcinogenesis dataset
    data_df['embedding'] = data_df.apply(lambda x: embeded_triples.score_hrt(torch.LongTensor(
        [[[x.s_id],[x.p_id], [x.o_id]]])).tolist()[0][0][0],axis=1)
        
    #drop predicate and object columns as part of data preprocessing
    data_df = data_df.drop(['p', 'o', 'p_id', 'o_id'], axis = 1)
    
    #Converting kg22-carcinogenesis_lps1-train graph to dataframe for future use.
    train_data = []
    for s,p,o in g2:
        x = int(str(s).split('_')[1])
        l = s
        train_data.extend([[str(o),str(p), x]])
    train_df = pd.DataFrame(train_data, columns=['s', 'label', 'lp'])
    train_df = train_df[(train_df['label']=='https://lpbenchgen.org/property/includesResource') |  
              (train_df['label']=='https://lpbenchgen.org/property/excludesResource')]
    
    #Join both training and DL-learner dataframes on individual level in order to get embedding for the particular individual.
    final_df = pd.merge(train_df, data_df,how='inner', on='s')
    
    #Converting https://lpbenchgen.org/property/includesResource to 1 and https://lpbenchgen.org/property/excludesResource to 0 for simplification.
    final_df['label'] = final_df['label'].apply(
        lambda x: 1 if (x=='https://lpbenchgen.org/property/includesResource') else 0)
        
    #Training KNN classifier model on our Training data 
    #Each Learning problem(LP) is divided into train and test set in 80:20 ratio.
    classifier = KNeighborsClassifier(n_neighbors=5)
    f1_scores = []
    for i in range(1,51):
        data = final_df[final_df['lp']==i]
        train = data[:int(0.8 * data.shape[0])]
        test = data[int(0.8 * data.shape[0]):]
        X_train = train[['s_id', 'embedding']]
        X_test = test[['s_id', 'embedding']]
        y_train = train['label']
        y_test = test['label']
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        f1_scores.append(f1_score(y_test, y_pred))  
    
    #This is the mean f1 score for our training dataset    
    sum(f1_scores)/len(f1_scores) 
    
    #Converting kg22-carcinogenesis_lps2-test dataset into dataframe for future use.
    test_data = []
    for s,p,o in g3:
        x = int(str(s).split('_')[1])
        test_data.extend([[str(o),str(p), x]])
        
    test_df = pd.DataFrame(test_data, columns=['s', 'label', 'lp'])
    test_df = test_df[(test_df['label']=='https://lpbenchgen.org/property/includesResource') |  
              (test_df['label']=='https://lpbenchgen.org/property/excludesResource')]
              
    #join test dataset and DL-learner carcinogenesis dataset on individual level to get embedding for individual.
    test_final_df = pd.merge(test_df, data_df,how='inner', on='s')
    
    #Converting https://lpbenchgen.org/property/includesResource to 1 and https://lpbenchgen.org/property/excludesResource to 0 for simplification.
    test_final_df['label'] = test_final_df['label'].apply(
        lambda x: 1 if (x=='https://lpbenchgen.org/property/includesResource') else 0)
    
    #Training KNN classifier model on the given testing dataset.
    #The 20% individuals that are not present in the testing dataset LPs are taken from the DL-learner carcinogenesis dataset and
    #labels are predicted for the  non-labeled unclassified individuals.
    classifier = KNeighborsClassifier(n_neighbors=5)
    
    result_df = pd.DataFrame([])
    
    for i in range(51,76):
        data = test_final_df[test_final_df['lp']==i]
        remaining_data_df = data_df.loc[~data_df['s'].isin(data['s'])]
    
        X_train = test_final_df[['s_id', 'embedding']]
        X_test = remaining_data_df[['s_id', 'embedding']]
        y_train = test_final_df['label']
    
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)    
        
        remaining_data_df.loc[:,'label'] = y_pred.tolist()
        remaining_data_df.loc[:,'lp'] = i
        
        result_df = result_df.append(remaining_data_df)   
    
    #The result will only contain the positively classified data, therefore filtering out all other data    
    result_df = result_df[result_df['label'] == 1]
    
    #Converting data back to URI format for creating the output in RDF format
    result_df['lp'] = result_df['lp'].apply(lambda x: "https://lpbenchgen.org/resource/lp_"+str(x))
    result_df.loc[:, 'label']='https://lpbenchgen.org/property/includesResource'
    
    #Taking positively classified triples from dataframe into Graph g
    g = Graph()
    
    pre = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"+"type"
    obj = "https://lpbenchgen.org/class/"+"LearningProblem"
    sub = "https://lpbenchgen.org/resource/lp_"
    for i,row in result_df[['lp','label','s']].iterrows():
        g.add((URIRef(row['lp']), URIRef(row['label']), URIRef(row['s'])))
    for i in range(51,76):
        g.add((URIRef(sub+str(i)), URIRef(pre), URIRef(obj)))
    
    #Adding namespaces and serializing our classified result into turtle format    
    g.namespace_manager.bind("lpres", "https://lpbenchgen.org/resource/")
    g.namespace_manager.bind("carcinogenesis", "http://dl-learner.org/carcinogenesis#")
    g.namespace_manager.bind("lpclass", "https://lpbenchgen.org/class/")
    g.namespace_manager.bind("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    g.namespace_manager.bind("lpprop", "https://lpbenchgen.org/property/")
    
    g.serialize('classification_result.ttl', format = 'turtle')

if __name__ == '__main__':
    # This code won't run if this file is imported.
    main()
