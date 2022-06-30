# ARM-KG-Mini-Project

NOTE: Our program needs following files for the data:
carcinogenesis.owl
kg22-carcinogenesis_lps1-train.ttl
kg22-carcinogenesis_lps2-test.ttl

Following installations need to be done:
 pip install rdflib
 pip install rdfpandas
 pip install pykeen
 pip install sklearn
 pip install pandas
 pip install numpy
 
 The main.py contains only one function i.e. main().
 
 The main() function will execute complete code and generate output file - classification_result.ttl in the same folder.
 The classification_result.ttl file contains the positively predicted data (The unclassified data that is present in carcinogenesis.owl 
 but not present in kg22-carcinogenesis_lps2-test.ttl is given to the model for prediction).
