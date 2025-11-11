# This script appends semantic groups to UMLS/SNOMEDCT-coded entities
# input:
# SEM_SNOMED: Dict[str, str] scui, semgroup
# SEM_UMLS: Dict[str,str] cui, semgroup
# texts: csv -> List[Dict[str,str]] id,text
# annotations: csv -> List[Dict[str,str]] id,CUI,from,to,value
# ISA
# Hypernyms
