from pprint import pprint
from src.Karyotype_retriever import Environement
from pickle import dump

env = Environement('C:\\Users\\Andrei\\Desktop', 'mmc2-karyotypes.csv')
result = env.compute_all_karyotypes()

pprint(result["meta"])

dump(result["meta"], open('CNV.dmp', 'w'))
