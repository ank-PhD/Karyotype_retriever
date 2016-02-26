from pprint import pprint
from src.Karyotype_retriever import Environement

env = Environement('C:\\Users\\Andrei\\Desktop', 'mmc2-karyotypes.csv')
result = env.compute_all_karyotypes()

pprint(result["meta"])
