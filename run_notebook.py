import os,sys,json,time
from pathlib import Path
p=Path.cwd()
os.environ.setdefault('MPLCONFIGDIR',str(p/'cache'/'matplotlib'))
os.environ.setdefault('HF_HOME',str(p/'cache'/'huggingface'))
os.environ.setdefault('JUPYTER_PATH',str(p/'.jupyter'))
os.environ.setdefault('IPYTHONDIR',str(p/'cache'/'ipython'))
os.environ.setdefault('NUMBA_CACHE_DIR',str(p/'cache'/'numba'))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
k=p/'.jupyter/kernels/python3'; k.mkdir(parents=True,exist_ok=True)
(k/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Grammar project Python','language':'python'}))
import nbformat
from nbclient import NotebookClient
path=p/'grammar_scoring_engine.ipynb'
nb=nbformat.read(path,as_version=4)
nbformat.validate(nb)
def save(**kwargs):
    nbformat.write(nb,path)
    print('Completed cell',kwargs.get('cell_index'),time.strftime('%H:%M:%S'),flush=True)
client=NotebookClient(nb,timeout=None,kernel_name='python3',resources={'metadata':{'path':str(p)}},on_cell_executed=save)
try: client.execute()
finally: nbformat.write(nb,path)
print('Notebook execution complete',flush=True)
