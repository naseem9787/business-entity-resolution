"""Exp3 copy of norm.py with ONE change: the tokenizer keeps Indic combining marks (U+0900-U+0DFF) so native-script words are not shattered."""
import re, unicodedata
from unidecode import unidecode
_TOK=re.compile(r'(?:[^\W_]|[ऀ-෿])+',re.U)   # Exp3 fix: keep Indic vowel signs / viramas (Mn,Mc) inside words
NOISE={'null','nan','n','a','na','none'}
def clean(s):
    """unicode-normalised, lowercased, accents stripped only on Latin letters, junk removed (keeps Indic scripts intact)."""
    if not s: return ''
    s=unicodedata.normalize('NFKC',s).replace('�','')
    if not s.isascii():
        out=[]
        for ch in unicodedata.normalize('NFKD',s):
            if unicodedata.combining(ch) and out and ord(out[-1])<0x250: continue
            out.append(ch)
        s=''.join(out)
    s=s.lower()
    return ' '.join(_TOK.findall(s))
def translit(s):
    return s if s.isascii() else ' '.join(_TOK.findall(unidecode(s).lower()))
LEGAL=set('inc incorporated llc ltd limited private pvt corp corporation co company llp lp plc sarl sas sa sci eurl sasu gmbh ag the of and et de la le les du des amp smt shri sri m s'.split())
def core(tokens): 
    t=[x for x in tokens if x not in LEGAL]
    return t or tokens

# ---- address canonicalisation (language-generic abbreviation folding; no entity lookups) ----
ABBR={}
for canon,vs in {'st':'street st saint str','rd':'road rd','ave':'avenue ave av','dr':'drive dr','ln':'lane ln','blvd':'boulevard blvd bd',
 'ct':'court ct','pl':'place pl','hwy':'highway hwy','pkwy':'parkway pkwy','cir':'circle cir','ter':'terrace ter','flr':'floor flr fl',
 'apt':'apartment apt','ste':'suite ste','bldg':'building bldg','n':'north n','s':'south s','e':'east e','w':'west w','no':'no number'}.items():
    for v in vs.split(): ABBR[v]=canon
US_ST=dict(alabama='al',alaska='ak',arizona='az',arkansas='ar',california='ca',colorado='co',connecticut='ct',delaware='de',florida='fl',georgia='ga',hawaii='hi',idaho='id',illinois='il',indiana='in',iowa='ia',kansas='ks',kentucky='ky',louisiana='la',maine='me',maryland='md',massachusetts='ma',michigan='mi',minnesota='mn',mississippi='ms',missouri='mo',montana='mt',nebraska='ne',nevada='nv',ohio='oh',oklahoma='ok',oregon='or',pennsylvania='pa',tennessee='tn',texas='tx',utah='ut',vermont='vt',virginia='va',washington='wa',wisconsin='wi',wyoming='wy')
US2={'new hampshire':'nh','new jersey':'nj','new mexico':'nm','new york':'ny','north carolina':'nc','north dakota':'nd','rhode island':'ri','south carolina':'sc','south dakota':'sd','west virginia':'wv'}
IN_ST={'uttar pradesh':'up','madhya pradesh':'mp','andhra pradesh':'ap','tamil nadu':'tn','west bengal':'wb','himachal pradesh':'hp','arunachal pradesh':'ar','jammu and kashmir':'jk','jammu kashmir':'jk',
 'maharashtra':'mh','rajasthan':'rj','gujarat':'gj','delhi':'dl','haryana':'hr','karnataka':'ka','telangana':'ts','kerala':'kl','punjab':'pb','bihar':'br','odisha':'od','orissa':'od','assam':'as','jharkhand':'jh','chhattisgarh':'cg','uttarakhand':'uk','goa':'ga'}
PHR={**US2,**{k:v for k,v in IN_ST.items() if ' ' in k}}
SING={**US_ST,**{k:v for k,v in IN_ST.items() if ' ' not in k}}
_PH=re.compile('|'.join(sorted(map(re.escape,PHR),key=len,reverse=True)))
def addr_canon(a):
    """a = clean/translit address string -> canonical token list"""
    a=_PH.sub(lambda m:PHR[m.group()],a)
    out=[]
    for t in a.split():
        t=ABBR.get(t,t); t=SING.get(t,t)
        if t in NOISE: continue
        out.append(t)
    return out
