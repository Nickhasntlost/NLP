"""
Pure-Python IndicProcessor - ported from IndicTransToolkit/processor.pyx.
All Cython annotations removed; faithful logic. 
Dependencies: indicnlp, sacremoses, regex, tqdm
"""
import regex as re
from queue import Queue
from tqdm import tqdm
from typing import List
from indicnlp.tokenize import indic_tokenize, indic_detokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from sacremoses import MosesPunctNormalizer, MosesTokenizer, MosesDetokenizer
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

FLORES_CODES = {
    "asm_Beng":"as","awa_Deva":"hi","ben_Beng":"bn","bho_Deva":"hi","brx_Deva":"hi",
    "doi_Deva":"hi","eng_Latn":"en","gom_Deva":"kK","gon_Deva":"hi","guj_Gujr":"gu",
    "hin_Deva":"hi","hne_Deva":"hi","kan_Knda":"kn","kas_Arab":"ur","kas_Deva":"hi",
    "kha_Latn":"en","lus_Latn":"en","mag_Deva":"hi","mai_Deva":"hi","mal_Mlym":"ml",
    "mar_Deva":"mr","mni_Beng":"bn","mni_Mtei":"mni","npi_Deva":"ne","ory_Orya":"or",
    "pan_Guru":"pa","san_Deva":"sa","sat_Olck":"sat","snd_Arab":"ur","snd_Deva":"hi",
    "tam_Taml":"ta","tel_Telu":"te","urd_Arab":"ur",
}
INDIC_NUMERAL_MAP = {
    "\u0966":"0","\u0967":"1","\u0968":"2","\u0969":"3","\u096a":"4","\u096b":"5",
    "\u096c":"6","\u096d":"7","\u096e":"8","\u096f":"9","\u09e6":"0","\u09e7":"1",
    "\u09e8":"2","\u09e9":"3","\u09ea":"4","\u09eb":"5","\u09ec":"6","\u09ed":"7",
    "\u09ee":"8","\u09ef":"9",
}
PUNC_REPLACEMENTS = [
    ("\u0964","."),("\u0965",".."),("\u2018","'"),("\u2019","'"),
    ("\u201c",'"'),("\u201d",'"'),("\u2013","-"),("\u2014","-"),("\u2026","..."),
]
_URL_PAT = re.compile(r"(?:https?://|www\.)\S+|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_NUM_PAT = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_MSPACE  = re.compile(r" +")

class IndicProcessor:
    def __init__(self, inference=True):
        self.inference = inference
        self._flores_codes = FLORES_CODES
        self._digits_table = str.maketrans(INDIC_NUMERAL_MAP)
        self._placeholder_maps = Queue()
        self._xliterator = UnicodeIndicTransliterator()
        self._en_tok = MosesTokenizer(lang="en")
        self._en_norm = MosesPunctNormalizer()
        self._en_detok = MosesDetokenizer(lang="en")

    def _punc_norm(self, text):
        for old, new in PUNC_REPLACEMENTS:
            text = text.replace(old, new)
        return text

    def _wrap_placeholders(self, text):
        pmap = {}
        serial = 0
        for pat, tag in [(_URL_PAT,"URL"),(_NUM_PAT,"NUM")]:
            for m in pat.findall(text):
                base = f"[{tag}{serial}]"
                for form in [base,f"[{tag} {serial}]",f"[ {tag}{serial} ]",
                              f"[ {tag} {serial} ]",f"{tag} {serial}",f"{tag}{serial}"]:
                    pmap[form] = m
                text = text.replace(m, base, 1)
                serial += 1
        text = _MSPACE.sub(" ", text).replace(">/",">").replace("]/","]")
        self._placeholder_maps.put(pmap)
        return text

    def _normalize(self, text):
        text = text.translate(self._digits_table)
        if self.inference:
            text = self._wrap_placeholders(text)
        return text

    def _indic_tok_xliterate(self, sent, normalizer, iso_lang, transliterate):
        normed = normalizer.normalize(sent.strip())
        tokens = indic_tokenize.trivial_tokenize(normed, iso_lang)
        joined = " ".join(tokens)
        if transliterate:
            joined = self._xliterator.transliterate(joined, iso_lang, "hi")
        return joined

    def _preprocess(self, sent, src_lang, tgt_lang, normalizer, is_target):
        iso_lang = self._flores_codes.get(src_lang, "hi")
        script   = src_lang.split("_")[1]
        do_xlit  = script not in ("Arab","Aran","Olck","Mtei","Latn")
        sent = self._punc_norm(sent)
        sent = self._normalize(sent)
        if iso_lang == "en":
            normed = self._en_norm.normalize(sent.strip())
            processed = " ".join(self._en_tok.tokenize(normed, escape=False))
        else:
            processed = self._indic_tok_xliterate(sent, normalizer, iso_lang, do_xlit)
        processed = processed.strip()
        return processed if is_target else f"{src_lang} {tgt_lang} {processed}"

    def _postprocess(self, sent, lang, pmap=None):
        if isinstance(sent,(tuple,list)): sent = sent[0]
        if pmap is None: pmap = self._placeholder_maps.get()
        lang_code, script = lang.split("_",1)
        iso_lang = self._flores_codes.get(lang,"hi")
        if script in ("Arab","Aran"):
            sent = sent.replace(" \u06cc","\u06cc").replace(" \u060c","\u060c")
        for k,v in pmap.items():
            sent = sent.replace(k,v)
        if lang == "eng_Latn":
            return self._en_detok.detokenize(sent.split())
        xlated = self._xliterator.transliterate(sent,"hi",iso_lang)
        return indic_detokenize.trivial_detokenize(xlated,iso_lang)

    def preprocess_batch(self, batch, src_lang, tgt_lang=None, is_target=False, visualize=False):
        iso = self._flores_codes.get(src_lang,"hi")
        normalizer = None if src_lang=="eng_Latn" else IndicNormalizerFactory().get_normalizer(iso)
        it = tqdm(batch,desc=f" | > Pre-processing {src_lang}") if visualize else batch
        return [self._preprocess(s,src_lang,tgt_lang,normalizer,is_target) for s in it]

    def postprocess_batch(self, sents, lang="hin_Deva", visualize=False, num_return_sequences=1):
        pmaps = [self._placeholder_maps.get() for _ in range(len(sents)//num_return_sequences)]
        self._placeholder_maps.queue.clear()
        it = tqdm(enumerate(sents),total=len(sents)) if visualize else enumerate(sents)
        results = []
        for i,s in it:
            results.append(self._postprocess(s,lang,pmaps[i//num_return_sequences]))
        return results
