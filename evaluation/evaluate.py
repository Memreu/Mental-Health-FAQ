"""Geri getirme (retrieval) kalitesini elle yazılmış bir soru setiyle ölçer.

Kullanım:
    python evaluation/evaluate.py              # yerel mod (TF-IDF), API anahtarı gerekmez
    GOOGLE_API_KEY=... python evaluation/evaluate.py --mode gemini

Metrikler (yalnızca kapsam içindeki sorular için):
    hit@1, hit@3 : doğru SSS kaydı ilk 1 / ilk 3 benzersiz sonuçta mı?
    MRR          : doğru kaydın sırasının tersinin ortalaması
Kapsam dışı sorular için yerel modda "bilgi yok" deme oranı raporlanır.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag import LOCAL_MIN_SCORE, build_pipeline  # noqa: E402


def evaluate(mode: str, k: int = 10) -> pd.DataFrame:
    api_key = os.environ.get("GOOGLE_API_KEY") if mode == "gemini" else None
    if mode == "gemini" and not api_key:
        sys.exit("Gemini modu için GOOGLE_API_KEY ortam değişkenini tanımla.")
    retriever = build_pipeline(api_key).retriever
    data = pd.read_csv(Path(__file__).with_name("eval_set.csv"), dtype=str).fillna("")
    rows = []
    for r in data.itertuples(index=False):
        hits = retriever.search(r.query, k)
        ranked = list(dict.fromkeys(str(d.metadata["question_id"]) for d, _ in hits))
        relevant = set(filter(None, r.relevant_ids.split(";")))
        rank = next((i for i, q in enumerate(ranked, 1) if q in relevant), None)
        rows.append({"query": r.query, "lang": r.lang, "in_scope": bool(relevant),
                     "rank": rank, "top_score": hits[0][1] if hits else 0.0})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, mode: str) -> str:
    lines = [f"Mod: {mode}", "", "| Dil | Soru | hit@1 | hit@3 | MRR |", "|---|---:|---:|---:|---:|"]
    ins = df[df.in_scope]
    for lang, g in [("en", ins[ins.lang == "en"]), ("tr", ins[ins.lang == "tr"]), ("tümü", ins)]:
        if g.empty:
            continue
        h1 = (g["rank"] == 1).mean(); h3 = (g["rank"] <= 3).mean()
        mrr = g["rank"].map(lambda x: 1 / x if pd.notna(x) else 0).mean()
        lines.append(f"| {lang} | {len(g)} | {h1:.2f} | {h3:.2f} | {mrr:.2f} |")
    out = df[~df.in_scope]
    if mode == "yerel" and not out.empty:
        refused = (out.top_score < LOCAL_MIN_SCORE).mean()
        en = ins[ins.lang == "en"]
        false_refusal = (en.top_score < LOCAL_MIN_SCORE).mean()
        lines += ["", f"Kapsam dışı sorularda 'bilgi yok' deme oranı: {refused:.2f} ({len(out)} soru)",
                  f"Kapsam içi İngilizce sorularda yanlışlıkla 'bilgi yok' deme oranı: {false_refusal:.2f}"]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["yerel", "gemini"], default="yerel")
    args = ap.parse_args()
    res = evaluate(args.mode)
    print(summarize(res, args.mode))
    misses = res[res.in_scope & (res["rank"].isna() | (res["rank"] > 3))]
    if not misses.empty:
        print("\nİlk 3'te bulunamayanlar:")
        for q in misses["query"]:
            print(" -", q)
