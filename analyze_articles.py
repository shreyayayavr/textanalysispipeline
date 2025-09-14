# scripts/analyze_articles.py
"""
analyze_articles.py
This script analyzes pre-extracted text articles (from extract_articles.py) and computes
various linguistic, readability, and sentiment metrics.
Input:
    - Excel file containing URL_ID (to locate article text files)
    - Folder containing article .txt files
    - Stopwords directory
    - MasterDictionary directory (positive/negative words)
Output:
    - Excel or CSV file with computed metrics appended to the original input data.
"""

import argparse, re
from pathlib import Path
import pandas as pd
import nltk
from textstat import textstat
nltk.download('punkt', quiet=True)

# Define vowels for syllable counting
VOWELS = set("aeiou")

def load_stopwords(stop_dir):
    """
    Load all stopwords from files in the given directory (recursively).
    Normalizes them to lowercase and strips non-alphabetic characters.
    """
    sw = set()
    for p in Path(stop_dir).glob("**/*"):
        if p.is_file():
            txt = p.read_text(encoding="latin-1", errors="ignore")
            for line in txt.splitlines():
                w = re.sub(r"[^A-Za-z]", " ", line).strip().lower()
                if not w:
                    continue
                for token in w.split():
                    sw.add(token)
    return sw

def load_dict(dict_dir, stopwords):
    """
    load positive and negative words from the MasterDictionary directory.
    filters out any stopwords to avoid overlap.
    returns:
        pos (set): positive words
        neg (set): negative words
    """
    pos = set()
    neg = set()
    for name, bucket in [("positive-words", pos), ("negative-words", neg)]:
        # Match any file whose name contains 'positive-words' or 'negative-words'
        files = list(Path(dict_dir).glob(f"*{name}*"))
        if not files:
            continue
        txt = "\n".join(f.read_text(encoding="latin-1", errors="ignore") for f in files)
        for line in txt.splitlines():
            w = re.sub(r"[^A-Za-z]", "", line).strip().lower()
            if w and w not in stopwords:
                bucket.add(w)
    return pos, neg


def sentences(text):
    """return a list of sentence strings from the given text"""
    return nltk.sent_tokenize(text)

def tokens_alpha(text):
    """return a list of alphabetic-only tokens in lowercase"""
    return [t.lower() for t in re.findall(r"[A-Za-z]+", text)]


def syllables_in_word(w):
    """approximate the number of syllables in a word.
    uses a simple vowel group counting heuristic with adjustments for common endings.
    """
    w = w.lower()
    if not w:
        return 0
    count = 0
    prev_is_vowel = False
    for ch in w:
        is_v = ch in VOWELS
        if is_v and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_v
    # Adjust for silent endings like 'es' and 'ed'
    if w.endswith(("es", "ed")) and count > 1:
        count -= 1
    return max(1, count)


def personal_pronouns_count(text):
    """Count personal pronouns (I, we, my, ours, us) in text
    excludes the country name 'US'
    """
    toks = re.findall(r"\b[A-Za-z]+\b", text)
    cnt = 0
    for t in toks:
        if t == "US":  # Ignore the country
            continue
        tl = t.lower()
        if tl in {"i", "we", "my", "ours", "us"}:
            cnt += 1
    return cnt

def analyze_text(raw_text, stopwords, pos_dict, neg_dict):
    """
    perform sentiment and readability analysis on a given text string
    Metrics:
        - POSITIVE SCORE / NEGATIVE SCORE
        - POLARITY SCORE / SUBJECTIVITY SCORE
        - AVG SENTENCE LENGTH / PERCENTAGE OF COMPLEX WORDS / FOG INDEX
        - COMPLEX WORD COUNT / WORD COUNT / SYLLABLE PER WORD
        - PERSONAL PRONOUNS / AVG WORD LENGTH
    """
    sents = sentences(raw_text)
    sent_count = max(1, len(sents))  # avoid division by zero
    toks = tokens_alpha(raw_text)

    # Remove stopwords for analysis
    clean = [w for w in toks if w not in stopwords]
    total_words = len(clean)

    # Sentiment scores
    pos_score = sum(1 for w in clean if w in pos_dict)
    neg_score = sum(1 for w in clean if w in neg_dict)
    polarity = (pos_score - neg_score) / ((pos_score + neg_score) + 1e-6)
    subjectivity = (pos_score + neg_score) / (total_words + 1e-6)

    # Complex words & syllables
    syll_counts = [syllables_in_word(w) for w in clean]
    complex_words = sum(1 for s in syll_counts if s > 2)
    pct_complex = (complex_words / total_words) if total_words else 0.0

    # Length-based metrics
    avg_sentence_len = (total_words / sent_count) if sent_count else 0.0
    fog = 0.4 * (avg_sentence_len + pct_complex)
    avg_words_per_sentence = avg_sentence_len
    syllables_per_word = (sum(syll_counts) / total_words) if total_words else 0.0
    avg_word_len = (sum(len(w) for w in clean) / total_words) if total_words else 0.0

    # Pronouns
    pronouns = personal_pronouns_count(raw_text)

    return {
        "POSITIVE SCORE": pos_score,
        "NEGATIVE SCORE": neg_score,
        "POLARITY SCORE": polarity,
        "SUBJECTIVITY SCORE": subjectivity,
        "AVG SENTENCE LENGTH": avg_sentence_len,
        "PERCENTAGE OF COMPLEX WORDS": pct_complex,
        "FOG INDEX": fog,
        "AVG NUMBER OF WORDS PER SENTENCE": avg_words_per_sentence,
        "COMPLEX WORD COUNT": complex_words,
        "WORD COUNT": total_words,
        "SYLLABLE PER WORD": syllables_per_word,
        "PERSONAL PRONOUNS": pronouns,
        "AVG WORD LENGTH": avg_word_len,
    }

def main(input_xlsx, articles_dir, stop_dir, dict_dir, output_path):
    """
   analysis pipeline:
        1.Load stopwords
        2.Load positive/negative word dictionaries
        3.Read the input Excel file
        4.For each URL_ID, find the corresponding article text
        5.Compute metrics and store results
        6.Save results to Excel
    """
    stopwords = load_stopwords(stop_dir)
    pos_dict, neg_dict = load_dict(dict_dir, stopwords)
    df_in = pd.read_excel(input_xlsx)

    rows = []
    for _, row in df_in.iterrows():
        url_id = str(row["URL_ID"])
        p = Path(articles_dir) / f"{url_id}.txt"
        if not p.exists():
            # If article file is missing, fill with None values
            metrics = {k: None for k in [
                "POSITIVE SCORE", "NEGATIVE SCORE", "POLARITY SCORE", "SUBJECTIVITY SCORE",
                "AVG SENTENCE LENGTH", "PERCENTAGE OF COMPLEX WORDS", "FOG INDEX",
                "AVG NUMBER OF WORDS PER SENTENCE", "COMPLEX WORD COUNT", "WORD COUNT",
                "SYLLABLE PER WORD", "PERSONAL PRONOUNS", "AVG WORD LENGTH"
            ]}
        else:
            text = p.read_text(encoding="utf-8", errors="ignore")
            metrics = analyze_text(text, stopwords, pos_dict, neg_dict)

        outrow = dict(row)  # Keep original input columns
        outrow.update(metrics)  # Add computed metrics
        rows.append(outrow)

    df_out = pd.DataFrame(rows)

    # Save results in desired format
    if output_path.lower().endswith(".csv"):
        df_out.to_csv(output_path, index=False)
    else:
        df_out.to_excel(output_path, index=False)
    print("saved:", output_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input Excel file with URL_IDs")
    ap.add_argument("--articles_dir", default="articles", help="Directory with article .txt files")
    ap.add_argument("--stopwords_dir", default="data/StopWords", help="Directory containing stopword files")
    ap.add_argument("--dict_dir", default="data/MasterDictionary", help="Directory with positive/negative word lists")
    ap.add_argument("--output", default="output/output.xlsx", help="Output Excel or CSV file path")
    args = ap.parse_args()

    main(args.input, args.articles_dir, args.stopwords_dir, args.dict_dir, args.output)
