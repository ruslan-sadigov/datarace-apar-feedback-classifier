import re

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

RANDOM_SEED = 42

TAG_MISSING_VALUE = "missing"

WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (3, 5)
WORD_MIN_DF = 2
CHAR_MIN_DF = 2

BEST_C = 0.5


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_dataframe(df):
    df = df.copy()
    df["feedback"] = df["feedback"].astype(str).apply(clean_text)
    df["tag"] = df["tag"].fillna(TAG_MISSING_VALUE)
    return df


def build_feature_pipeline() -> ColumnTransformer:
    word_tfidf = TfidfVectorizer(
        ngram_range=WORD_NGRAM_RANGE, min_df=WORD_MIN_DF, sublinear_tf=True
    )
    char_tfidf = TfidfVectorizer(
        ngram_range=CHAR_NGRAM_RANGE,
        min_df=CHAR_MIN_DF,
        analyzer="char_wb",
        sublinear_tf=True,
    )
    tag_encoder = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer(
        transformers=[
            ("word_tfidf", word_tfidf, "feedback"),
            ("char_tfidf", char_tfidf, "feedback"),
            ("tag_ohe", tag_encoder, ["tag"]),
        ]
    )


def build_model_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("features", build_feature_pipeline()),
            (
                "classifier",
                LinearSVC(
                    C=BEST_C,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
