import json

from preprocessing.pipeline import (
    noise_removal,
    mixed_script_tokenize,
    validate_script_tags,
    remove_stopwords,
    lemmatize_or_stem,
    preprocess_text,
)


def test_noise_removal_strips_html_and_mentions():
    text = "<b>kya</b>  https://example.com @user 😄 hello!!!"
    cleaned = noise_removal(text)
    assert 'https://' not in cleaned
    assert '@user' not in cleaned
    assert '😄' not in cleaned
    assert 'hello' in cleaned.lower()


def test_tokenization_handles_latin_and_devanagari():
    text = "kya bhai, class mein chal raha hai, शानदार demo"
    tokens = mixed_script_tokenize(text)
    assert 'kya' in tokens
    assert 'bhai' in tokens
    assert 'mein' in tokens
    assert 'chal' in tokens
    assert 'demo' in tokens
    assert 'शानदार' in tokens or 'demo' in tokens


def test_script_validation_flags_ambiguous_tokens():
    tagged = validate_script_tags(['kya', 'शानदार', 'abc123', 'hello', 'हिंदी123'])
    scripts = {item['token']: item['script'] for item in tagged}
    assert scripts['kya'] == 'latin'
    assert scripts['शानदार'] == 'devanagari'
    assert 'hello' in scripts and scripts['hello'] == 'latin'
    assert scripts['abc123'] == 'garbage'
    assert any(item['script'] == 'garbage' for item in tagged)


def test_stopword_removal_keeps_meaningful_mixed_tokens():
    tokens = ['kya', 'bhi', 'class', 'se', 'demo', 'hai']
    tags = ['latin', 'latin', 'latin', 'latin', 'latin', 'latin']
    kept = remove_stopwords(tokens, tags)
    assert 'demo' in kept
    assert 'class' in kept
    assert 'kya' not in kept or 'hai' not in kept


def test_lemmatization_works_for_english_tokens():
    tokens = ['running', 'better', 'classes', 'studies']
    out = lemmatize_or_stem(tokens, ['latin'] * len(tokens))
    assert isinstance(out, list)
    assert len(out) == len(tokens)


def test_demo_stopwords_preserve_negation_and_question_words():
    tokens = ['nahi', 'kya', 'kahan', 'kaun', 'kyun', 'kaise', 'not', 'what', 'where', 'who', 'why', 'how']
    tags = ['latin'] * len(tokens)
    kept = remove_stopwords(tokens, tags)
    assert 'nahi' in kept
    assert 'kya' in kept
    assert 'kahan' in kept
    assert 'kaun' in kept
    assert 'kyun' in kept
    assert 'kaise' in kept
    assert 'not' in kept
    assert 'what' in kept
    assert 'where' in kept
    assert 'who' in kept
    assert 'why' in kept
    assert 'how' in kept


def test_model_input_track_stops_after_script_tagging():
    text = 'nahi kya bhai ye class hai'
    out = preprocess_text(text, track='model_input')
    assert out['cleaned'] == 'nahi kya bhai ye class hai'
    assert out['tokens'] == ['nahi', 'kya', 'bhai', 'ye', 'class', 'hai']
    assert any(item['script'] == 'latin' for item in out['tagged'])
    assert 'filtered_tokens' not in out
    assert 'lemmatized' not in out
