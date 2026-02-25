from app.rag import split_text


def test_split_text_produces_chunks():
    text = "A" * 1200
    chunks = split_text(text, chunk_size=500, overlap=100)
    assert len(chunks) >= 2
    assert all(chunks)


def test_split_text_rejects_invalid_overlap():
    try:
        split_text("abc", chunk_size=10, overlap=10)
        assert False, "Expected ValueError"
    except ValueError:
        assert True
