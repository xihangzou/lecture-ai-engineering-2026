from app import retrieval


def test_add_neighbors_adds_pages_around_hit_once():
    pages = retrieval.load_pages()
    first, second = pages[10], pages[11]
    hits = [{**first, "distance": 0.1}, {**second, "distance": 0.2}]
    results = retrieval.add_neighbors(hits, window=1)
    ids = [r["page_id"] for r in results]
    assert ids == [pages[9]["page_id"], first["page_id"], second["page_id"], pages[12]["page_id"]]
    assert len(ids) == len(set(ids))
    assert results[0]["match"] == "neighbor" and results[0]["neighbor_of"] == first["page_id"]


def test_add_neighbors_stays_in_same_document_and_can_be_off():
    pages = retrieval.load_pages()
    last = max(i for i, p in enumerate(pages) if p["document_id"] == pages[0]["document_id"])
    hit = {**pages[last], "distance": 0.1}
    results = retrieval.add_neighbors([hit], window=1)
    assert all(r["document_id"] == hit["document_id"] for r in results)
    assert retrieval.add_neighbors([hit], window=0) == [hit]
