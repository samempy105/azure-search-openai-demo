# --- in retrievethenread.py ---

async def run_search_approach(self, messages, overrides, auth_claims):
    use_text_search = overrides.get("retrieval_mode") in ["text", "hybrid", None]
    use_vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
    use_semantic_ranker = True if overrides.get("semantic_ranker") else False
    use_query_rewriting = True if overrides.get("query_rewriting") else False
    use_semantic_captions = True if overrides.get("semantic_captions") else False
    top = overrides.get("top", 3)
    minimum_search_score = overrides.get("minimum_search_score", 0.0)
    minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)
    filter = self.build_filter(overrides, auth_claims)
    q = str(messages[-1]["content"])

    # If retrieval mode includes vectors, compute an embedding for the query
    vectors = []
    if use_vector_search:
        vectors.append(await self.compute_text_embedding(q))

    results = await self.search(
        top,
        q,
        filter,
        vectors,
        use_text_search,
        use_vector_search,
        use_semantic_ranker,
        use_semantic_captions,
        minimum_search_score,
        minimum_reranker_score,
        use_query_rewriting,
    )

    text_sources = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)

    # Build simple citations from Document objects we return in Approach.search
    citations = []
    for r in results:
        # We only have sourcepage/sourcefile in Document; include what we can
        source = r.sourcepage or r.sourcefile
        filepath = r.sourcepage or r.sourcefile
        url = None  # add in the future if you plumb this through Document
        citations.append({"source": source, "filepath": filepath, "url": url})

    return ExtraInfo(
        DataPoints(text=text_sources, citations=citations),
        thoughts=[
            ThoughtStep(
                "Search using user query",
                q,
                {
                    "use_semantic_captions": use_semantic_captions,
                    "use_semantic_ranker": use_semantic_ranker,
                    "use_query_rewriting": use_query_rewriting,
                    "top": top,
                    "filter": filter,
                    "use_vector_search": use_vector_search,
                    "use_text_search": use_text_search,
                },
            ),
            ThoughtStep(
                "Search results",
                [result.serialize_for_results() for result in results],
            ),
        ],
    )


async def run_agentic_retrieval_approach(self, messages, overrides, auth_claims):
    minimum_reranker_score = overrides.get("minimum_reranker_score", 0)
    search_index_filter = self.build_filter(overrides, auth_claims)
    top = overrides.get("top", 3)
    max_subqueries = overrides.get("max_subqueries", 10)
    results_merge_strategy = overrides.get("results_merge_strategy", "interleaved")
    # 50 is the amount of documents that the reranker can process per query
    max_docs_for_reranker = max_subqueries * 50

    response, results = await self.run_agentic_retrieval(
        messages,
        self.agent_client,
        search_index_name=self.search_index_name,
        top=top,
        filter_add_on=search_index_filter,
        minimum_reranker_score=minimum_reranker_score,
        max_docs_for_reranker=max_docs_for_reranker,
        results_merge_strategy=results_merge_strategy,
    )

    text_sources = self.get_sources_content(results, use_semantic_captions=False, use_image_citation=False)

    citations = []
    for r in results:
        source = r.sourcepage or r.sourcefile
        filepath = r.sourcepage or r.sourcefile
        url = None
        citations.append({"source": source, "filepath": filepath, "url": url})

    extra_info = ExtraInfo(
        DataPoints(text=text_sources, citations=citations),
        thoughts=[
            ThoughtStep(
                "Use agentic retrieval",
                messages,
                {
                    "reranker_threshold": minimum_reranker_score,
                    "max_docs_for_reranker": max_docs_for_reranker,
                    "results_merge_strategy": results_merge_strategy,
                    "filter": search_index_filter,
                },
            ),
            ThoughtStep(
                f"Agentic retrieval results (top {top})",
                [result.serialize_for_results() for result in results],
                {
                    "query_plan": ([activity.as_dict() for activity in response.activity] if response.activity else None),
                    "model": self.agent_model,
                    "deployment": self.agent_deployment,
                },
            ),
        ],
    )
    return extra_info
