#!/usr/bin/env python3
"""Debug script to check why Hypos/Glossary isn't being retrieved."""

from src.hybrid_retriever import HybridRetriever
from src.graph_retriever import GraphRetriever, NodeLayer

print("=" * 60)
print("DEBUGGING RETRIEVAL FOR: 'does Ishmael have depression?'")
print("=" * 60)

# Test 1: Direct fulltext search
print("\n1. Direct fulltext_search for 'depression':")
gr = GraphRetriever()
results = gr.fulltext_search('depression', limit=10)
for r in results:
    print(f"   [{r.get('type')}] {r.get('name', r.get('term', '?'))}")
    if r.get('definition'):
        print(f"       Definition: {r.get('definition')[:80]}...")

# Test 2: Hybrid retriever (no vector)
print("\n2. Hybrid retriever (graph only, no vector):")
hr = HybridRetriever()
results = hr.retrieve('does Ishmael have depression?', max_facts=10, max_analysis=5, use_vector=False)

print(f"\n   FACTS ({len(results.facts)} results):")
for r in results.facts:
    name = r.node.get("name", r.node.get("term", "?"))
    print(f"   [{r.node_type}] {name} - score: {r.score:.2f}")

print(f"\n   ANALYSIS ({len(results.analysis)} results):")
for r in results.analysis:
    name = r.node.get("name", r.node.get("term", "?"))
    print(f"   [{r.node_type}] {name} - score: {r.score:.2f}")

# Check if Glossary is in FACT_TYPES
print("\n3. Checking FACT_TYPES configuration:")
print(f"   FACT_TYPES: {hr.FACT_TYPES}")
print(f"   'Glossary' in FACT_TYPES: {'Glossary' in hr.FACT_TYPES}")

print("\n" + "=" * 60)
