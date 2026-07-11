#!/usr/bin/env python3
"""
CLI for Self-Morphing Adaptive Recursion Engine

Usage examples:
    python cli.py query "How to handle recursive explosions?"
    python cli.py pog "Optimize data pipelines" --max-hops 3
    python cli.py teach --iterations 2
    python cli.py ingest file1.txt file2.txt --strategy semantic
    python cli.py metrics
    python cli.py demo
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from organized_self_morphing_engine import (
    ProductionAdaptiveEngine,
    FAISS_AVAILABLE,
    SENTENCE_TRANSFORMERS_AVAILABLE,
)

def main():
    """Parse CLI arguments, initialize the engine, and dispatch to the selected subcommand."""
    parser = argparse.ArgumentParser(description="Self-Morphing Adaptive Recursion Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Query command
    query_parser = subparsers.add_parser("query", help="Run hybrid or PoG reasoning")
    query_parser.add_argument("query_text", type=str, help="The query to reason about")
    query_parser.add_argument("--pog", action="store_true", default=True, help="Use PoG planning (default)")
    query_parser.add_argument("--max-hops", type=int, default=3, help="Max PoG hops")

    # PoG specific
    pog_parser = subparsers.add_parser("pog", help="Run dedicated PoG planning")
    pog_parser.add_argument("query_text", type=str)
    pog_parser.add_argument("--max-hops", type=int, default=3)

    # Teach
    teach_parser = subparsers.add_parser("teach", help="Run self-teaching loop")
    teach_parser.add_argument("--iterations", type=int, default=2)
    teach_parser.add_argument("--background", action="store_true", default=True)

    # Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into RAG/KG")
    ingest_parser.add_argument("files", nargs="+", help="Text files to ingest")
    ingest_parser.add_argument("--strategy", choices=["fixed", "semantic", "recursive"], default="semantic")

    # Metrics
    subparsers.add_parser("metrics", help="Show learning metrics and system state")

    # Demo
    subparsers.add_parser("demo", help="Run the full built-in demo")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    print("🚀 Initializing Self-Morphing Adaptive Recursion Engine...")
    engine = ProductionAdaptiveEngine(target_solution_text="General Purpose Reasoning", similarity_threshold=80.0)

    if args.command == "query":
        if args.pog:
            result = engine.pog_plan_and_reason(args.query_text, max_hops=args.max_hops)
            print("\n=== PoG Result ===")
            print(f"Query: {args.query_text}")
            print(f"Result: {result['result']}")
            print(f"Confidence: {result['confidence']:.1f}%")
            print(f"Verified: {result['verified']}")
            print(f"Sub-objectives: {result['sub_objectives']}")
        else:
            score = engine.hybrid_similarity(args.query_text, engine.raw_target)
            print(f"Hybrid similarity score: {score:.1f}%")

    elif args.command == "pog":
        result = engine.pog_plan_and_reason(args.query_text, max_hops=args.max_hops)
        print(result)

    elif args.command == "teach":
        print("Starting self-teaching loop...")
        thread = engine.self_teaching_loop(background=args.background, max_iterations=args.iterations)
        if not args.background:
            print("Teaching completed.")
        else:
            print(f"Background teaching started (thread: {thread.ident if thread else 'N/A'})")

    elif args.command == "ingest":
        docs = []
        for f in args.files:
            if os.path.exists(f):
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    docs.append(fh.read())
            else:
                print(f"⚠️ File not found: {f}")
        if docs:
            engine.ingest_documents(docs, strategy=args.strategy)
            print(f"✅ Ingested {len(docs)} documents.")

    elif args.command == "metrics":
        print("=== Engine Metrics ===")
        print(f"Learning Metrics: {engine.learning_metrics}")
        print(f"Synonyms loaded: {len(engine.synonym_dictionary)}")
        print(f"Neo4j connected: {getattr(engine, 'has_neo4j', False)}")
        print(f"FAISS available: {FAISS_AVAILABLE}")
        print(f"sentence-transformers available: {SENTENCE_TRANSFORMERS_AVAILABLE}")

    elif args.command == "demo":
        print("Running full built-in demo...")
        # Reuse the rich demo from __main__ logic
        engine.run_benchmarks()
        engine.run_basic_tests()
        # Quick PoG + RAG smoke
        print(engine.pog_plan_and_reason("Test recursive optimization"))
        print("Demo completed successfully.")

if __name__ == "__main__":
    main()