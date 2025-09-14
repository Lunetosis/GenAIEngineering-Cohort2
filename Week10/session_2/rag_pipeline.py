"""
SCRIPT 4/5: rag_pipeline.py - Complete RAG Pipeline Integration for Shoe Search

Colab - https://colab.research.google.com/drive/1rq-ywjykHBw7xPXCmd3DmZdK6T9bhDtA?usp=sharing

This script integrates all three phases of the RAG pipeline:
1. RETRIEVAL: Vector search and data management (from retriever.py)
2. AUGMENTATION: Context enhancement and prompt engineering (from augmenter.py)
3. GENERATION: LLM setup and response generation (from generator.py)

Key Concepts:
- RAG (Retrieval-Augmented Generation): A technique that combines information retrieval
  with language generation to provide accurate, contextual responses
- Pipeline Integration: Connecting multiple AI components in sequence
- End-to-End Processing: Complete workflow from query to final response
- Multi-modal Search: Supporting both text and image queries

Required Dependencies:
- All dependencies from retriever.py, augmenter.py, and generator.py

Commands to run:
# Complete RAG pipeline with text query
python rag_pipeline.py --query "recommend running shoes for men"

# Complete RAG pipeline with image query
python rag_pipeline.py --query "hf_shoe_images/shoe_0000.jpg"

# RAG pipeline with OpenAI model (Requires API key)
python rag_pipeline.py --query "comfortable sneakers" --model-provider openai --openai-api-key YOUR_KEY

# RAG pipeline with detailed step tracking
python rag_pipeline.py --query "blue shoes" --detailed-steps

# Setup database and run pipeline
python rag_pipeline.py --setup-db --query "recommend me men's casual shoes"

# Pipeline without LLM (retrieval only)
python rag_pipeline.py --query "recommend me men's running shoes" --no-llm
"""

import argparse
from typing import Any, Dict, List, Optional

from augmenter import QueryType, SimpleShoePrompts
from generator import (
    generate_shoes_rag_response,
    get_available_models,
    setup_openai_client,
    setup_qwen_model,
)
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv  # For loading API key from a .env file
import os

# Import components from other modules
from retriever import MyntraShoesEnhanced, create_shoes_table_from_hf, run_shoes_search


def run_complete_shoes_rag_pipeline(
    database: str,
    table_name: str,
    schema: Any,
    search_query: Any,  # Can be text string or image path/PIL Image
    limit: int = 3,
    use_llm: bool = True,
    use_advanced_prompts: bool = True,
    search_type: str = "auto",
    model_provider: str = "qwen",
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run complete RAG pipeline integrating Retrieval, Augmentation, and Generation."""

    # SECTION 1: RETRIEVAL - Get relevant shoes from vector database
    print("🔍 RETRIEVAL: Searching for relevant shoes...")
    results, actual_search_type = run_shoes_search(
        database, table_name, schema, search_query, limit, search_type=search_type
    )

    if not results:
        return {
            "query": search_query,
            "results": [],
            "response": "No results found",
            "search_type": actual_search_type,
        }

    if not use_llm:
        return {
            "query": search_query,
            "results": results,
            "response": None,
            "search_type": actual_search_type,
        }

    # SECTION 2: AUGMENTATION - Process and enhance context with prompt engineering
    try:
        print("📝 AUGMENTATION: Enhancing context with prompt engineering...")

        # Set up prompt manager and analyze query
        prompt_manager = SimpleShoePrompts()

        # For image search, use appropriate query text
        if actual_search_type == "image":
            query_text = "similar shoes based on the provided image"
            print(f"   └─ Image search - using search query type")
        else:
            query_text = str(search_query)
            query_type = prompt_manager.classify_query(query_text)
            print(f"   └─ Text query classified as: {query_type.value}")

        # Format context and generate enhanced prompt
        enhanced_prompt = prompt_manager.generate_prompt(
            query_text, results, actual_search_type
        )
        print(f"   └─ Context formatted with {len(results)} retrieved shoes")

        # SECTION 3: GENERATION - Setup LLM and generate response
        print("🤖 GENERATION: Setting up LLM and generating response...")

        tokenizer, model, openai_client = None, None, None
        use_dotenv =True

        if model_provider == "openai":
            if not openai_api_key:
                if use_dotenv:
                    load_dotenv()  # Load environment variables from .env file
                    args.openai_api_key = os.getenv("OPENAI_API_KEY")
                else:
                    raise ValueError("OpenAI API key is required for OpenAI models")
            openai_client = setup_openai_client(openai_api_key)
            print(f"   └─ OpenAI client setup with model: {model_name}")
        else:
            tokenizer, model = setup_qwen_model(model_name)
            print(f"   └─ Qwen model loaded: {model_name}")

        # Generate final response using augmented context
        response = generate_shoes_rag_response(
            query=query_text,
            retrieved_shoes=results,
            model_provider=model_provider,
            model_name=model_name,
            openai_client=openai_client,
            tokenizer=tokenizer,
            model=model,
            max_tokens=200,
            use_advanced_prompts=use_advanced_prompts,
        )

        # Add prompt analysis
        if actual_search_type == "image":
            final_query_type = QueryType.SEARCH.value
        else:
            final_query_type = query_type.value

        prompt_analysis = {
            "query_type": final_query_type,
            "num_results": len(results),
            "search_type": actual_search_type,
        }

        return {
            "query": search_query,
            "results": results,
            "response": response,
            "prompt_analysis": prompt_analysis,
            "search_type": actual_search_type,
        }
    except Exception as e:
        print(f"LLM generation failed: {e}")
        return {
            "query": search_query,
            "results": results,
            "response": "LLM unavailable - showing search results only",
            "search_type": actual_search_type,
        }


def run_complete_shoes_rag_pipeline_with_details(
    database: str,
    table_name: str,
    schema: Any,
    search_query: Any,  # Can be text string or image path/PIL Image
    limit: int = 3,
    use_llm: bool = True,
    use_advanced_prompts: bool = True,
    search_type: str = "auto",
    model_provider: str = "qwen",
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run complete RAG pipeline with detailed step tracking."""

    # Initialize step details
    retrieval_details = ""
    augmentation_details = ""
    generation_details = ""

    # SECTION 1: RETRIEVAL - Get relevant shoes from vector database
    retrieval_details += "🔍 RETRIEVAL PHASE\n"
    retrieval_details += "=" * 50 + "\n"
    retrieval_details += f"🎯 Query Type: {search_type}\n"
    retrieval_details += f"🔍 Searching vector database...\n"

    results, actual_search_type = run_shoes_search(
        database, table_name, schema, search_query, limit, search_type=search_type
    )

    retrieval_details += f"✅ Search completed!\n"
    retrieval_details += f"📊 Search Type Detected: {actual_search_type}\n"
    retrieval_details += f"📈 Results Found: {len(results)}\n\n"

    if results:
        retrieval_details += "🎯 Retrieved Products:\n"
        for i, result in enumerate(results, 1):
            retrieval_details += f"  {i}. {result.get('product_type', 'Shoe')} for {result.get('gender', 'Unisex')}\n"
            retrieval_details += f"     Color: {result.get('color', 'N/A')}\n"
            retrieval_details += f"     Pattern: {result.get('pattern', 'N/A')}\n"
            if result.get("description"):
                # Show full description without truncation
                retrieval_details += f"     Description: {result['description']}\n"
            retrieval_details += "\n"
    else:
        retrieval_details += "❌ No results found\n"
        return {
            "query": search_query,
            "results": [],
            "response": "No results found",
            "search_type": actual_search_type,
            "retrieval_details": retrieval_details,
            "augmentation_details": "⏭️ Skipped - No results to process",
            "generation_details": "⏭️ Skipped - No results to process",
        }

    if not use_llm:
        return {
            "query": search_query,
            "results": results,
            "response": None,
            "search_type": actual_search_type,
            "retrieval_details": retrieval_details,
            "augmentation_details": "⏭️ Skipped - LLM disabled",
            "generation_details": "⏭️ Skipped - LLM disabled",
        }

    # SECTION 2: AUGMENTATION - Process and enhance context with prompt engineering
    try:
        augmentation_details += "📝 AUGMENTATION PHASE\n"
        augmentation_details += "=" * 50 + "\n"

        # Set up prompt manager and analyze query
        prompt_manager = SimpleShoePrompts()

        # For image search, use appropriate query text
        if actual_search_type == "image":
            query_text = "similar shoes based on the provided image"
            augmentation_details += f"🖼️ Image Search Detected\n"
            augmentation_details += f"🔄 Query Text: '{query_text}'\n"
        else:
            query_text = str(search_query)
            query_type = prompt_manager.classify_query(query_text)
            augmentation_details += f"📝 Text Query: '{query_text}'\n"
            augmentation_details += f"🎯 Query Classification: {query_type.value}\n"

        # Format context and generate enhanced prompt
        enhanced_prompt = prompt_manager.generate_prompt(
            query_text, results, actual_search_type
        )

        augmentation_details += f"📊 Context Processing:\n"
        augmentation_details += f"  • Products formatted: {len(results)}\n"
        augmentation_details += (
            f"  • Prompt strategy: {'Advanced' if use_advanced_prompts else 'Basic'}\n"
        )
        augmentation_details += (
            f"  • Prompt length: {len(enhanced_prompt)} characters\n\n"
        )

        # Show the full prompt instead of preview
        augmentation_details += f"🔍 Full Prompt:\n{enhanced_prompt}\n\n"

        # SECTION 3: GENERATION - Setup LLM and generate response
        generation_details += "🤖 GENERATION PHASE\n"
        generation_details += "=" * 50 + "\n"
        generation_details += f"🏭 Model Provider: {model_provider}\n"
        generation_details += f"🎯 Model Name: {model_name}\n"
        
        tokenizer, model, openai_client = None, None, None
        use_dotenv =True

        if model_provider == "openai":
            if not openai_api_key:
                if use_dotenv:
                    load_dotenv()  # Load environment variables from .env file
                    args.openai_api_key = os.getenv("OPENAI_API_KEY")
                else:
                    raise ValueError("OpenAI API key is required for OpenAI models")
            openai_client = setup_openai_client(openai_api_key)
            generation_details += f"✅ OpenAI client initialized\n"
            generation_details += f"🔑 API key: {'*' * (len(openai_api_key) - 8) + openai_api_key[-4:] if len(openai_api_key) > 8 else '****'}\n"
        else:
            tokenizer, model = setup_qwen_model(model_name)
            generation_details += f"✅ Qwen model loaded\n"
            generation_details += f"💾 Model size: {model_name}\n"

        generation_details += f"⚙️ Generation settings:\n"
        generation_details += f"  • Max tokens: 200\n"
        generation_details += f"  • Temperature: 0.1 (low for consistency)\n"
        generation_details += f"  • Advanced prompts: {use_advanced_prompts}\n\n"

        generation_details += f"🔄 Generating response...\n"

        # Generate final response using augmented context
        response = generate_shoes_rag_response(
            query=query_text,
            retrieved_shoes=results,
            model_provider=model_provider,
            model_name=model_name,
            openai_client=openai_client,
            tokenizer=tokenizer,
            model=model,
            max_tokens=200,
            use_advanced_prompts=use_advanced_prompts,
        )

        generation_details += f"✅ Response generated!\n"
        generation_details += f"📏 Response length: {len(response)} characters\n"
        generation_details += f"📝 Full Response:\n{response}\n"

        # Add prompt analysis
        if actual_search_type == "image":
            final_query_type = QueryType.SEARCH.value
        else:
            final_query_type = query_type.value

        prompt_analysis = {
            "query_type": final_query_type,
            "num_results": len(results),
            "search_type": actual_search_type,
        }

        return {
            "query": search_query,
            "results": results,
            "response": response,
            "prompt_analysis": prompt_analysis,
            "search_type": actual_search_type,
            "retrieval_details": retrieval_details,
            "augmentation_details": augmentation_details,
            "generation_details": generation_details,
        }
    except Exception as e:
        error_msg = f"❌ LLM generation failed: {str(e)}"
        generation_details += error_msg
        return {
            "query": search_query,
            "results": results,
            "response": "LLM unavailable - showing search results only",
            "search_type": actual_search_type,
            "retrieval_details": retrieval_details,
            "augmentation_details": augmentation_details,
            "generation_details": generation_details,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete RAG Pipeline for Shoe Search"
    )
    parser.add_argument(
        "--query", type=str, help="Search query (text) or image file path"
    )
    parser.add_argument(
        "--search-type",
        choices=["auto", "text", "image"],
        default="auto",
        help="Force search type (default: auto-detect)",
    )
    parser.add_argument(
        "--limit", type=int, default=3, help="Number of results to retrieve"
    )
    parser.add_argument(
        "--database", type=str, default="myntra_shoes_db", help="Database path"
    )
    parser.add_argument(
        "--table-name", type=str, default="myntra_shoes_table", help="Table name"
    )

    # Model configuration
    parser.add_argument(
        "--model-provider",
        choices=["qwen", "openai"],
        default="qwen",
        help="Model provider to use",
    )
    parser.add_argument("--model-name", type=str, help="Model name to use")
    parser.add_argument(
        "--openai-api-key", type=str, help="OpenAI API key (required for OpenAI models)"
    )
    parser.add_argument(
        "--use-advanced-prompts",
        action="store_true",
        default=True,
        help="Use advanced prompt engineering",
    )
    parser.add_argument(
        "--basic-prompts",
        action="store_true",
        help="Use basic prompts instead of advanced",
    )

    # Pipeline options
    parser.add_argument(
        "--no-llm", action="store_true", help="Run retrieval only, skip LLM generation"
    )
    parser.add_argument(
        "--detailed-steps",
        action="store_true",
        help="Show detailed step-by-step breakdown",
    )
    parser.add_argument(
        "--setup-db",
        action="store_true",
        help="Setup database from HuggingFace dataset",
    )
    parser.add_argument(
        "--sample-size", type=int, default=500, help="Sample size for database setup"
    )

    args = parser.parse_args()

    # Setup database if requested
    if args.setup_db:
        print("🔄 Setting up database from HuggingFace dataset...")
        create_shoes_table_from_hf(
            database=args.database,
            table_name=args.table_name,
            sample_size=args.sample_size,
            save_images=True,
        )
        print("✅ Database setup complete!")
        if not args.query:
            exit(0)

    # Validate query
    if not args.query:
        print("❌ Please provide a query using --query")
        print("\nExample usage:")
        print("  # Setup database first")
        print("  python rag_pipeline.py --setup-db")
        print("  # Complete RAG pipeline with text query")
        print("  python rag_pipeline.py --query 'recommend running shoes for men'")
        print("  # RAG pipeline with image query")
        print("  python rag_pipeline.py --query 'path/to/shoe.jpg' --search-type image")
        print("  # RAG pipeline with OpenAI")
        print(
            "  python rag_pipeline.py --query 'comfortable sneakers' --model-provider openai --openai-api-key YOUR_KEY"
        )
        print("  # Detailed step tracking")
        print("  python rag_pipeline.py --query 'blue shoes' --detailed-steps")
        exit(1)

    # Set default model names based on provider
    available_models = get_available_models()
    if not args.model_name:
        args.model_name = available_models[args.model_provider][0]

    # Handle basic prompts flag
    use_advanced_prompts = (
        not args.basic_prompts if args.basic_prompts else args.use_advanced_prompts
    )

    # Validate OpenAI setup
    if args.model_provider == "openai":
        if not args.openai_api_key:
            print(
                "❌ OpenAI API key is required for OpenAI models. Use --openai-api-key"
            )
            exit(1)

    print("🚀 Starting Complete RAG Pipeline...")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"Search Type: {args.search_type}")
    print(f"Model Provider: {args.model_provider}")
    print(f"Model Name: {args.model_name}")
    print(f"Use LLM: {not args.no_llm}")
    print(f"Advanced Prompts: {use_advanced_prompts}")

    # Run pipeline
    if args.detailed_steps:
        rag_result = run_complete_shoes_rag_pipeline_with_details(
            database=args.database,
            table_name=args.table_name,
            schema=MyntraShoesEnhanced,
            search_query=args.query,
            limit=args.limit,
            use_llm=not args.no_llm,
            use_advanced_prompts=use_advanced_prompts,
            search_type=args.search_type,
            model_provider=args.model_provider,
            model_name=args.model_name,
            openai_api_key=args.openai_api_key,
        )

        # Display detailed results
        print("\n" + "=" * 60)
        print("📊 RAG PIPELINE DETAILED RESULTS")
        print("=" * 60)

        print("\n" + rag_result.get("retrieval_details", "No retrieval details"))
        print("\n" + rag_result.get("augmentation_details", "No augmentation details"))
        print("\n" + rag_result.get("generation_details", "No generation details"))

    else:
        rag_result = run_complete_shoes_rag_pipeline(
            database=args.database,
            table_name=args.table_name,
            schema=MyntraShoesEnhanced,
            search_query=args.query,
            limit=args.limit,
            use_llm=not args.no_llm,
            use_advanced_prompts=use_advanced_prompts,
            search_type=args.search_type,
            model_provider=args.model_provider,
            model_name=args.model_name,
            openai_api_key=args.openai_api_key,
        )

    # Display results
    print("\n" + "=" * 60)
    print("📊 RAG PIPELINE RESULTS")
    print("=" * 60)
    print(f"Query: {rag_result['query']}")
    print(f"Search Type: {rag_result['search_type']}")
    if rag_result.get("prompt_analysis"):
        print(f"Query Type: {rag_result['prompt_analysis']['query_type']}")
        print(f"Results Found: {rag_result['prompt_analysis']['num_results']}")

    if rag_result.get("response"):
        print(f"\n💬 RAG Response:")
        print(rag_result["response"])

    print(f"\n👟 Retrieved Shoes:")
    for result in rag_result["results"]:
        print(
            f"- {result['product_type']} ({result['gender']}) - {result['color']} - {result['pattern']}"
        )
        if rag_result["search_type"] == "image":
            print(f"  📁 Image saved: {result['image_path']}")

    if rag_result["search_type"] == "image":
        print(f"\n🖼️  Search results images saved in: shoe_search_output/")

    print("\n✅ RAG Pipeline Complete!")