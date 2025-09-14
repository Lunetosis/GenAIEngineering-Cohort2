"""
Fashion RAG Pipeline - Assignment
Week 9: Multimodal RAG Pipeline with H&M Fashion Dataset

OBJECTIVE: Build a complete multimodal RAG (Retrieval-Augmented Generation) pipeline
that can search through fashion items using both text and image queries, then generate
helpful responses using an LLM.

LEARNING GOALS:
- Understand the three phases of RAG: Retrieval, Augmentation, Generation
- Work with multimodal data (images + text)
- Use vector databases for similarity search
- Integrate LLM for response generation
- Build an end-to-end AI pipeline

DATASET: H&M Fashion Caption Dataset
- 20K+ fashion items with images and text descriptions
- URL: https://huggingface.co/datasets/tomytjandra/h-and-m-fashion-caption

PIPELINE OVERVIEW:
1. RETRIEVAL: Find similar fashion items using vector search
2. AUGMENTATION: Create enhanced prompts with retrieved context
3. GENERATION: Generate helpful responses using LLM

Commands to run:
python assignment_fashion_rag.py --query "black dress for evening"
python assignment_fashion_rag.py --app
"""

import argparse
import os
import re

# Suppress warnings
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Gradio for web interface
import gradio as gr

# Core dependencies
import lancedb
import pandas as pd
from random import sample
import torch
from datasets import load_dataset
from lancedb.embeddings import EmbeddingFunctionRegistry
from lancedb.pydantic import LanceModel, Vector
from PIL import Image

# LLM dependencies
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")


def is_huggingface_space():
    """
    Checks if the code is running within a Hugging Face Spaces environment.

    Returns:
        bool: True if running in HF Spaces, False otherwise.
    """
    if os.environ.get("SYSTEM") == "spaces":
        return True
    else:
        return False


# =============================================================================
# SECTION 1: DATABASE SETUP AND SCHEMA
# =============================================================================


def register_embedding_model(model_name: str = "open-clip") -> Any:
    """
    Register embedding model for vector search
    Args:
        model_name: Name of the embedding model
    Returns:
        Embedding model instance
    """
    registry = EmbeddingFunctionRegistry.get_instance()
    model = registry.get(model_name).create()
    return model


# Global embedding model
clip_model = register_embedding_model()


class FashionItem(LanceModel):
    """
    Schema for fashion items in vector database

    TODO: Complete the schema definition
    HINT: This defines the structure of data stored in the vector database

    REQUIRED FIELDS:
    1. vector: Vector field for CLIP embeddings (use clip_model.ndims())
    2. image_uri: String field for image file paths
    3. description: Optional string field for text descriptions
    """
    
    vector: Vector(clip_model.ndims()) = clip_model.VectorField()
    image_uri: str = clip_model.SourceField()

    description: Optional[str] = None

    @property
    def image(self):
        if isinstance(self.image_uri, str) and os.path.exists(self.image_uri):
            return Image.open(self.image_uri)
        elif hasattr(self.image_uri, "save"):  # PIL Image object
            return self.image_uri
        else:
            # Return a placeholder or handle the case appropriately
            return None

# =============================================================================
# SECTION 2: RETRIEVAL - Vector Database Operations
# =============================================================================


def setup_fashion_database(
    database_path: str = "fashion_db",
    table_name: str = "fashion_items",
    dataset_name: str = "tomytjandra/h-and-m-fashion-caption",
    schema: Any = FashionItem,
    sample_size: int = 1000,
    images_dir: str = "fashion_images",
) -> None:
    """
    Set up vector database with H&M fashion dataset
    """

    # 1. Connect to LanceDB database
    db = lancedb.connect(database_path)
    
    # 2. Check if table already exists (skip if it does)
    if table_name in db.table_names():
        existing_table = db.open_table(table_name)           # open table
        print(f"✅ Table '{table_name}' already exists with {len(existing_table)} items")
        return
    else:
        print(f"🏗️ Table '{table_name}' does not exist, creating new fashion database...")

    # 3. Load H&M dataset from HuggingFace
    print("📥 Loading H&M fashion dataset...")
    dataset = load_dataset(dataset_name)
    

    # 4. Process and save images locally
    train_data = dataset["train"]
        # Sample data if needed
    if len(train_data) > sample_size:
        indices = sample(range(len(train_data)), sample_size)
        train_data = train_data.select(indices)
        
    print(f"Processing {len(train_data)} fashion items...")
    # Create images directory
    os.makedirs(images_dir, exist_ok=True)

    # Process each item
    table_data = []
    for i, item in enumerate(train_data):
        # Get image and text
        image = item["image"]
        text = item["text"]

        # Save image
        image_path = os.path.join(images_dir, f"fashion_{i:04d}.jpg")
        image.save(image_path)

        # Create record
        record = {
            "image_uri": image_path,
            "description": text
        }
        table_data.append(record)

        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(train_data)} items...")

    # 5. Create vector database table
    print("🗄️ Creating vector database table...")
    table = db.create_table(table_name, schema=schema,mode="create")
    table.add(pd.DataFrame(table_data))
    print(f"✅ Created table '{table_name}' with {len(table_data)} items")

def search_fashion_items(
    database_path: str,
    table_name: str,
    schema: Any,
    query: str,
    limit: int = 3,
    output_folder: str = "output_retriever_fashion",
    search_type: str = "auto", # "auto", "text", "image"
) -> Tuple[List[Dict], str]:
    """
    Search for fashion items using text or image query

    TODO: Complete this function to:
    1. Determine if query is text or image (auto-detection)
    2. Connect to the vector database
    3. Perform similarity search using CLIP embeddings
    4. Return search results and detected search type

    Args:
        database_path: Path to LanceDB database
        table_name: Name of the table to search
        query: Search query (text or image path)
        search_type: "auto", "text", or "image"
        limit: Number of results to return

    Returns:
        Tuple of (results_list, actual_search_type)
    """

    print(f"🔍 Searching for: {query}")

    # 1. Determine if query is text or image (auto-detection)
    actual_search_type = search_type
    processed_query = query
    if search_type == "auto":
        # Auto-detect search type
        if isinstance(query, str) or isinstance(query, Image.Image):
            if isinstance(query, Image.Image) or query.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")) or os.path.exists(query):
                # Image file path
                try:
                    print(f"Attempting to load image from path...{query}")                
                    processed_query = query if isinstance(query, Image.Image) else Image.open(query)
                    actual_search_type = "image"
                    print(f"🖼️  Detected image search: {query}")
                except Exception as e:
                    print(f"❌ Error loading image: {e}")
                    return [], "error"
            else:
                # Text query
                print(f"Detected text query...{query}")
                actual_search_type = "text"
                print(f"📝 Detected text search: {query}")
        else:  #this seems wrong (how come is not a string, but is a text query?)
            print("This seems wrong (how come is not a string, but is a text query")
            actual_search_type = "text"
            print(f"📝 Detected text search: {query}")
    elif search_type == "image":
        if isinstance(query, Image.Image) or isinstance(query, str):
            try:
                print("Attempting image query...")
                processed_query =  query if isinstance(query, Image.Image) else Image.open(query)
                actual_search_type = "image"
                print(f"🖼️  Image search: {query}")
            except Exception as e:
                print(f"❌ Error loading image: {e}")
                return [], "error"
        else:
            print("❌ Invalid image input for image search")
            return [], "error"
    else:  # text search
        print("Assuming text search...")
        actual_search_type = "text"
        print(f"📝 Text search: {query}")

    # 2. Connect to database    
    db =lancedb.connect(database_path) 

    # 3. Open table
    table = db.open_table(table_name)

    # 4. Search based on type:
    #   - Image: load with PIL and search
    #   - Text: search directly with string
    try:
        results = table.search(processed_query).limit(limit).to_pydantic(schema) #missing schema?
    except Exception as e:
        print(f"❌ Search error: {e}")
        return [], "error"

    print(f"   Found {len(results)} results using {actual_search_type} search")

    # 5. Return results and search type
    # return results, actual_search_type
    search_results = []
    
    # Clean output folder
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, file))
    else:
        os.makedirs(output_folder)
        
    for i, result in enumerate(results):
        image_path = os.path.join(output_folder, f"result_{i}.jpg")

        # Handle different image storage methods
        if result.image:
            result.image.save(image_path, "JPEG")
        else:
            print(f"Warning: No image available for result {i}")
            continue

        search_results.append(
            {
                "rank": i + 1,
                "description": result.description,  # Remove truncation
                "image_path": image_path,
                "image_uri": result.image_uri,
            }
        )

    return search_results, actual_search_type

# =============================================================================
# SECTION 3: AUGMENTATION - Prompt Engineering
# =============================================================================

def create_fashion_prompt(
    query: str, retrieved_items: List[Dict], search_type: str
) -> str:
    """
    Create enhanced prompt for LLM using retrieved fashion items

    PROMPT STRUCTURE:
    1. System prompt: Define the AI as a fashion assistant
    2. Context section: List retrieved fashion items with descriptions
    3. Query section: Include user's original query
    4. Instruction: Ask for fashion recommendations

    Args:
        query: Original user query
        retrieved_items: List of retrieved fashion items
        search_type: Type of search performed

    Returns:
        Enhanced prompt string for LLM
    """

    # 1. Creates a system prompt defining the AI assistant's role
    system_prompt =\
        '''You are a fashion assistant AI very knowledgeable about clothing styles, trends, and recommendations.
            Provide helpful and stylish fashion recommendations based on the user's query and the context of 
            retrieved fashion items."
        '''
    # 2. Formats retrieved items as context for the LLM
    context = "Here are some relevant fashion items from our catalog:\n\n"
    for i, item in enumerate(retrieved_items, 1):
        context += f"{i}. {item['description']}\n\n"

    # 3. Query section: Include user's original query
    if search_type == "image" or search_type == "auto":
        query_type = "search"
    else:
        print(f"lower text query: {query}")
        query_lower = query.lower() if isinstance(query, str) else query
        if any(word in 
                query_lower
            for word in ["recommend", "suggest", "best", "need", "looking for"]
        ):
            query_type = "recommendation"
        else:
            query_type = "search"
            
    query_section = f"User's {query_type} query: {query}"

    # 4. Combines everything into a coherent prompt
    prompt = f"{system_prompt}\n\n{context}\n{query_section}\n\nResponse:"
    return prompt

# =============================================================================
# SECTION 4: GENERATION - LLM Response Generation
# =============================================================================
def get_available_models() -> Dict[str, List[str]]:
    """Get available models for each provider."""
    models = {
        "qwen": [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
        "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    }
    return models

def setup_llm_model(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> Tuple[Any, Any]:
    """
    Set up LLM model and tokenizer

    TODO: Complete this function to load the LLM model and tokenizer

    STEPS TO IMPLEMENT:    
    3. Configure model settings for GPU/CPU
    5. Return tokenizer and model

    Args:
        model_name: Name of the model to load

    Returns:
        Tuple of (tokenizer, model)
    """

    print(f"🤖 Loading LLM model: {model_name}")

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ LLM model loaded successfully")
    return tokenizer, model


def generate_fashion_response(
    prompt: str, tokenizer: Any, model: Any, max_tokens: int = 200
) -> str:
    """
    Generate response using LLM

    Args:
        prompt: Input prompt for the model
        tokenizer: Model tokenizer
        model: LLM model
        max_tokens: Maximum tokens to generate

    Returns:
        Generated response text
    """
    # 1. Check if tokenizer and model are loaded
    if not tokenizer or not model:
        return "⚠️ LLM not loaded - showing search results only"

    # 2. Encode the prompt with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=True)

    # 3. Generate response using model.generate()
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 4. Decode the response and clean it up
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response.replace(prompt, "").strip()
    # 5. Return the generated text
    return response

# =============================================================================
# SECTION 5: IMAGE STORAGE
# =============================================================================


def save_retrieved_images(
    results: Dict[str, Any], output_dir: str = "retrieved_fashion_images"
) -> List[str]:
    """Save retrieved fashion images to output directory"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    query_safe = re.sub(r"[^\w\s-]", "", str(results["query"]))[:30]
    query_safe = re.sub(r"[-\s]+", "_", query_safe)

    saved_paths = []

    print(f"💾 Saving {len(results['results'])} retrieved images...")

    for i, item in enumerate(results["results"], 1):
        original_path = item["image_uri"]
        image = Image.open(original_path)

        # Generate new filename
        filename = f"{query_safe}_result_{i:02d}.jpg"
        save_path = os.path.join(output_dir, filename)

        # Save image
        image.save(save_path, "JPEG", quality=95)
        saved_paths.append(save_path)

        print(f"   ✅ Saved image {i}: {filename}")
        print(f"      Description: {item.get('description', 'No description')[:60]}...")

    print(f"💾 Saved {len(saved_paths)} images to: {output_dir}")
    return saved_paths


# =============================================================================
# SECTION 6: COMPLETE RAG PIPELINE
# =============================================================================


def run_fashion_rag_pipeline(
    query: str,
    database_path: str = "fashion_db",
    table_name: str = "fashion_items",
    schema: Any = FashionItem,
    search_type: str = "auto",
    limit: int = 3,
    save_images: bool = True,
) -> Dict[str, Any]:
    """
    Run complete fashion RAG pipeline

    TODO: Complete this function to orchestrate the entire pipeline:
    3. GENERATION: Generate LLM response using the enhanced prompt
    4. IMAGE STORAGE: Save retrieved images if requested

    This is the main function that ties everything together!

    PIPELINE PHASES:
    Phase 1 - RETRIEVAL: Find similar fashion items
    Phase 2 - AUGMENTATION: Create context-rich prompt
    Phase 3 - GENERATION: Generate helpful response
    Phase 4 - STORAGE: Save retrieved images
    """

    print("🚀 Starting Fashion RAG Pipeline")
    print("=" * 50)

    # PHASE 1: RETRIEVAL: Search for relevant fashion items using vector database
    print("🔍 PHASE 1: RETRIEVAL")
    # TODO: Search for fashion items using the search function
    results, actual_search_type = search_fashion_items(
        database_path,table_name, schema, query, limit, output_folder = "output_retriever_fashion",
        search_type = search_type)
    
    print(f"   Found {len(results)} relevant items")

    # PHASE 2: AUGMENTATION: Create enhanced prompt with retrieved context
    print("📝 PHASE 2: AUGMENTATION")
    print("Actual search type:", actual_search_type)
    enhanced_prompt = create_fashion_prompt(query, results, actual_search_type)
    print(f"   Created enhanced prompt ({len(enhanced_prompt)} chars)")

    # PHASE 3: GENERATION: enerate LLM response using the enhanced prompt
    print("🤖 PHASE 3: GENERATION")
    # Set up LLM and generate response
    tokenizer, model = setup_llm_model()
    response = generate_fashion_response(
        enhanced_prompt,tokenizer, model, max_tokens= 200)
    print(f"   Generated response ({len(response)} chars)")

    # Prepare final results dictionary
    final_results = {
        "query": query,
        "results": results,
        "response": response,
        "search_type": actual_search_type
    }

    # Save retrieved images if requested
    if save_images:
        saved_image_paths = save_retrieved_images(final_results)
        final_results["saved_image_paths"] = saved_image_paths

    # Return final results
    return final_results

# =============================================================================
# GRADIO WEB APP
# =============================================================================


def fashion_search_app(query):
    """
    Process fashion query and return response with images for Gradio

    TODO: Complete this function to handle web app queries

    STEPS TO IMPLEMENT:
    
    
    
    """

    # 1. Check if query is provided
    if not query.strip():
        return "Please enter a search query", []

    # 2. Setup database if needed
    setup_fashion_database(
            database_path = "fashion_db",
            table_name = "fashion_items",
            dataset_name = "tomytjandra/h-and-m-fashion-caption",
            schema = FashionItem,
            sample_size = 1000,
            images_dir = "fashion_images"
            ) 

    # 3. Run RAG pipeline
    result = run_fashion_rag_pipeline(
        query,
        database_path = "fashion_db",
        table_name = "fashion_items",
        schema = FashionItem,
        search_type = "auto",
        limit = 3,
        save_images = True)

    # 4. Extract LLM response and images
    llm_response = result['response']

    # 5. Return formatted results for Gradio
    retrieved_images = []
    for item in result['results']:
        if 'image_uri' in item and os.path.exists(item['image_uri']):
            img = Image.open(item['image_uri'])
            retrieved_images.append(img)

    # Return response and images
    return llm_response, retrieved_images

def launch_gradio_app():
    """Launch the Gradio web interface"""

    # Create Gradio interface
    with gr.Blocks(title="Fashion RAG Assistant") as app:

        gr.Markdown("# 👗 Fashion RAG Assistant")
        gr.Markdown("Search for fashion items and get AI-powered recommendations!")

        with gr.Row():
            with gr.Column(scale=1):
                # Input
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter your fashion query (e.g., 'black dress for evening')",
                    lines=2,
                )

                search_btn = gr.Button("Search", variant="primary")

                # Examples
                gr.Examples(
                    examples=[
                        "black dress for evening",
                        "casual summer outfit",
                        "blue jeans",
                        "white shirt",
                        "winter jacket",
                    ],
                    inputs=query_input,
                )

            with gr.Column(scale=2):
                # Output
                response_output = gr.Textbox(
                    label="Fashion Recommendation", lines=8, interactive=False
                )

        # Retrieved Images
        images_output = gr.Gallery(
            label="Retrieved Fashion Items", columns=3, height=400
        )

        # Connect the search function
        search_btn.click(
            fn=fashion_search_app,
            inputs=query_input,
            outputs=[response_output, images_output],
        )

        # Also trigger on Enter key
        query_input.submit(
            fn=fashion_search_app,
            inputs=query_input,
            outputs=[response_output, images_output],
        )

    print("🚀 Starting Fashion RAG Gradio App...")
    print("📝 Note: First run will download dataset and setup database")
    app.launch(share=True)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main function to handle command line arguments and run the pipeline"""

    # If running in Hugging Face Spaces, automatically launch the app
    if is_huggingface_space():
        print("🤗 Running in Hugging Face Spaces - launching web app automatically")
        launch_gradio_app()
        return

    parser = argparse.ArgumentParser(
        description="Fashion RAG Pipeline Assignment - SOLUTION"
    )
    parser.add_argument("--query", type=str, help="Search query (text or image path)")
    parser.add_argument("--app", action="store_true", help="Launch Gradio web app")

    args = parser.parse_args()

    # Launch web app if requested
    if args.app:
        launch_gradio_app()
        return

    if not args.query:
        print("❌ Please provide a query with --query or use --app for web interface")
        print("Examples:")
        print("  python solution_fashion_rag.py --query 'black dress for evening'")
        print("  python solution_fashion_rag.py --query 'fashion_images/dress.jpg'")
        print("  python solution_fashion_rag.py --app")
        return

    # Setup database first (will skip if already exists)
    print("🔧 Checking/setting up fashion database...")
    setup_fashion_database()

    # Run the complete RAG pipeline with default settings
    result = run_fashion_rag_pipeline(
        query=args.query,
        database_path="fashion_db",
        table_name="fashion_items",
        search_type="auto",
        limit=3,
        save_images=True,
    )

    # Display results
    print("\n" + "=" * 50)
    print("🎯 PIPELINE RESULTS")
    print("=" * 50)
    print(f"Query: {result['query']}")
    print(f"Search Type: {result['search_type']}")
    print(f"Results Found: {len(result['results'])}")
    print("\n📋 Retrieved Items:")
    for i, item in enumerate(result["results"], 1):
        print(f"{i}. {item.get('description', 'No description')}")

    print(f"\n🤖 LLM Response:")
    print(result["response"])

    # Show saved images info if any
    if result.get("saved_image_paths"):
        print(f"\n📸 Saved Images:")
        for i, path in enumerate(result["saved_image_paths"], 1):
            print(f"{i}. {path}")


if __name__ == "__main__":
    main()