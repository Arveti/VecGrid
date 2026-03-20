import os
import sys
import time
import struct
import wave
import numpy as np
from PIL import Image
from google import genai
from vecgrid import VecGrid

def create_sample_files():
    """Generates dummy files for testing the supported modalities."""
    files = {}
    
    # 1. Image
    files["image"] = "sample_image.jpg"
    if not os.path.exists(files["image"]):
        img = Image.new('RGB', (200, 200), color='blue')
        img.save(files["image"])
        print(f"[*] Created sample image: {files['image']}")

    # 2. Audio (WAV file with 1 second of silence)
    files["audio"] = "sample_audio.wav"
    if not os.path.exists(files["audio"]):
        with wave.open(files["audio"], "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(44100)
            f.writeframes(b"".join([struct.pack("<h", 0)] * 44100))
        print(f"[*] Created sample audio: {files['audio']}")

    # 3. PDF (Minimal valid PDF)
    files["pdf"] = "sample_doc.pdf"
    if not os.path.exists(files["pdf"]):
        pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources <<>> >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n199\n%%EOF\n"
        with open(files["pdf"], 'wb') as f:
            f.write(pdf_content)
        print(f"[*] Created sample PDF: {files['pdf']}")

    # 4. Video
    # We create a dummy byte file. Gemini will reject it if parsed strictly as an MP4, so 
    # we advise replacing it in the CLI output.
    files["video"] = "sample_video.mp4"
    if not os.path.exists(files["video"]):
        with open(files["video"], 'wb') as f:
            f.write(b"") # Note: Replace this empty file with a real video
        print(f"[*] Created empty sample video: {files['video']} (PLEASE REPLACE WITH REAL MP4!)")

    return files

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is missing.", file=sys.stderr)
        print("Please run: export GEMINI_API_KEY='your_api_key'", file=sys.stderr)
        sys.exit(1)

    # Initialize Gemini client
    client = genai.Client()
    
    # Using gemini-embedding-2-preview which is Gemini's new multimodal embedding model
    # Replace this if your targeted multimodal model string is different.
    model_id = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview")
    print(f"[*] Using embedding model: {model_id}")

    print("\n[*] Preparing dummy files for ALL modalities...")
    modality_files = create_sample_files()
    
    print("\n[*] Uploading supported large files (Audio, Video, PDF) to Gemini File API...")
    # GenAI SDK expects larger files (video, audio, pdfs usually) to be uploaded via the File API first
    uploaded_files = {}
    for mod in ["audio", "video", "pdf"]:
        path = modality_files[mod]
        try:
            print(f"    -> Uploading {mod} ({path})...")
            # If the dummy format is rejected by the API, this might fail or get marked as FAILED! 
            # In production, ensure valid content.
            uploaded_file = client.files.upload(file=path)
            
            # Wait for file processing if needed (videos require this)
            while uploaded_file.state.name == "PROCESSING":
                print(".", end="", flush=True)
                time.sleep(2)
                # Check status
                uploaded_file = client.files.get(name=uploaded_file.name)
            
            if uploaded_file.state.name == "FAILED":
                print(f"\n    -> Failed to process {path} server-side. (Expected if the file is an empty dummy!)")
            else:
                print(" Done.")
                uploaded_files[mod] = uploaded_file
                
        except Exception as e:
            print(f"    -> API threw an error during upload/processing (Expected if empty file): {e}")

    # Prepare array of multimodal documents. Text and PIL Images can be passed directly!
    sample_img = Image.open(modality_files["image"])

    documents = [
        {"id": "doc_text_shape", "type": "text", "content": "A beautiful bright blue square shape."},
        {"id": "doc_text_sound", "type": "text", "content": "Absolute silence, nothing but quiet."},
        {"id": "doc_text_pdf",   "type": "text", "content": "A short blank PDF document."},
        {"id": "doc_image_1",    "type": "image", "content": sample_img},
    ]
    
    # Add uploaded File API objects to the documents array ONLY if they uploaded successfully
    for mod, f_obj in uploaded_files.items():
        documents.append({
            "id": f"doc_{mod}_1",
            "type": mod,
            "content": f_obj, # Pass the Google GenAI File Object directly for embedding calculation
        })

    print(f"\n[*] Generating embeddings for all single-space modalities... ({len(documents)} valid objects)")
    vectors = {}
    valid_documents = [] # Tracks successful ones
    
    for doc in documents:
        try:
            response = client.models.embed_content(
                model=model_id,
                contents=doc["content"],
            )
            vec = response.embeddings[0].values
            vectors[doc["id"]] = np.array(vec, dtype=np.float32)
            valid_documents.append(doc)
            print(f"    -> Generated {len(vec)}-dimensional vector for {doc['id']} (modality: {doc['type']})")
            
        except Exception as e:
            print(f"    -> Error generating embedding for {doc['id']} ({doc['type']}): {e}")

    if not vectors:
        print("No vectors generated. Exiting.")
        sys.exit(1)

    # All vectors in the multimodal space share the EXACT same dimensionality!
    dim = len(next(iter(vectors.values())))
    print(f"\n[*] Initializing VecGrid with dim={dim}...")
    
    # Initialize one embedded node (in-process mode)
    grid = VecGrid(node_id="embed-multi-all-modals", dim=dim)
    grid.start()

    print("\n[*] Inserting successful multimodal vectors into VecGrid index...")
    for doc in valid_documents:
        meta = {"type": doc["type"]}
        meta["identifier"] = doc["id"]
        grid.put(doc["id"], vectors[doc["id"]], meta)
    
    print("\n[*] Performing a multimodal semantic search!")
    # We will search using our sample audio (if valid) or the image!
    target_query_id = "doc_image_1"
    
    if target_query_id in vectors:
        print(f"    -> Searching VecGrid using the IMAGE ({target_query_id}) vector as query...")
        query_vector = vectors[target_query_id]
        
        # We want top 3 closest matches, which should find "doc_text_shape" since they represent the same semantic object!
        results = grid.search(query_vector, k=3)
        
        print("\n" + "="*40)
        print("SEMANTIC MULTIMODAL SEARCH RESULTS:")
        print("="*40)
        for i, r in enumerate(results, start=1):
            print(f"Result {i}:")
            print(f"   ID:       {r.vector_id}")
            print(f"   Distance: {r.distance:.4f} (lower is closer semantically)")
            print(f"   Metadata: {r.metadata}")
            print("-" * 40)
    else:
        print("\n    -> Skipped search because the image embedding was not generated successfully.")

    grid.stop()
    print("\n[*] Done! Multi-modal demo completed.")

if __name__ == "__main__":
    main()
