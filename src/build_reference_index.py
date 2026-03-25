"""
Reference Image Index Builder

Scans reference_images/{Category}/ folders, encodes each image with CLIP,
and builds a FAISS index for fast image-to-image similarity search.

Usage:
    python -m src.build_reference_index

This runs 100% locally — zero API cost. Takes ~1-2 minutes for ~200 images.

Output:
    reference_images/clip_reference_index.faiss
    reference_images/clip_reference_metadata.json
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List
from PIL import Image

# Ensure project root is on path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

REFERENCE_IMAGES_DIR = os.path.join(project_root, "reference_images")
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
MIN_IMAGE_SIZE = 50  # Skip images smaller than 50x50


def build_reference_index():
    """
    Main function: scan reference folders → encode with CLIP → build FAISS.
    """
    import torch
    import faiss
    from transformers import CLIPProcessor, CLIPModel

    print("=" * 60)
    print("  CLIP Reference Image Index Builder")
    print("  (runs locally — zero API cost)")
    print("=" * 60)

    # 1. Scan reference folders
    if not os.path.isdir(REFERENCE_IMAGES_DIR):
        print(f"\nERROR: '{REFERENCE_IMAGES_DIR}' not found.")
        print("Create the folder and add disease category subfolders with images.")
        return

    categories = []
    for item in sorted(os.listdir(REFERENCE_IMAGES_DIR)):
        item_path = os.path.join(REFERENCE_IMAGES_DIR, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            categories.append(item)

    if not categories:
        print(f"\nERROR: No category folders found in {REFERENCE_IMAGES_DIR}")
        print("Create folders like Late_Blight/, Early_Blight/, etc. and add images.")
        return

    print(f"\nFound {len(categories)} disease categories:")
    total_images = 0
    category_counts = {}

    for cat in categories:
        cat_path = os.path.join(REFERENCE_IMAGES_DIR, cat)
        images = [
            f for f in os.listdir(cat_path)
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
        ]
        category_counts[cat] = len(images)
        total_images += len(images)
        status = "✓" if len(images) >= 5 else "⚠ (low count)"
        print(f"  {cat}: {len(images)} images {status}")

    if total_images == 0:
        print("\nERROR: No images found! Add .jpg/.png images to the category folders.")
        return

    print(f"\nTotal: {total_images} images across {len(categories)} categories")

    # 2. Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading CLIP model ({CLIP_MODEL_NAME}) on {device}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()
    print("Model loaded.")

    # 3. Encode all images
    print(f"\nEncoding {total_images} images...")
    embeddings = []
    metadata = []
    skipped = 0
    processed = 0

    for cat in categories:
        cat_path = os.path.join(REFERENCE_IMAGES_DIR, cat)
        image_files = [
            f for f in sorted(os.listdir(cat_path))
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
        ]

        for img_file in image_files:
            img_path = os.path.join(cat_path, img_file)

            try:
                image = Image.open(img_path).convert("RGB")

                # Skip tiny images
                if image.size[0] < MIN_IMAGE_SIZE or image.size[1] < MIN_IMAGE_SIZE:
                    print(f"  Skipping {img_file} (too small: {image.size})")
                    skipped += 1
                    continue

                # Encode with CLIP
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    features = model.get_image_features(**inputs)
                    features = features / features.norm(dim=-1, keepdim=True)

                embedding = features.squeeze(0).cpu().numpy().astype(np.float32)
                embeddings.append(embedding)

                metadata.append({
                    "image_path": img_path,
                    "image_name": img_file,
                    "disease": cat,
                })

                processed += 1
                if processed % 25 == 0:
                    print(f"  Processed {processed}/{total_images}...")

            except Exception as e:
                print(f"  ERROR processing {img_file}: {e}")
                skipped += 1

    print(f"\nEncoded {len(embeddings)} images (skipped {skipped})")

    if not embeddings:
        print("ERROR: No valid images encoded!")
        return

    # 4. Build FAISS index (inner product = cosine sim for L2-normalized vectors)
    embedding_matrix = np.stack(embeddings).astype(np.float32)
    embed_dim = embedding_matrix.shape[1]

    index = faiss.IndexFlatIP(embed_dim)
    index.add(embedding_matrix)

    print(f"\nFAISS index built: {index.ntotal} vectors, dim={embed_dim}")

    # 5. Save
    faiss_path = os.path.join(REFERENCE_IMAGES_DIR, "clip_reference_index.faiss")
    meta_path = os.path.join(REFERENCE_IMAGES_DIR, "clip_reference_metadata.json")

    faiss.write_index(index, faiss_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nIndex saved: {faiss_path}")
    print(f"Metadata saved: {meta_path}")

    # 6. Summary
    print("\n" + "=" * 60)
    print("  Category Distribution")
    print("=" * 60)

    label_counts: Dict[str, int] = {}
    for m in metadata:
        d = m['disease']
        label_counts[d] = label_counts.get(d, 0) + 1

    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * count
        print(f"  {label:30s} {count:3d}  {bar}")

    print(f"\n✅ Reference index ready! ({index.ntotal} images)")
    print("   You can now use image analysis in the app.")


if __name__ == "__main__":
    build_reference_index()