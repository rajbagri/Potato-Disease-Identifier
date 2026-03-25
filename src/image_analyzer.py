"""
Image Analyzer Module — CLIP-based Disease Classification using Reference Images

How it works:
  1. You manually collect 10-12 reference images per disease/deficiency category
     and place them in reference_images/{Category_Name}/ folders.
  2. Run `python -m src.build_reference_index` ONCE to encode all reference images
     with CLIP and build a small FAISS index (runs 100% locally, zero API cost).
  3. At query time: user uploads an image → CLIP encodes it → cosine similarity
     search against the reference FAISS index → top matches → predicted disease.
  4. The predicted disease is turned into a text query and fed into the existing
     RAG pipeline for a detailed explanation + follow-up questions.

Cost: $0 for classification. Only 1 LLM call when RAG generates the answer.
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
import time

from src.logging_utils import setup_logger, log_timing

logger = setup_logger('image_analyzer')

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────

REFERENCE_IMAGES_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
    "reference_images"
)
CLIP_INDEX_PATH = os.path.join(REFERENCE_IMAGES_DIR, "clip_reference_index.faiss")
CLIP_METADATA_PATH = os.path.join(REFERENCE_IMAGES_DIR, "clip_reference_metadata.json")

# CLIP model — small and fast, runs on CPU in ~50ms per image
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# ──────────────────────────────────────────────────────────────────────
# Disease text prompts for zero-shot (backup/boost to image matching)
# These cost nothing — CLIP encodes them locally
# ──────────────────────────────────────────────────────────────────────

DISEASE_TEXT_PROMPTS = {
    "Late_Blight": [
        "A potato leaf infected with late blight showing dark water-soaked lesions and white sporulation",
        "Potato plant with Phytophthora infestans showing brown necrotic patches on leaves",
        "Potato tuber with late blight showing reddish-brown granular rot beneath the skin",
    ],
    "Early_Blight": [
        "A potato leaf with early blight showing concentric ring target-shaped brown lesions",
        "Potato plant infected with Alternaria solani showing dark spots with concentric rings",
        "Potato foliage with early blight showing yellowing and brown target-spot lesions",
    ],
    "Blackleg": [
        "Potato stem with blackleg disease showing dark black slimy rot at the base",
        "Potato plant with Pectobacterium causing blackleg and wilting stems",
        "Potato seed tuber with bacterial soft rot and blackened stem end",
    ],
    "Bacterial_Soft_Rot": [
        "Potato tuber with bacterial soft rot showing wet mushy decaying tissue",
        "Potato showing soft rot with cream-colored slimy bacterial decay",
        "Potato storage with soft rot caused by Pectobacterium bacteria",
    ],
    "Ring_Rot": [
        "Potato tuber cross-section showing ring rot with vascular discoloration",
        "Potato with bacterial ring rot caused by Clavibacter showing cheesy vascular ring",
        "Potato tuber sliced showing yellow ring rot in vascular tissue",
    ],
    "Common_Scab": [
        "Potato tuber skin with common scab showing rough corky raised lesions",
        "Potato affected by Streptomyces scabies with scab-like patches on surface",
        "Potato tuber with common scab showing rough pitted brown surface lesions",
    ],
    "Powdery_Scab": [
        "Potato tuber with powdery scab showing raised pustules filled with dark powder",
        "Potato skin with powdery scab caused by Spongospora subterranea",
        "Potato tuber with round raised powdery lesions on the skin surface",
    ],
    "Dry_Rot": [
        "Potato tuber with dry rot showing shrunken wrinkled skin and internal cavity",
        "Potato affected by Fusarium dry rot with concentric wrinkles and dry internal decay",
        "Potato in storage showing dry rot with brown dry cavities inside",
    ],
    "Verticillium_Wilt": [
        "Potato plant with verticillium wilt showing one-sided leaf yellowing and wilting",
        "Potato stem cross-section showing vascular browning from verticillium wilt",
        "Potato plant with early dying syndrome from Verticillium dahliae",
    ],
    "Brown_Rot": [
        "Potato tuber with brown rot showing bacterial ooze from eyes and vascular ring",
        "Potato plant with Ralstonia solanacearum causing brown rot and wilting",
        "Potato tuber cross-section with brown discoloration in vascular ring from brown rot",
    ],
    "Charcoal_Rot": [
        "Potato plant with charcoal rot showing grey-black stem discoloration",
        "Potato stem with charcoal rot caused by Macrophomina showing dark microsclerotia",
        "Potato with charcoal rot showing wilting and dark stem lesions",
    ],
    "Rhizoctonia_Black_Scurf": [
        "Potato tuber with black scurf showing dark brown to black hard masses on skin",
        "Potato sprout with Rhizoctonia solani causing stem canker and brown lesions",
        "Potato tuber covered in black sclerotia from Rhizoctonia disease",
    ],
    "Silver_Scurf": [
        "Potato tuber with silver scurf showing silvery metallic sheen on skin surface",
        "Potato skin with Helminthosporium solani causing silver scurf discoloration",
        "Potato tuber with grey-silver patches from silver scurf storage disease",
    ],
    "Virus_Disease": [
        "Potato leaf with virus mosaic showing light and dark green mottled pattern",
        "Potato plant with leaf roll virus showing upward rolling and yellowing of leaves",
        "Potato leaf with potato virus Y showing necrotic stipple and mosaic symptoms",
    ],
    "Nitrogen_Deficiency": [
        "Potato plant with nitrogen deficiency showing uniform pale yellow-green leaves",
        "Potato leaf with nitrogen deficiency showing chlorosis starting from older leaves",
        "Potato foliage with stunted growth and pale yellow leaves from nitrogen deficiency",
    ],
    "Phosphorus_Deficiency": [
        "Potato plant with phosphorus deficiency showing dark green to purple leaf coloration",
        "Potato leaf with phosphorus deficiency showing purplish undersides and stunted growth",
        "Potato plant with P deficiency showing dark leaves with reddish-purple tinge",
    ],
    "Potassium_Deficiency": [
        "Potato leaf with potassium deficiency showing brown scorching at leaf margins",
        "Potato plant with K deficiency showing marginal leaf burn and necrotic edges",
        "Potato foliage with potassium deficiency showing interveinal chlorosis and leaf edge browning",
    ],
    "Magnesium_Deficiency": [
        "Potato leaf with magnesium deficiency showing interveinal chlorosis with green veins",
        "Potato plant with Mg deficiency showing yellow patches between veins on older leaves",
        "Potato leaf with magnesium deficiency showing V-shaped yellowing pattern",
    ],
    "Calcium_Deficiency": [
        "Potato plant with calcium deficiency showing distorted curled young leaves",
        "Potato tuber with internal brown spot from calcium deficiency",
        "Potato leaf with Ca deficiency showing necrotic leaf tips and margins on new growth",
    ],
    "Healthy": [
        "A healthy green potato plant with no visible disease symptoms",
        "Healthy potato leaf with uniform green color and no spots or lesions",
        "Normal potato tuber with smooth clean skin and no disease signs",
    ],
}

# Display-friendly names (folder_name -> pretty name)
DISPLAY_NAMES = {
    "Late_Blight": "Late Blight",
    "Early_Blight": "Early Blight",
    "Blackleg": "Blackleg",
    "Bacterial_Soft_Rot": "Bacterial Soft Rot",
    "Ring_Rot": "Ring Rot",
    "Common_Scab": "Common Scab",
    "Powdery_Scab": "Powdery Scab",
    "Dry_Rot": "Dry Rot",
    "Verticillium_Wilt": "Verticillium Wilt",
    "Brown_Rot": "Brown Rot",
    "Charcoal_Rot": "Charcoal Rot",
    "Rhizoctonia_Black_Scurf": "Rhizoctonia (Black Scurf)",
    "Silver_Scurf": "Silver Scurf",
    "Virus_Disease": "Virus Disease (PVY/PLRV)",
    "Nitrogen_Deficiency": "Nitrogen Deficiency",
    "Phosphorus_Deficiency": "Phosphorus Deficiency",
    "Potassium_Deficiency": "Potassium Deficiency",
    "Magnesium_Deficiency": "Magnesium Deficiency",
    "Calcium_Deficiency": "Calcium Deficiency",
    "Healthy": "Healthy",
}


class CLIPDiseaseAnalyzer:
    """
    CLIP-based potato disease analyzer using manually curated reference images.

    Two complementary matching strategies:
      1. Image-to-Image: user image vs reference images (FAISS index)
      2. Zero-shot text: user image vs disease text descriptions

    Combined with weighted scoring for robust predictions.
    """

    def __init__(self, load_faiss_index: bool = True):
        """
        Args:
            load_faiss_index: If True, load the pre-built FAISS index of reference images.
                              Set False if index hasn't been built yet (zero-shot only mode).
        """
        import torch
        from transformers import CLIPProcessor, CLIPModel

        init_start = time.perf_counter()
        logger.info("Initializing CLIPDiseaseAnalyzer...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.model.eval()
        self.torch = torch

        # Pre-compute disease text embeddings for zero-shot matching
        self.disease_names = list(DISEASE_TEXT_PROMPTS.keys())
        self.disease_text_embeddings = self._encode_disease_prompts()

        # Load pre-built reference image FAISS index
        self.ref_index = None
        self.ref_metadata: List[Dict] = []
        if load_faiss_index:
            self._load_reference_index()

        elapsed = time.perf_counter() - init_start
        log_timing(logger, "CLIP_ANALYZER_INIT", {
            'duration_ms': round(elapsed * 1000, 2),
            'device': self.device,
            'disease_categories': len(self.disease_names),
            'ref_index_loaded': self.ref_index is not None,
            'ref_images_count': len(self.ref_metadata),
        })

    # ──────────────────────────────────────────────────────────────────
    # Encoding helpers
    # ──────────────────────────────────────────────────────────────────

    def _encode_disease_prompts(self) -> np.ndarray:
        """Encode all disease text prompts with CLIP; average per category."""
        logger.info("Encoding disease text prompts...")
        all_embeddings = []

        for disease_name in self.disease_names:
            prompts = DISEASE_TEXT_PROMPTS[disease_name]
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self.torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            avg_emb = text_features.mean(dim=0).cpu().numpy()
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            all_embeddings.append(avg_emb)

        return np.stack(all_embeddings).astype(np.float32)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode a single PIL Image into a CLIP embedding vector."""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.squeeze(0).cpu().numpy().astype(np.float32)

    # ──────────────────────────────────────────────────────────────────
    # Reference FAISS index
    # ──────────────────────────────────────────────────────────────────

    def _load_reference_index(self):
        """Load the pre-built FAISS index of reference images."""
        import faiss

        if not os.path.exists(CLIP_INDEX_PATH) or not os.path.exists(CLIP_METADATA_PATH):
            logger.warning(
                f"Reference FAISS index not found at {REFERENCE_IMAGES_DIR}. "
                "Image-to-image matching disabled. "
                "Run: python -m src.build_reference_index"
            )
            return

        self.ref_index = faiss.read_index(CLIP_INDEX_PATH)

        with open(CLIP_METADATA_PATH, "r", encoding="utf-8") as f:
            self.ref_metadata = json.load(f)

        logger.info(f"Reference FAISS index loaded: {self.ref_index.ntotal} images")

    # ──────────────────────────────────────────────────────────────────
    # Strategy 1: Zero-Shot Text Matching
    # ──────────────────────────────────────────────────────────────────

    def zero_shot_classify(self, image: Image.Image, top_k: int = 5) -> List[Dict]:
        """
        Match user image against disease text descriptions.
        Returns sorted list of {disease, display_name, similarity, confidence}.
        """
        start = time.perf_counter()

        image_emb = self.encode_image(image)
        similarities = image_emb @ self.disease_text_embeddings.T  # cosine sim

        # Temperature-scaled softmax for confidence
        temp = 100.0
        exp_sims = np.exp((similarities - similarities.max()) * temp)
        probs = exp_sims / exp_sims.sum()

        ranked = np.argsort(-probs)
        results = []
        for idx in ranked[:top_k]:
            name = self.disease_names[idx]
            results.append({
                "disease": name,
                "display_name": DISPLAY_NAMES.get(name, name),
                "similarity": float(similarities[idx]),
                "confidence": float(probs[idx]),
            })

        elapsed = time.perf_counter() - start
        log_timing(logger, "ZERO_SHOT_CLASSIFY", {
            'duration_ms': round(elapsed * 1000, 2),
            'top_disease': results[0]['display_name'] if results else 'N/A',
            'top_confidence': round(results[0]['confidence'] * 100, 1) if results else 0,
        })
        return results

    # ──────────────────────────────────────────────────────────────────
    # Strategy 2: Image-to-Image Matching (Reference Images)
    # ──────────────────────────────────────────────────────────────────

    def match_reference_images(self, image: Image.Image, top_k: int = 10) -> List[Dict]:
        """
        Find the most visually similar reference images from the curated dataset.
        Returns list of {image_path, disease, display_name, similarity_score}.
        """
        if self.ref_index is None:
            return []

        start = time.perf_counter()

        image_emb = self.encode_image(image).reshape(1, -1)
        scores, indices = self.ref_index.search(image_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.ref_metadata):
                continue
            meta = self.ref_metadata[idx]
            disease = meta.get("disease", "Unknown")
            results.append({
                "image_path": meta.get("image_path", ""),
                "disease": disease,
                "display_name": DISPLAY_NAMES.get(disease, disease),
                "similarity_score": float(score),
            })

        elapsed = time.perf_counter() - start
        log_timing(logger, "REF_IMAGE_MATCHING", {
            'duration_ms': round(elapsed * 1000, 2),
            'results': len(results),
        })
        return results

    # ──────────────────────────────────────────────────────────────────
    # Combined Analysis Pipeline
    # ──────────────────────────────────────────────────────────────────

    def analyze_image(
        self,
        image: Image.Image,
        top_k_diseases: int = 5,
        top_k_ref_images: int = 10,
        ref_weight: float = 0.7,
        zs_weight: float = 0.3,
    ) -> Dict:
        """
        Full analysis pipeline combining both strategies.

        Weights:
          - ref_weight (0.7): weight for image-to-image reference matching
          - zs_weight  (0.3): weight for zero-shot text matching
          - If no reference index is available, falls back to 100% zero-shot

        Returns:
            {
                'prediction': str,
                'display_name': str,
                'confidence': float,
                'all_candidates': [{disease, display_name, score}],
                'zero_shot_results': [...],
                'matched_ref_images': [...],
                'rag_query': str,
            }
        """
        total_start = time.perf_counter()

        # 1. Zero-shot text matching
        zs_results = self.zero_shot_classify(image, top_k=len(self.disease_names))

        # 2. Reference image matching
        ref_results = self.match_reference_images(image, top_k=top_k_ref_images)

        # 3. Build combined scores
        combined_scores: Dict[str, float] = {}

        # Add zero-shot scores
        for r in zs_results:
            combined_scores[r['disease']] = r['confidence'] * zs_weight

        # If reference index available, add image-to-image voting
        if ref_results:
            # Aggregate reference votes by disease (weighted by similarity)
            ref_votes: Dict[str, float] = {}
            for ref in ref_results:
                d = ref['disease']
                ref_votes[d] = ref_votes.get(d, 0) + ref['similarity_score']

            # Normalize reference votes to probabilities
            total_ref = sum(ref_votes.values())
            if total_ref > 0:
                for d, vote in ref_votes.items():
                    ref_prob = vote / total_ref
                    combined_scores[d] = combined_scores.get(d, 0) + ref_prob * ref_weight
        else:
            # No reference index — use 100% zero-shot
            for r in zs_results:
                combined_scores[r['disease']] = r['confidence']

        # Sort and normalize
        sorted_diseases = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        total_score = sum(s for _, s in sorted_diseases)

        candidates = []
        for disease, score in sorted_diseases[:top_k_diseases]:
            norm_score = score / total_score if total_score > 0 else 0
            candidates.append({
                "disease": disease,
                "display_name": DISPLAY_NAMES.get(disease, disease),
                "score": round(norm_score, 4),
            })

        top_disease = candidates[0]['disease'] if candidates else "Unknown"
        top_display = candidates[0]['display_name'] if candidates else "Unknown"
        top_confidence = candidates[0]['score'] if candidates else 0.0

        # 4. Build RAG query
        rag_query = self._build_rag_query(top_disease, top_display, candidates[:3])

        total_elapsed = time.perf_counter() - total_start
        log_timing(logger, "FULL_IMAGE_ANALYSIS", {
            'duration_ms': round(total_elapsed * 1000, 2),
            'prediction': top_display,
            'confidence': round(top_confidence * 100, 1),
            'ref_images_used': len(ref_results),
            'mode': 'hybrid' if ref_results else 'zero_shot_only',
        })

        return {
            'prediction': top_disease,
            'display_name': top_display,
            'confidence': round(top_confidence, 4),
            'all_candidates': candidates,
            'zero_shot_results': zs_results[:top_k_diseases],
            'matched_ref_images': ref_results,
            'rag_query': rag_query,
        }

    # ──────────────────────────────────────────────────────────────────
    # RAG Query Builder
    # ──────────────────────────────────────────────────────────────────

    def _build_rag_query(
        self,
        top_disease: str,
        top_display: str,
        top_candidates: List[Dict],
    ) -> str:
        """Build a text query from predictions to feed into the existing RAG pipeline."""
        candidate_str = ", ".join([
            f"{c['display_name']} ({c['score']:.0%})" for c in top_candidates
        ])

        deficiency_keywords = ["Deficiency", "Nitrogen", "Phosphorus", "Potassium", "Magnesium", "Calcium"]
        is_deficiency = any(kw in top_disease for kw in deficiency_keywords)

        if top_disease == "Healthy":
            return (
                "The uploaded potato plant image appears healthy with no visible disease symptoms. "
                "What are the best practices for maintaining healthy potato plants and preventing common diseases?"
            )
        elif is_deficiency:
            nutrient = top_display.replace(" Deficiency", "")
            return (
                f"Based on visual analysis, the potato plant shows symptoms consistent with "
                f"{top_display}. Other possible conditions: {candidate_str}. "
                f"What are the symptoms, causes, and management practices for {top_display} in potatoes? "
                f"What fertilization recommendations can correct {nutrient} deficiency?"
            )
        else:
            return (
                f"Based on visual analysis, the potato plant shows symptoms most consistent with "
                f"{top_display}. Other possible conditions: {candidate_str}. "
                f"What are the detailed symptoms, causal organism, favorable conditions, and "
                f"integrated disease management strategies for {top_display} in potato?"
            )