# Final Capstone Presentation Outline
**Time Limit:** 15 Minutes
**Speakers:** Student A & Student B

---

## Slide 1: Title Slide
- **Title:** From Polygon Soup to Editable Scenes
- **Subtitle:** Dual-Input Pipeline for Compositional 3D Asset Reconstruction
- **Names:** Student A, Student B
- **Guide:** Prof. Archana Bhise
- *Visual:* Split image showing 4 input photos → final parsed USD mesh in Maya.

## Slide 2: The Problem
- **Current State:** Generating 3D meshes from images is fast, but yields a single fused "polygon soup".
- **The Issue:** 3D Artists cannot easily edit, rig, or animate these fused meshes.
- **The Gap:** Manual segmentation takes hours.
- *Visual:* Monolithic messy wireframe vs. clean multi-color segmented mesh.

## Slide 3: Our Solution & Objectives
- **Objective:** Fully automate the conversion of sparse 2D images into segmented, production-ready 3D assets.
- **Key Deliverables:** 
  1. A 5-stage automated pipeline.
  2. Output as standard OpenUSD with layer toggles.
  3. Optimized for consumer hardware (8GB VRAM).

## Slide 4: Pipeline Overview (Architecture)
- *Visual:* Block Diagram of the 5 Stages
- Stage 0: Ingest (Padding & Normalization)
- Stage 1: Segment (SAM 2)
- Stage 2: Reconstruct (CRM)
- Stage 3: Partition (SAMPart3D)
- Stage 4: Export (OpenUSD)
- *Speaker Note:* Student A covers Stages 0-2; Student B covers 3-4.

## Slide 5: Deep Dive - Reconstruction (Stage 1 & 2)
- **SAM 2:** Why we chose it for silhouette consistency.
- **CRM:** Why CRM beats Zero123 in speed.
- *Visual:* Input images -> Silhouettes -> Gray Monolithic Mesh.

## Slide 6: Deep Dive - Slicing (Stage 3)
- **The Challenge:** How to find semantic parts without text prompts.
- **The Solution:** DINOv2 distillation via PointTransformer V3.
- *Visual:* Heatmap of DINO features mapping to the 3D surface.

## Slide 7: Deep Dive - USD Export (Stage 4)
- **Why OpenUSD?** It's the industry standard for Pixar, Apple, NVIDIA.
- **Hierarchy:** How we structured the parts into visibility layers.
- *Visual:* Maya Outliner showing the cleanly named parts.

## Slide 8: Evaluation Metrics
- **Chamfer Distance:** Geometric fidelity.
- **IoU:** Semantic boundaries.
- **FPS:** Maya viewport performance (achieved >30 FPS).
- *Visual:* Graph comparing monolithic vs. sliced performance.

## Slide 9: Demo Video
- A 60-second screen recording of the Web UI and Autodesk Maya import.
- *Speaker Note:* Talk through the video live, highlighting the drag-and-drop feature and the Maya layers.

## Slide 10: Limitations & Future Scope
- **Limitations:** Struggles with very thin structures (wires, hair); heavy VRAM spikes during PTv3.
- **Future Scope:** Adding automated UV unwrapping and per-part texture generation.

## Slide 11: Conclusion
- Summary of the achievement: Bridged the gap between AI generation and DCC workflows.
- Thank you & Q&A.
