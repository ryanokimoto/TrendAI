# TrendAI — Trend-Aware Agentic Recommendation System for TikTok Creators

TrendAI is a trend-aware agentic recommendation system that helps TikTok creators discover high-potential hashtags to maximize content virality.
The system combines multi-modal video understanding, niche-aware retrieval, and temporal trend modeling to recommend hashtags that are both contextually relevant and emerging in popularity.

## System Architecture

Multi-stage agentic recommendation pipeline:

1. Video Understanding  
   - Metadata summarization  
   - Vision-language caption generation (LLaVA / GPT-4V)

2. Embedding + Retrieval  
   - Sentence-BERT embeddings  
   - FAISS vector search  
   - Multi-index niche retrieval  

3. Trend-Aware Ranking Policy  
   - Semantic similarity scoring  
   - Trend velocity modeling  
   - Saturation penalty  

4. Recommendation Interface  
   - Real-time hashtag suggestions via Gradio UI  
   - Explanation-aware recommendations  

## Recommendation Policy

Each hashtag is scored using a trend-aware objective:

Score(h) = w_sim · Similarity  
         + w_vel · Velocity  
         − w_sat · Saturation  

## File Structure

### Data and Multi-Modal Processing

- `collect_data.py` — TikTok scraping and AI summary generation  
- `descriptions_from_metadata.py` — metadata text synthesis  
- `video_vision_descriptions.py` — frame extraction and vision captioning  

### Embedding and Retrieval Engine

- `create_embeddings.py` — dataset embedding generation  
- `tiktok_search_system.py` — FAISS vector search management  
- `tiktok_retrieval_system.py` — multi-index retrieval architecture  

### Analysis and Optimization

- `hashtag_clustering.py` — niche discovery via co-occurrence graph  
- `filter.py` — action-space pruning  

### Application Layer

- `run_pipeline_ui.py` — Gradio interface  
- `tiktok_vision_index_metadata.json` — retrieval metadata

## Installation

Install dependencies:

pip install pandas numpy torch sentence-transformers faiss-cpu gradio

For vision captioning:

ollama run llava

---

## Quick Start

Generate video descriptions:

python video_vision_descriptions.py --csv dataset.csv --vision  

Launch the recommendation UI:

python run_pipeline_ui.py  

---

## Example Recommendation

Input video description:

visiting magic kingdom at walt disney world and exploring
the theme park rides and attractions during our vacations 

Recommended hashtags:

#disneyland (similarity=0.45)
#wdw (similarity=0.32)
