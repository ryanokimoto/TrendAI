# TrendAI (TikTok Trend-Aware Agentic Recommendation System)

This project implements a trend-aware agentic recommendation system designed for TikTok creators. By combining multi-modal content understanding through vision and text, niche-specific clustering, and temporal trend dynamics, the system recommends high-potential hashtags to maximize content virality.

### Project Overview

The system architecture is built on multi-modal content understanding, using metadata and vision AI models like LLaVA or GPT-4V to generate deep video descriptions. It employs niche-specific indexing to cluster hashtags into vertical categories, ensuring accurate retrieval. The core of the system is a trend-aware ranking policy, which acts as a decision-making agent to balance semantic relevance, trend velocity, and saturation penalties. Users interact with the system through a Gradio-based interface that provides real-time hashtag recommendations and explanations.

### Core Project Structure

The data collection and multi-modal processing phase involves collect_data.py for scraping and AI summary generation, descriptions_from_metadata.py for text synthesis, and the core module video_vision_descriptions.py, which downloads and analyzes video frames. The embedding and retrieval engine consists of tiktok_search_system.py for vector search management, create_embeddings.py for dataset processing, and tiktok_retrieval_system.py, which implements the multi-index architecture. Analysis and optimization are handled by hashtag_clustering.py for defining content niches and filter.py for refining the hashtag action space. The application layer is managed by run_pipeline_ui.py, supported by the tiktok_vision_index_metadata.json file.

### Installation and Usage

To set up the environment, install the required dependencies using 'pip install pandas numpy torch sentence-transformers faiss-cpu gradio.' For generating vision descriptions, the system requires Ollama to be running with the llava model. The generation process is initiated by running 'python video_vision_descriptions.py' with the appropriate csv and vision arguments. The user interface is launched by executing 'python run_pipeline_ui.py.'

### Recommendation Logic

The system models the recommendation process as an agentic decision pipeline where the final ranking is determined by a policy function. The score for a given hashtag is calculated based on weighted similarity, velocity, and saturation.

$$Score(h) = w_{sim} \cdot Similarity + w_{vel} \cdot Velocity - w_{sat} \cdot Saturation$$

This mathematical approach ensures that recommended hashtags are contextually relevant to the video content while also being positioned on an upward growth trajectory within their respective niches.
