# Enoro

YouTube channel recommendations based on content similarity, not just engagement metrics.

## What it does

Analyzes your YouTube subscriptions to suggest new channels you might actually want to watch, using content analysis and collaborative filtering instead of optimizing for click-through rates.

## Quick Start

**Self-hosted setup:**

1. Get YouTube API credentials from [Google Cloud Console](https://console.cloud.google.com)
2. Copy `.env.example` to `.env` and add your credentials  
3. Run: `uv sync && python run.py`
4. Visit: <http://localhost:8000>

**Hosted version:** Coming soon - no setup required, just login with YouTube

## How it works

- Connects via YouTube OAuth to access your subscriptions
- Generates content tags using NLP analysis
- Builds recommendation models (collaborative + content-based)
- Suggests channels based on similarity to what you already watch

## API Overview

```http
GET /api/v1/auth/youtube/login          # Login with YouTube
GET /api/v1/subscriptions               # Get your subscriptions
POST /api/v1/ml/recommendations/initialize  # Train models
GET /api/v1/ml/recommendations/{user_id}    # Get recommendations
```

## Why this exists

YouTube's algorithm optimizes for engagement and watch time, not discovering quality content in your interests. Enoro focuses on content similarity and user preferences instead of just click-through rates.
