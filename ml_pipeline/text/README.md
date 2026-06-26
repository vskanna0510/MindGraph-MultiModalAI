# MindGraph++ Text Pipeline

Multilingual text engineering pipeline for depression-related conversational data.

## Stages

1. Load transcript
2. Encoding validation
3. Cleaning
4. Normalization
5. Language detection
6. Code-mix detection
7. Sentence segmentation
8. Tokenization
9. POS tagging
10. Dependency parsing
11. NER
12. Sentiment analysis
13. Emotion detection
14. Psycholinguistic features
15. Mental health keywords
16. Transformer embeddings
17. Embedding normalization
18. Quality assessment
19. Validation & export

## Usage

```bash
make text-init
make text-pipeline
python scripts/text/run_text_pipeline.py --dataset daic_woz --limit 5
```

## Datasets

- `DAICTextDataset` — DAIC-WOZ `*_TRANSCRIPT.csv`
- `DVLOGTextDataset` — optional sidecar transcripts

## Configuration

`configs/text_pipeline.yaml`
