# Code Alpha Internship AI Demo

This workspace implements two tasks from the internship prompt:

1. Language Translation Tool
2. FAQ Chatbot using NLP similarity matching
3. Real-time Object Detection and Tracking

## Features

- Streamlit UI with separate tabs for translation and chatbot
- Microsoft Translator API integration
- Google Translate API integration
- Demo translation mode for local testing via LibreTranslate
- FAQ matching with NLTK tokenization, stopword removal, lemmatization, TF-IDF, and cosine similarity

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Translation API Configuration

### Microsoft Translator

Set these environment variables:

- `TRANSLATOR_KEY`
- `TRANSLATOR_REGION`

Optional:

- `TRANSLATOR_ENDPOINT`

You can place them in a local `.env` file in the project root:

```env
TRANSLATOR_KEY=your_key_here
TRANSLATOR_REGION=your_region_here
TRANSLATOR_ENDPOINT=https://api.cognitive.microsofttranslator.com
```

### Google Translate API

Set this environment variable:

- `GOOGLE_TRANSLATE_API_KEY`

Or add it to the same `.env` file:

```env
GOOGLE_TRANSLATE_API_KEY=your_google_key_here
```

### Demo Mode

Demo mode uses a public LibreTranslate endpoint and does not require credentials.

## FAQ Bot Data

The FAQ bot reads from `faq_data.json`. You can replace the sample Code Alpha internship FAQs with your own topic or product FAQs.

## Object detection & tracking (new)

A simple real-time object detection and centroid-tracking script has been added: `object_detection_tracking.py`.

Run it with your webcam:

```bash
python object_detection_tracking.py --source 0
```

Or with a video file:

```bash
python object_detection_tracking.py --source path/to/video.mp4
```

The script uses YOLO-World by default so you can include custom everyday labels
such as mobile, cloth, shirt, pants, cap, and person. You can pass your own
comma-separated labels:

```bash
python object_detection_tracking.py --source 0 --world-classes "person,mobile phone,clothes,shirt,pants,cap,laptop,bottle"
```

For the older COCO YOLOv5 model:

```bash
python object_detection_tracking.py --source 0 --backend yolov5 --model yolov5m
```

Install dependencies first: `pip install -r requirements.txt` (see note about `torch` on Windows; use the appropriate wheel or follow PyTorch install instructions if needed).
