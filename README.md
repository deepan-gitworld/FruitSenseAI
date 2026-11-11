# FruitSenseAI

**FruitSenseAI** is a vision-based fruit quality analysis project. It classifies fruit images as fresh or rotten, highlights image regions that influence the prediction, and generates a plain-language explanation.

## Features

- Classifies five fruit types: apples, bananas, grapes, mangoes, and oranges
- Predicts fresh or rotten condition across **10 classes**
- Uses **Grad-CAM** heatmaps to show image regions that influenced the CNN prediction
- Uses **SmolVLM-500M-Instruct** to describe visible fruit characteristics
- Uses **Google Gemini** to generate concise, user-friendly explanations
- Supports spoken output with **Sarvam AI** in English, Tamil, and Hindi
- Provides an interactive web interface built with Streamlit

## Model and technology

| Component | Technology | Role |
|---|---|---|
| Fruit classifier | MobileNetV3Large | Predicts fruit type and freshness |
| Deep learning framework | PyTorch | Model training and inference |
| Explainability | Grad-CAM | Visualizes regions influencing the CNN prediction |
| Image description | SmolVLM-500M-Instruct | Describes visible fruit characteristics |
| Text explanation | Google Gemini API | Turns model evidence into a concise explanation |
| Translation and speech | Sarvam AI | Translates explanations and generates speech |
| User interface | Streamlit | Image upload and result display |

**MobileNetV3Large** was selected for its classification performance and compact architecture. The active training configuration trains this model for the current 10-class dataset.

## Dataset

The dataset contains RGB images of five fruits, each labeled as fresh or rotten. It was organized into training and test folders. The training script reserves part of the training data for validation.

| Fruit | Training images | Test images | Total |
|---|---:|---:|---:|
| Apple | 1,116 | 279 | 1,395 |
| Banana | 1,105 | 276 | 1,381 |
| Grape | 1,120 | 280 | 1,400 |
| Mango | 1,114 | 279 | 1,393 |
| Orange | 1,127 | 282 | 1,409 |
| **Total** | **5,582** | **1,396** | **6,978** |

The training script splits the 5,582 training images into **4,744 training images** and **838 validation images**. The test set remains separate.

The dataset folders should follow this structure:

```text
data/processed/dataset/
├── train/
│   ├── freshapples/
│   ├── rottenapples/
│   ├── freshbanana/
│   ├── rottenbanana/
│   ├── freshgrapes/
│   ├── rottengrapes/
│   ├── freshmango/
│   ├── rottenmango/
│   ├── freshoranges/
│   └── rottenoranges/
└── test/
    └── (the same class folders as train/)
```

Dataset files are excluded from Git and are not included in the repository. The supplied dataset contained Fresh and Rotten folders; it did not include a formalin-mixed class.

## Setup

The project requires **Python 3.12** and uses `uv` to manage dependencies.

```bash
uv sync
```

Create a `.env` file in the project root to enable Gemini explanations and Sarvam translation and speech:

```env
GEMINI_API_KEY=your_gemini_api_key
SARVAM_API_KEY=your_sarvam_api_key
```

Keep API keys private and do not commit the `.env` file. Gemini explanations can fall back to a rule-based explanation if the Gemini key is unavailable. Sarvam translation and speech require a Sarvam API key.

SmolVLM downloads its model files the first time it runs. Model speed depends on the available device; the app uses MPS or CUDA when available and otherwise runs on the CPU.

## Train the model

From the project root, run:

```bash
uv run python src/02_train_compare_models.py
```

The trained checkpoint and reports are saved under:

```text
outputs/six_model_comparison/
```

The Streamlit app expects the MobileNetV3Large checkpoint at:

```text
outputs/six_model_comparison/checkpoints/MobileNetV3Large_best.pth
```

Retrain the model after changing the class folders. An existing checkpoint cannot predict classes that it was not trained on.

## Run the app

```bash
uv run streamlit run src/app.py
```

Upload a fruit image or select a sample image, then choose **Analyze Fruit Quality** to view the prediction, Grad-CAM heatmap, visual description, and generated explanation.

## Limitations

- The system analyzes visible surface characteristics in images. It does not detect internal spoilage, bacteria, or chemical contamination, and it does not certify food safety.
- Grad-CAM visualizes regions that influenced the CNN prediction; it does not prove that a prediction is correct.
- Results may vary with image quality, lighting, fruit variety, and camera conditions.