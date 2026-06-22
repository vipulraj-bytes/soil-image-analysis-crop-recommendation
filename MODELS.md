# Model Files Guide

## Required Model Files

This project requires the following model files to run predictions:

| File Name | Size | Purpose |
|-----------|------|---------|
| `best_soil_cnn_model.pkl` | ~196 MB | CNN model for soil classification |
| `best_soil_resnet18_finetuned.pth` | ~50 MB | ResNet-18 fine-tuned model |
| `label_encoder.pkl` | <1 MB | Label encoder for soil types |
| `soil_model.pkl` | ~196 MB | Alternative soil classification model |

**Total Size**: ~500 MB

## Why are models not in the repository?

GitHub has a file size limit of 100 MB per file. The model files exceed this limit, so they are excluded from version control using `.gitignore`.

## How to get the models?

### Option 1: Use Pre-trained Models (Recommended)

If you just want to run the application:

1. **Contact the repository owner** to request the model files
2. Download the model files to your local machine
3. Place them in the `backend/` directory
4. Verify the files are in the correct location:
   ```
   backend/
   ├── best_soil_cnn_model.pkl
   ├── best_soil_resnet18_finetuned.pth
   ├── label_encoder.pkl
   └── soil_model.pkl
   ```

### Option 2: Train Models Yourself

If you want to train the models from scratch:

1. **Obtain a soil image dataset** with the following classes:
   - Alluvial
   - Black
   - Clay
   - Loamy
   - Red
   - Saline
   - Sandy

2. **Organize the dataset**:
   ```
   backend/soil_dataset/
   ├── train/
   │   ├── Alluvial/
   │   ├── Black/
   │   ├── Clay/
   │   ├── Loamy/
   │   ├── Red/
   │   ├── Saline/
   │   └── Sandy/
   └── test/
       ├── Alluvial/
       ├── Black/
       ├── Clay/
       ├── Loamy/
       ├── Red/
       ├── Saline/
       └── Sandy/
   ```

3. **Run the training script**:
   ```bash
   cd backend
   python train_cnn.py
   ```

4. **Training will generate**:
   - Model files in the `backend/` directory
   - Training logs and metrics

### Option 3: Use Cloud Storage

For collaborators and users, consider hosting model files on:

- **Google Drive**: Share a public link
- **Dropbox**: Public folder link
- **GitHub Releases**: Attach files to a release (up to 2GB per file)
- **Hugging Face Hub**: Host ML models
- **AWS S3**: Public bucket with download script

## Model Performance

| Model | Accuracy | Training Time | Best Use Case |
|-------|----------|---------------|---------------|
| CNN Model | TBD | ~2-3 hours | General soil classification |
| ResNet-18 | TBD | ~3-4 hours | High accuracy requirements |

## Troubleshooting

**Error: Model file not found**
- Verify model files are in the `backend/` directory
- Check file names match exactly (case-sensitive)

**Error: Incompatible model version**
- Ensure you're using the same PyTorch/scikit-learn versions as in `requirements.txt`
- Retrain models if necessary

**Out of memory during training**
- Reduce batch size in `train_cnn.py`
- Use a machine with more RAM/GPU memory
- Train on a cloud platform (Colab, AWS, Azure)

## Next Steps

After obtaining the model files:
1. Follow the main README.md installation guide
2. Run `python app.py` to start the application
3. Test predictions with sample soil images

For questions or model file requests, please open an issue on GitHub.
