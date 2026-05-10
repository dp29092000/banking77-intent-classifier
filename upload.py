from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="banking77_model",
    repo_id="dp29092k/banking77-intent-classifier",
    repo_type="model"
)