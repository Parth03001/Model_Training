# import torch
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# import os

# # Force disable symlinks
# os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
# os.environ['HF_DISABLE_XET'] = "1"

# # Set cache directory
# cache_dir = "./model_cache"
# os.makedirs(cache_dir, exist_ok=True)

# print("Downloading with manual cache...")

# # Download processor
# processor = AutoImageProcessor.from_pretrained(
#     'google/efficientnet-b3',
#     cache_dir=cache_dir,
#     force_download=True)
# print("✓ Processor ready")

# # Download model
# print("Downloading model (this may take 5-10 minutes)...")
# model = AutoModelForImageClassification.from_pretrained(
#     'google/efficientnet-b3',
#     num_labels=2,
#     ignore_mismatched_sizes=True,
#     cache_dir=cache_dir,
#     force_download=True,
#     resume_download=True  # Resume if interrupted
# )
# print("✓ Model ready")


# # import requests
# # print(requests.get('https://huggingface.co').status_code)



import os
import ssl
import requests
from huggingface_hub import configure_http_backend

# Disable SSL verification
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Configure SSL
ssl._create_default_https_context = ssl._create_unverified_context

# Configure requests to skip SSL verification
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

# Suppress SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Now import and use transformers
from transformers import AutoModelForImageClassification

print("Downloading model...")
model = AutoModelForImageClassification.from_pretrained('google/efficientnet-b3')
print("Model downloaded successfully!")