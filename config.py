model_type = 'api' # or 'local'
model = 'deepseek-chat' # local e.g. 'medalpaca/medalpaca-13b'
model_base = 'https://api.deepseek.com/v1' 
model_api_key = 'your_api_key' # dummy if model_type == 'api'

embed_model_type = 'local' # or 'api'
embed_model = 'abhinand/MedEmbed-large-v0.1'
embed_api_key = 'your_api_key' # dummy if embed_model_type == 'api'
embed_model_base = 'https://api.openai.com/v1'  # dummy if embed_model_type == 'api'

temperature = 0.2

YOUR_PATH_TO_PPMI = 'YOUR_PATH_TO_PPMI'
OUTPUT_PATH = 'PPMI_SYN_Data'# modify this
WEIGHT_PATH = 'Models'
EXP_OUTPUT_PATH = 'exp_results'