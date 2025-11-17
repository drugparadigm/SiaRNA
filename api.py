from flask import Flask,request, Response,jsonify
import subprocess
from datetime import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import shutil
import traceback
from src.infer import main
from src.transformer_util import check_and_download_model
import torch


# --- Model Loading and Download Logic ---
# Directory where the model will be saved locally
LOCAL_MODEL_DIR = "src/data/input/t5_local"
# LOCAL_MODEL_DIR = "/raid/apitempfiles/siarna/t5_local"
# The model identifier on Hugging Face Model Hub
MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"
# Determine the device for the model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

app = Flask(__name__)
# with app.app_context():
#     downloaded, message = check_and_download_model()
#     app.logger.info(f"[warmup] downloaded={downloaded} message='{message}' device={DEVICE}")

UPLOAD_FOLDER = 'src/data/input'

@app.route('/score', methods=['POST'])
def score():
    downloaded, message = check_and_download_model()
    app.logger.info(f"Model status: {message}, device={DEVICE}")
    try:
        data = request.files
        if 'reqId' not in request.form:
            raise ValueError ("Missing reqId in form data")
        reqId=request.form.get('reqId')
        
        values=['siRNA','mRNA']
        for value in values:
            if value not in data:
                raise ValueError (f"Missing {value} in form data")
        
        file_paths = []
        for value in values:
            file = request.files[value]
            if file:
                if os.path.splitext(file.filename)[1] != '.fa':
                    raise ValueError(f"Invalid file type for {value}. Expected .fa file.")
                else:
                    file_path = os.path.join(UPLOAD_FOLDER, f"{reqId}_{value}.fa")
                    file_paths.append(file_path)
                    file.save(file_path)
        # main_dir = os.getcwd()
        # os.chdir('src')
            
        result=main()
        # os.chdir(main_dir)

        return jsonify({"message": "Inference completed successfully", "output": float(result),}), 200
    
    except (RuntimeError,FileNotFoundError, ValueError,ImportError) as e:
        traceback.print_exc()
        return jsonify({"message":"Inference failed","error": f"{e}"}), 500


    finally:
        if len(os.listdir(UPLOAD_FOLDER)) > 0:
            for file in os.listdir(UPLOAD_FOLDER):
                if os.path.isfile(os.path.join(UPLOAD_FOLDER, file)):
                    if file.startswith(reqId+'_'):
                        file_path = os.path.join(UPLOAD_FOLDER, file)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                else:
                    if file.startswith(reqId+'_'):
                        dir_path = os.path.join(UPLOAD_FOLDER, file)
                        if os.path.exists(dir_path):
                            shutil.rmtree(dir_path)

@app.route('/health/<sample>', methods=['GET'])
def samplescore(sample) -> Response:
 
    if sample == "hi":
        date=datetime.now().strftime("%H:%M:%S")
        return f"Hello {date}"
    else:
        return jsonify({'error':"Unauthorized access"})



if __name__ == '__main__':
    # app.run(host='0.0.0.0',port=5000,debug=False,use_reloader=False)
    app.run(host='0.0.0.0', port=5000,debug=True)