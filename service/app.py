from flask import Flask, send_file, render_template_string, request, send_from_directory
import threading, time
from selenium import webdriver
from flask_cors import CORS
from eval import run_model
# app = Flask(__name__, static_folder='../build', static_url_path='')
app = Flask(__name__)
# driver = webdriver.Chrome()
screenshot_path = "static/screenshot.png"
CORS(app)


@app.route("/start", methods=['POST'])
def start():
    try:
        data = request.get_json()
        
        model_file = f'models/dino_{data.get("modelType")}_model_{data.get("rewardFunction")}_{data.get("step")}'
        run_model(model_file, type=data.get("modelType"))
        return "Model run started", 200
    except Exception as e:
        print(f"Error: {e}")
        return str(e), 500
    
    # threading.Thread(target=evaluate_model, args=(model_file, num_episodes)).start()

# @app.route("/monitor")
# def monitor():
#     return render_template_string("""
#         <html><head><meta http-equiv="refresh" content="0.1"></head><body>
#         <h4>Game Monitor</h4>
#         <img src="/screen" style="width:100%; max-width:600px;">
#         </body></html>
#     """)

@app.route("/video")
def video():
    return send_file("static/output.mp4", mimetype='video/mp4')


if __name__ == "__main__":
    app.run(debug=True, port=5050)
