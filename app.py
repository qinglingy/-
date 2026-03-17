import os
import cv2
import uuid
import requests
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO

# ===============================
# 基础路径
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ===============================
# 加载模型
# ===============================
maturity_model = YOLO(os.path.join(MODEL_DIR, "maturity_best.pt"))
disease_model = YOLO(os.path.join(MODEL_DIR, "disease_best.pt"))

print("模型加载成功")

app = Flask(__name__)

# ===============================
# 填入你的通义千问 API KEY
# ===============================
QWEN_API_KEY = "sk-a9d90bdcae1943adb6560d4a88f0714d"

# ===============================
# 调用大模型
# ===============================
def generate_advice(stats):

    maturity = stats.get("maturity", {})
    disease = stats.get("disease", {})

    total_fruits = sum(maturity.values())

    prompt = f"""
    番茄检测结果如下（单位为数量）：

    果实总数：{total_fruits}
    成熟度统计：{maturity}
    病虫害统计：{disease}

    请输出：
    1. 植株健康分析
    2. 是否建议采摘
    3. 病害处理建议
    4. 种植管理建议
    5. 综合风险等级（低/中/高）
    """

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "qwen-turbo",
        "input": {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    }

    try:
        response = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            headers=headers,
            json=data,
            timeout=30
        )
        result = response.json()
        return result["output"]["text"]
    except Exception as e:
        return f"大模型调用失败：{str(e)}"


# ===============================
# 模型推理函数
# ===============================
def run_model(model, image_path, save_name):

    img = cv2.imread(image_path)
    result = model(img)[0]

    stats = {}

    for box in result.boxes:
        cls_id = int(box.cls[0])
        name = result.names[cls_id]
        stats[name] = stats.get(name, 0) + 1

    plotted = result.plot()
    result_path = os.path.join(RESULT_DIR, save_name)
    cv2.imwrite(result_path, plotted)

    return stats, f"/static/results/{save_name}"


# ===============================
# 首页
# ===============================
@app.route("/")
def index():
    return render_template("index.html")


# ===============================
# 预测接口
# ===============================
@app.route("/predict", methods=["POST"])
def predict():

    fruit_file = request.files.get("fruit")
    leaf_file = request.files.get("leaf")

    if not fruit_file and not leaf_file:
        return jsonify({"error": "请至少上传一张图片"})

    stats = {}
    fruit_img_url = None
    leaf_img_url = None

    # 果实图
    if fruit_file and fruit_file.filename != "":
        fruit_name = f"{uuid.uuid4().hex}_fruit.jpg"
        fruit_path = os.path.join(UPLOAD_DIR, fruit_name)
        fruit_file.save(fruit_path)

        maturity_stats, fruit_img_url = run_model(
            maturity_model, fruit_path, fruit_name
        )
        stats["maturity"] = maturity_stats

    # 叶片图
    if leaf_file and leaf_file.filename != "":
        leaf_name = f"{uuid.uuid4().hex}_leaf.jpg"
        leaf_path = os.path.join(UPLOAD_DIR, leaf_name)
        leaf_file.save(leaf_path)

        disease_stats, leaf_img_url = run_model(
            disease_model, leaf_path, leaf_name
        )
        stats["disease"] = disease_stats

    advice = generate_advice(stats)

    return jsonify({
        "fruit_image": fruit_img_url,
        "leaf_image": leaf_img_url,
        "stats": stats,
        "advice": advice
    })


if __name__ == "__main__":
    app.run(debug=True)