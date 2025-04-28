from flask import Flask, request, jsonify
from app import NexalyzeChatbot
from pymongo import MongoClient
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}, r"/end-session": {"origins": "*"}, r"/get-session-chat": {"origins": "*"}}, supports_credentials=True)

# MongoDB Setup
client = MongoClient("mongodb://admin:cisco123@13.234.241.103:27017/solarflux?authSource=admin")

# client = MongoClient("mongodb+srv://safooraiftikhar:NLJIpEsP6gZptLTz@cluster0.y99bf.mongodb.net/")
db = client["chatbot"]

# Chatbot instance
chatbot = NexalyzeChatbot()
session_memory = {}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id")
    company_id = data.get("company_id")
    user_id = data.get("user_id")

    if session_id not in session_memory:
        session_memory[session_id] = chatbot.get_message_history(session_id)

    response = chatbot.chat(user_input, session_id=session_id)

    # 🟢 Upsert session
    db[company_id].update_one(
        {"session_id": session_id, "user_id": user_id},
        {
            "$setOnInsert": {
                "company_id": company_id,
                "session_id": session_id,
                "user_id": user_id,
                "created_at": datetime.utcnow()
            },
            "$push": {
                "messages": {
                    "user-query": user_input,
                    "bot-response": response
                }
            },
            "$set": { "updated_at": datetime.utcnow() }
        },
        upsert=True
    )

    return jsonify({"response": response})


@app.route('/end-session', methods=['POST'])
def end_session():
    data = request.json
    session_id = data.get("session_id")
    company_id = data.get("company_id")
    user_id = data.get("user_id")
    end_time = data.get("end_time")

    if not all([session_id, company_id, end_time]):
        return jsonify({"status": "error", "message": "Missing data"}), 400

    collection = db[company_id]
    result = collection.update_one(
        {
            "session_id": session_id,
            "user_id": user_id
        },
        {
            "$set": {"session_end_time": end_time, "updated_at": datetime.utcnow()}
        }
    )

    if result.modified_count:
        return jsonify({"status": "success", "message": "Session end time saved"}), 200
    else:
        return jsonify({"status": "warning", "message": "Session not found or not updated"}), 404


@app.route('/get-session-chat', methods=['POST'])
def get_session_chat():
    data = request.json
    session_id = data.get("session_id")
    company_id = data.get("company_id")

    if not session_id or not company_id:
        return jsonify({"error": "Missing session_id or company_id"}), 400

    collection = db[company_id]
    record = collection.find_one({"session_id": session_id}, {"messages": 1, "_id": 0})
    
    if record:
        return jsonify({"messages": record.get("messages", [])})
    else:
        return jsonify({"messages": []})

if __name__ == '__main__':
    app.run(debug=True)
