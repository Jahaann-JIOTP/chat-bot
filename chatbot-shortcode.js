function sendToChatbot() {
    const userInput = document.getElementById("user_input").value;
    const chatlog = document.getElementById("chatlog");

    chatlog.innerHTML += "<p><strong>You:</strong> " + userInput + "</p>";
    document.getElementById("user_input").value = "";

    fetch(chatbot_ajax.api_url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            message: userInput,
            session_id: "wp_user_session"
        })
    })
    .then(response => response.json())
    .then(data => {
        chatlog.innerHTML += "<p><strong>Marcus:</strong> " + data.response + "</p>";
        chatlog.scrollTop = chatlog.scrollHeight;
    })
    .catch(error => {
        chatlog.innerHTML += "<p><strong>Error:</strong> " + error + "</p>";
    });
}
