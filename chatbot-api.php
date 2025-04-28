<?php
/*
Plugin Name: Nexalyze Chatbot Integration
Description: Connects WordPress site with Nexalyze AI Chatbot via API.
Version: 1.0
Author: Your Name
*/

function chatbot_enqueue_scripts() {
    wp_enqueue_script('chatbot-shortcode', plugin_dir_url(__FILE__) . 'chatbot-shortcode.js', array('jquery'), null, true);
    wp_localize_script('chatbot-shortcode', 'chatbot_ajax', array(
        'api_url' => 'http://127.0.0.1:5000/chat' 
    ));
}
add_action('wp_enqueue_scripts', 'chatbot_enqueue_scripts');

function chatbot_shortcode() {
    return '<div id="chatbox">
                <div id="chatlog" style="height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 10px;"></div>
                <input type="text" id="user_input" placeholder="Type your message..." style="width: 80%;" />
                <button onclick="sendToChatbot()">Send</button>
            </div>';
}
add_shortcode('nexalyze_chatbot', 'chatbot_shortcode');
