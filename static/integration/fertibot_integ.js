//hideErrorMessage()
document.getElementById('send-button').addEventListener('click', async function() {
    const input = document.getElementById('message-input');
    const messageText = input.value.trim();
        if (messageText) {
            const chatMessages = document.getElementById('chat-messages');
            chatElement = addChatMessage(messageText, 'chat-message','sent')
            chatMessages.appendChild(chatElement)
            showLoading(chatMessages)
            input.value = '';
            chatMessages.scrollTop = chatMessages.scrollHeight;
            user_question = messageText
            try {
            const response = await fetch('/fertibot', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({user_question})
            });
            console.log("request_send")
                result = await response.json();
                hideLoading(chatMessages);
                
                /*model response*/
                console.log(result)
                received_message = addChatMessage(result, 'chat-message','received')
                chatMessages.appendChild(received_message)
            }
            catch (error) {
                errorMessage = 'Somehow the server is unable to respond right now. Please try our other services and contact us if you need more guidance or if it\'s urgent.'
                hideLoading(chatMessages)
                console.log("error occured")
                displayErrorMessage(chatMessages, errorMessage) 
            }
        }
});


    // Function to show the loading animation
function showLoading(chatMessages) {
    const loadingElement = document.createElement('div');
        loadingElement.className = 'loading chat-message';
        loadingElement.innerHTML = `
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        `;
        chatMessages.appendChild(loadingElement);
    }


// Function to hide the loading animation
function hideLoading(chatMessages) {
    const loadingElement = document.querySelector('.loading.chat-message');
    if (loadingElement) {
        chatMessages.removeChild(loadingElement);
    }
}


// Function to add a new chat message
function addChatMessage(message, className1,className2) {
    const messageElement = document.createElement('div');
    messageElement.classList.add(className1, className2);
    messageElement.textContent = message;
    return messageElement
}

function displayErrorMessage(chatMessages, errorMessage) {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger';
    alertDiv.id = 'error-message';
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `❗` + errorMessage;
    chatMessages.appendChild(alertDiv)
}

/*function hideErrorMessage() {
    const errorContainer = document.getElementById('error-message');
    errorContainer.style.display = 'none'
}*/
    