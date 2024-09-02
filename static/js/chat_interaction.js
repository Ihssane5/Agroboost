send = document.querySelector('#send')
chat_box_body =  document.querySelector('#chat_box_body')
input = document.querySelector('#input')

send.addEventListener('click', async function(event) {
        input_user = document.querySelector('#input').value;
        if(input_user != '') {
        /*Create the recieved div*/
        const messageReceiveDiv = document.createElement('div');
        messageReceiveDiv.className = 'message-receive';

        // Create the inner div
        const chatBoxBodyReceiveDiv = document.createElement('div');
        chatBoxBodyReceiveDiv.className = 'chat-box-body-receive';

        // Create and append the paragraph
        const paragraph = document.createElement('p');
        paragraph.textContent = input_user;
        chatBoxBodyReceiveDiv.appendChild(paragraph);

        // Append the inner div to the outer div
        messageReceiveDiv.appendChild(chatBoxBodyReceiveDiv);

        // Create and append the image
        const image = document.createElement('img');
        image.src = '../static/images/humain.png';
        image.alt = '';
        messageReceiveDiv.appendChild(image);

        // Append the messageReceiveDiv to the document body or a specific container
        chat_box_body.appendChild(messageReceiveDiv);
        input.value = ''
        /*sending the input to the model the model*/ 
        const response = await fetch('/chatbot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({input_user})
        });
         result = await response.json();
        /*create the send div*/
        const messageSentDiv = document.createElement('div');
        messageSentDiv.className = 'message-send';

        // Create the inner div
        const chatBoxBodySendDiv = document.createElement('div');
        chatBoxBodySendDiv.className = 'chat-box-body-send';

        // Create and append the paragraph
        const para = document.createElement('p');
        para.textContent = result;
        chatBoxBodySendDiv.appendChild(para);

        // Create and append the image
        const leaf_image = document.createElement('img');
        image.src = '../static/images/leaf.png';
        image.alt = '';
        messageSentDiv.appendChild(image);

        // Append the inner div to the outer div
        messageSentDiv.appendChild(chatBoxBodySendDiv);

        // Append the messageReceiveDiv to the document body or a specific container
        chat_box_body.appendChild(messageSentDiv);

        }


    
});

//clear the chat if the trash button

trash_button = document.querySelector('#trash')
trash.addEventListener('click', function() {
    location.reload();
})

//hide the chat button when clicked
button = document.querySelector('.chat-button');
chatbox=document.querySelector('.chat-box');
button.addEventListener('click', function() {
    button.style.display = 'none';
    chatbox.style.visibility = 'visible';
});

// hide the chatbot window when the del icon is clicked

del_button = document.querySelector('.del');
del_button.addEventListener('click', function() {
    button.style.display = 'block';
    chatbox.style.visibility = 'hidden';
});






