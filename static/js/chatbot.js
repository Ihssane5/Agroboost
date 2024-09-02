document.addEventListener('scroll', function() {
    const chatBox = document.querySelector('.chat-box');
    const scrollPosition = window.scrollY;

    // Adjust the position of the chat box based on scroll position
    chatBox.style.bottom = `${250 + scrollPosition}px`;
});
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

/*Adjust the chat-box position*/





