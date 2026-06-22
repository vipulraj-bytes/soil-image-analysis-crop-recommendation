// static/js/script.js

document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded and parsed. Initializing scripts...");

    // Helper function to get element
    function getElement(id) {
        return document.getElementById(id);
    }

    // --- Homepage ---
    const marketTickerData = getElement('market-ticker-data');
    const homePageIdentifier = document.querySelector('.hero-section');

    function loadMarketTicker() {
        if (!marketTickerData) return; // Only run on homepage
        console.log("Attempting to fetch market ticker data...");
        marketTickerData.innerHTML = '<span>சந்தை விலைகள் ஏற்றப்படுகின்றன...</span>';
        fetch('/api/market_ticker_data')
            .then(response => response.ok ? response.json() : response.json().then(err => { throw new Error(err.error || `HTTP error ${response.status}`) }))
            .then(data => {
                console.log("Received market ticker data:", data);
                let tickerHtml = '';
                if (data.ticker_items && Array.isArray(data.ticker_items) && data.ticker_items.length > 0) {
                    data.ticker_items.forEach(itemString => {
                        tickerHtml += `<span>${itemString}</span>`;
                    });
                } else { tickerHtml = '<span>சந்தை விலை தரவு கிடைக்கவில்லை.</span>'; }
                marketTickerData.innerHTML = tickerHtml;
            })
            .catch(error => { console.error("Error fetching market ticker data:", error); marketTickerData.innerHTML = `<span>சந்தை விலைகளைப் பெற முடியவில்லை.</span>`; });
    }

    // --- Chatbot ---
    const chatLog = getElement('chat-log');
    const chatInput = getElement('chat-input');
    const chatSendButton = getElement('chat-send-btn');
    const chatOptionsContainer = getElement('chat-options'); // Container for buttons
    const loadingIndicatorChat = getElement('loading-indicator-chat');

    let currentChatState = 'START'; // Track conversation state on the client

    // Function to add messages to the chat log
    function addMessageToChatLog(message, sender) {
        if (!chatLog) return;
        const messageElement = document.createElement('div');
        messageElement.classList.add('message-bubble', sender === 'user' ? 'user-bubble' : 'bot-bubble');
        messageElement.textContent = message; // Use textContent for safety
        chatLog.appendChild(messageElement);
        chatLog.scrollTop = chatLog.scrollHeight; // Scroll to bottom
    }

    // Function to display options as buttons
    function displayOptions(options) {
        if (!chatOptionsContainer) return;
        chatOptionsContainer.innerHTML = ''; // Clear previous options
        if (!options || options.length === 0) {
            chatOptionsContainer.style.display = 'none'; // Hide if no options
            return;
        }

        options.forEach(option => {
            const button = document.createElement('button');
            button.classList.add('btn', 'btn-outline-primary', 'btn-sm', 'm-1');
            button.textContent = option.label; // Text displayed on button
            button.dataset.value = option.value; // Value sent to backend when clicked
            button.addEventListener('click', () => {
                const selectedValue = button.dataset.value;
                console.log(`Option clicked: Label='${option.label}', Value='${selectedValue}'`);
                // Add user's choice visually (optional, but good UX)
                addMessageToChatLog(option.label, 'user');
                // Send the selected value and current state to the backend
                sendMessageToBot(selectedValue);
            });
            chatOptionsContainer.appendChild(button);
        });
        chatOptionsContainer.style.display = 'flex'; // Show the options container
    }

    // Function to send message (text or button value) to backend
    function sendMessageToBot(message) {
        if (!message || message.trim() === '') return;

        console.log(`Sending to backend: Message='${message}', State='${currentChatState}'`);
        if (loadingIndicatorChat) loadingIndicatorChat.style.display = 'flex';
        if (chatInput) chatInput.disabled = true; // Disable input during request
        if (chatSendButton) chatSendButton.disabled = true;
        if (chatOptionsContainer) chatOptionsContainer.innerHTML = ''; // Clear options while waiting

        fetch('/chatbot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // Send current state along with the message/value
            body: JSON.stringify({ message: message, state: currentChatState })
         })
        .then(response => response.ok ? response.json() : response.json().then(err => { throw new Error(err.response || `Server error ${response.status}`) }))
        .then(data => {
            console.log("Received from backend:", data);
            addMessageToChatLog(data.response, "bot"); // Display bot's text response
            currentChatState = data.next_state || 'START'; // Update client state
            console.log("Updated client state to:", currentChatState);
            displayOptions(data.options || []); // Display new options/buttons
        })
        .catch(error => {
            console.error("Chatbot fetch error:", error);
            addMessageToChatLog(`மன்னிக்கவும், பிழை ஏற்பட்டது: ${error.message}`, "bot");
            // Reset state on error might be helpful
            currentChatState = 'START';
            displayOptions([{"label": "மீண்டும் முயற்சிக்க?", "value": "start_over"}]); // Offer retry option
        })
        .finally(() => {
            if (loadingIndicatorChat) loadingIndicatorChat.style.display = 'none';
            if (chatInput) chatInput.disabled = false; // Re-enable input
            if (chatSendButton) chatSendButton.disabled = false;
            if (chatLog) chatLog.scrollTop = chatLog.scrollHeight;
            if (chatInput && currentChatState !== 'AWAITING_BUTTON_ONLY') { // Only focus if free text might be expected
                 // chatInput.focus(); // Might be annoying if only buttons are expected
            }
        });
    }

    // --- Initialize Chatbot ---
    if (chatLog && chatInput && chatSendButton && chatOptionsContainer) {
        console.log("Chatbot elements found. Initializing state.");
        // Send an initial empty message to get the starting prompt and options
        // We pass 'START' explicitly, though backend defaults to it too
        sendMessageToBot("INIT_CHAT"); // Use a special message or just trigger initial state

        // Handle sending text from input field (less emphasis now)
        chatSendButton.addEventListener('click', () => {
            const userText = chatInput.value.trim();
            if (userText) {
                 addMessageToChatLog(userText, 'user'); // Show user text
                 sendMessageToBot(userText); // Send text to backend
                 chatInput.value = ''; // Clear input
            }
        });
        chatInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                const userText = chatInput.value.trim();
                if (userText) {
                    addMessageToChatLog(userText, 'user');
                    sendMessageToBot(userText);
                    chatInput.value = '';
                }
            }
        });
    } else {
        console.log("Chatbot elements not found on this page.");
    }

    // --- Run homepage functions ONLY if on homepage ---
    if (homePageIdentifier) {
        console.log("Homepage detected.");
        loadMarketTicker();
        // loadTipOfDay(); // Add if you want tip on homepage
    } else {
         console.log("Not on homepage.");
    }

    console.log("Script initialization complete.");

}); // --- End of DOMContentLoaded ---