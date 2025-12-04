const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const themeToggle = document.getElementById('theme-toggle');
const html = document.documentElement;

let history = [];
let currentController = null;
let isGenerating = false;

// Icons
const sendIcon = '<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>';
const stopIcon = '<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M6 6h12v12H6z"/></svg>';

// Theme Toggle
themeToggle.addEventListener('click', () => {
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', newTheme);
    themeToggle.textContent = newTheme === 'dark' ? '☀️' : '🌙';
});

function setButtonState(state) {
    if (state === 'generating') {
        sendBtn.innerHTML = stopIcon;
        isGenerating = true;
    } else {
        sendBtn.innerHTML = sendIcon;
        isGenerating = false;
    }
}

// Send Message
async function sendMessage() {
    const text = userInput.value.trim();

    // Handle Interrupt
    if (isGenerating) {
        if (currentController) {
            currentController.abort();
            currentController = null;
        }
        setButtonState('idle');
        // If user just wanted to stop (no text), return
        if (!text) return;
    }

    if (!text) return;

    // Start New Generation
    currentController = new AbortController();
    setButtonState('generating');

    // Add User Message
    addMessage(text, 'user');
    userInput.value = '';

    // Add AI Placeholder
    const aiBubble = addMessage('...', 'ai');
    aiBubble.textContent = ''; // Clear placeholder

    // Call API
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, history: history }),
            signal: currentController.signal
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            fullResponse += chunk;
            aiBubble.textContent = fullResponse;
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        // Update History
        history.push({ role: "user", content: text });
        history.push({ role: "assistant", content: fullResponse });

    } catch (error) {
        if (error.name === 'AbortError') {
            aiBubble.textContent += " [Interrupted]";
            history.push({ role: "user", content: text });
            // Don't save partial AI response to history to avoid confusion, or save it?
            // Let's save it so context is maintained.
            history.push({ role: "assistant", content: aiBubble.textContent });
        } else {
            aiBubble.textContent = "Error: Could not connect to Bhondu.";
            console.error(error);
        }
    } finally {
        setButtonState('idle');
        currentController = null;
    }
}

function addMessage(text, role) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;

    msgDiv.appendChild(bubble);
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    return bubble;
}

// Event Listeners
sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});
