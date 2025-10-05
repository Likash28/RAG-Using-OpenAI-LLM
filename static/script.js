// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadedFiles = document.getElementById('uploadedFiles');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const resetButton = document.getElementById('resetButton');
const messages = document.getElementById('messages');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const loadingOverlay = document.getElementById('loadingOverlay');
const charCount = document.getElementById('charCount');

// State
let uploadedFileList = [];
let isProcessing = false;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    checkServerStatus();
    setupEventListeners();
    autoResizeTextarea();
});

// Event Listeners
function setupEventListeners() {
    // File upload
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);

    // Message input
    messageInput.addEventListener('input', handleMessageInput);
    messageInput.addEventListener('keydown', handleKeyDown);
    sendButton.addEventListener('click', sendMessage);
    resetButton.addEventListener('click', resetConversation);
}

// Server Status Check
async function checkServerStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        if (response.ok) {
            updateStatus('connected', 'Connected');
        } else {
            updateStatus('error', 'Server Error');
        }
    } catch (error) {
        updateStatus('error', 'Disconnected');
        console.error('Server connection failed:', error);
    }
}

function updateStatus(type, text) {
    statusDot.className = `status-dot ${type}`;
    statusText.textContent = text;
}

// File Upload Handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files);
    processFiles(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    processFiles(files);
}

async function processFiles(files) {
    if (files.length === 0) return;

    showLoading();
    
    try {
        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
        });

        const response = await fetch(`${API_BASE_URL}/api/ingest`, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            displayUploadedFiles(result.ingested);
            addMessage('assistant', `Successfully uploaded ${result.ingested.length} file(s)!`);
        } else {
            throw new Error('Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        addMessage('assistant', 'Sorry, there was an error uploading your files. Please try again.');
    } finally {
        hideLoading();
    }
}

function displayUploadedFiles(files) {
    uploadedFiles.innerHTML = '';
    uploadedFiles.classList.add('show');
    
    files.forEach(file => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div>
                <i class="fas fa-file"></i>
                <span>${file}</span>
            </div>
            <i class="fas fa-check-circle" style="color: #4ecdc4;"></i>
        `;
        uploadedFiles.appendChild(fileItem);
    });
}

// Message Handling
function handleMessageInput() {
    const text = messageInput.value;
    charCount.textContent = `${text.length}/500`;
    
    // Auto-resize textarea
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
    
    // Enable/disable send button
    sendButton.disabled = text.trim().length === 0 || isProcessing;
}

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isProcessing) return;

    // Add user message to chat
    addMessage('user', message);
    messageInput.value = '';
    charCount.textContent = '0/500';
    messageInput.style.height = 'auto';
    sendButton.disabled = true;

    showLoading();

    try {
        const response = await fetch(`${API_BASE_URL}/api/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: message,
                k: 5
            })
        });

        if (response.ok) {
            const result = await response.json();
            
            // Display main response
            addMessage('assistant', result.main_response);
            
            // Display sentiment analysis if available
            if (result.sentiment_analysis && result.sentiment_analysis.trim()) {
                addSentimentAnalysis(result.sentiment_analysis);
            }
            
            // Show context sources if available
            if (result.contexts && result.contexts.length > 0) {
                const sources = result.contexts.map(ctx => ctx.metadata?.source || 'Unknown').filter(Boolean);
                if (sources.length > 0) {
                    addMessage('assistant', `Sources: ${sources.join(', ')}`);
                }
            }
        } else {
            throw new Error('Request failed');
        }
    } catch (error) {
        console.error('API error:', error);
        addMessage('assistant', 'Sorry, I encountered an error processing your request. Please try again.');
    } finally {
        hideLoading();
        sendButton.disabled = false;
    }
}


function addSentimentAnalysis(sentimentContent) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant sentiment-analysis';
    
    const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <i class="fas fa-chart-line"></i>
            <div class="sentiment-header">
                <h4>Sentiment Analysis</h4>
            </div>
            <div class="sentiment-content">
                <pre>${sentimentContent}</pre>
            </div>
            <div class="message-time">${time}</div>
        </div>
    `;
    
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
}

function addMessage(type, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    messageDiv.innerHTML = `
        <div class="message-content">
            ${type === 'assistant' ? '<i class="fas fa-robot"></i>' : ''}
            <p>${content}</p>
            <div class="message-time">${time}</div>
        </div>
    `;
    
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
}

function resetConversation() {
    if (isProcessing) return;
    
    if (confirm('Are you sure you want to reset the conversation and clear all uploaded files?')) {
        showLoading();
        
        fetch(`${API_BASE_URL}/api/reset`, { method: 'POST' })
            .then(response => {
                if (response.ok) {
                    messages.innerHTML = `
                        <div class="message welcome">
                            <div class="message-content">
                                <i class="fas fa-robot"></i>
                                <p>Hello! I'm your depression research assistant. Upload some documents and ask me questions about depression, mental health, or any related topics.</p>
                            </div>
                        </div>
                    `;
                    uploadedFiles.innerHTML = '';
                    uploadedFiles.classList.remove('show');
                    uploadedFileList = [];
                    addMessage('assistant', 'Conversation reset successfully!');
                } else {
                    throw new Error('Reset failed');
                }
            })
            .catch(error => {
                console.error('Reset error:', error);
                addMessage('assistant', 'Sorry, there was an error resetting the conversation.');
            })
            .finally(() => {
                hideLoading();
            });
    }
}

// Utility Functions
function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = messageInput.scrollHeight + 'px';
}

function showLoading() {
    isProcessing = true;
    loadingOverlay.classList.add('show');
    sendButton.disabled = true;
}

function hideLoading() {
    isProcessing = false;
    loadingOverlay.classList.remove('show');
    sendButton.disabled = messageInput.value.trim().length === 0;
}

// Auto-check server status every 30 seconds
setInterval(checkServerStatus, 30000);
