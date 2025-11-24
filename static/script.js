// API Configuration - Auto-detect based on current host
const API_BASE_URL = window.location.origin;

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const selectedFiles = document.getElementById('selectedFiles');
const uploadedFiles = document.getElementById('uploadedFiles');
const uploadActions = document.getElementById('uploadActions');
const submitButton = document.getElementById('submitButton');
const clearButton = document.getElementById('clearButton');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const resetButton = document.getElementById('resetButton');
const messages = document.getElementById('messages');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const loadingOverlay = document.getElementById('loadingOverlay');
const charCount = document.getElementById('charCount');

// State
let selectedFileList = []; // Files selected but not yet uploaded
let uploadedFileList = []; // Files that have been uploaded
let isProcessing = false;

// Supported file types - STRICT: Only audio, images, and txt files
const SUPPORTED_AUDIO = ['mp3', 'wav', 'm4a', 'flac', 'ogg'];
const SUPPORTED_IMAGE = ['png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'];
const SUPPORTED_TEXT = ['txt']; // Only plain text files - NO PDF, DOC, DOCX
const ALL_SUPPORTED = [...SUPPORTED_AUDIO, ...SUPPORTED_IMAGE, ...SUPPORTED_TEXT];

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    checkServerStatus();
    setupEventListeners();
    autoResizeTextarea();
});

// Event Listeners
function setupEventListeners() {
    // File upload - single click fix
    // The file input is positioned absolutely over the upload area (opacity 0, covers 100%)
    // It will handle clicks directly, so we don't need the upload area click handler
    // Just stop propagation to prevent any double-triggering
    fileInput.addEventListener('click', (e) => {
        e.stopPropagation();
    });
    
    // Drag and drop handlers
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // File selection handler
    fileInput.addEventListener('change', handleFileSelect);
    
    // Action buttons
    submitButton.addEventListener('click', submitFiles);
    clearButton.addEventListener('click', clearSelectedFiles);

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
    addFilesToSelection(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    addFilesToSelection(files);
    // Reset input to allow selecting same file again
    e.target.value = '';
}

function isValidFileType(filename) {
    const ext = filename.split('.').pop()?.toLowerCase();
    return ext && ALL_SUPPORTED.includes(ext);
}

function getFileType(filename) {
    const ext = filename.split('.').pop()?.toLowerCase();
    if (SUPPORTED_AUDIO.includes(ext)) return 'audio';
    if (SUPPORTED_IMAGE.includes(ext)) return 'image';
    if (SUPPORTED_TEXT.includes(ext)) return 'text';
    return 'unknown';
}

function addFilesToSelection(files) {
    if (files.length === 0) return;
    
    // Filter only supported files - STRICT validation
    const validFiles = files.filter(file => {
        if (!isValidFileType(file.name)) {
            const ext = file.name.split('.').pop()?.toLowerCase() || 'unknown';
            addMessage('assistant', `⚠️ File "${file.name}" (${ext}) is not supported. Only audio files (mp3, wav, m4a, flac, ogg), image files (png, jpg, jpeg, webp, bmp, tiff), and text files (txt) are allowed.`, 'warning');
            return false;
        }
        // Check for duplicates
        if (selectedFileList.some(f => f.name === file.name && f.size === file.size)) {
            return false;
        }
        return true;
    });
    
    if (validFiles.length === 0) return;
    
    // Add to selected files list
    selectedFileList.push(...validFiles);
    updateSelectedFilesDisplay();
    
    // Show submit button
    uploadActions.style.display = 'flex';
    
    addMessage('assistant', `Added ${validFiles.length} file(s) to selection. Click "Submit & Process Files" to upload.`);
}

function updateSelectedFilesDisplay() {
    selectedFiles.innerHTML = '';
    selectedFiles.classList.add('show');
    
    if (selectedFileList.length === 0) {
        selectedFiles.classList.remove('show');
        uploadActions.style.display = 'none';
        return;
    }
    
    selectedFileList.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item selected';
        const fileType = getFileType(file.name);
        const iconMap = {
            'audio': 'fa-music',
            'image': 'fa-image',
            'text': 'fa-file-text',
            'unknown': 'fa-file'
        };
        
        fileItem.innerHTML = `
            <div>
                <i class="fas ${iconMap[fileType] || 'fa-file'}"></i>
                <span>${file.name}</span>
                <small>(${(file.size / 1024).toFixed(1)} KB)</small>
            </div>
            <button class="remove-file-btn" data-index="${index}">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        // Add remove button handler
        const removeBtn = fileItem.querySelector('.remove-file-btn');
        removeBtn.addEventListener('click', () => {
            selectedFileList.splice(index, 1);
            updateSelectedFilesDisplay();
        });
        
        selectedFiles.appendChild(fileItem);
    });
}

function clearSelectedFiles() {
    selectedFileList = [];
    updateSelectedFilesDisplay();
    addMessage('assistant', 'Cleared all selected files.');
}

async function submitFiles() {
    if (selectedFileList.length === 0) {
        addMessage('assistant', 'No files selected to upload.');
        return;
    }
    
    if (isProcessing) {
        addMessage('assistant', 'Please wait, files are being processed...');
        return;
    }
    
    showLoading();
    submitButton.disabled = true;
    
    try {
        const formData = new FormData();
        selectedFileList.forEach(file => {
            formData.append('files', file);
        });

        addMessage('assistant', `Uploading and processing ${selectedFileList.length} file(s)...`);

        const response = await fetch(`${API_BASE_URL}/api/ingest`, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            
            // Move files from selected to uploaded
            uploadedFileList.push(...selectedFileList);
            selectedFileList = [];
            updateSelectedFilesDisplay();
            
            // Display uploaded files
            displayUploadedFiles(result.files || result.ingested);
            
            // IMPORTANT: Hide loading BEFORE scheduling automatic queries
            // This ensures isProcessing is false when queries execute
            hideLoading();
            
            // Check if any images have BLIP captions or audio files have transcripts
            console.log('Ingestion result:', result);
            console.log('Files in result:', result.files);
            const imageFiles = (result.files || []).filter(f => f.is_image && f.blip_caption);
            const audioFiles = (result.files || []).filter(f => f.is_audio && f.audio_transcript);
            console.log(`Found ${imageFiles.length} image file(s) with captions, ${audioFiles.length} audio file(s) with transcripts`);
            
            if (imageFiles.length > 0 || audioFiles.length > 0) {
                // Show success message
                addMessage('assistant', `✅ Successfully processed ${result.ingested.length} file(s)!`);
                
                let queryIndex = 0;
                
                // Display BLIP captions for each image and automatically query LLM
                imageFiles.forEach((file) => {
                    displayImageCaption(file.filename, file.blip_caption);
                    
                    // Automatically query LLM with the BLIP caption text after a short delay
                    setTimeout(() => {
                        const query = file.blip_caption;
                        addMessage('user', query);
                        queryLLM(query, false);
                    }, 1000 + (queryIndex++ * 500));
                });
                
                // Display audio transcripts for each audio file and automatically query LLM
                audioFiles.forEach((file) => {
                    console.log(`Processing audio file: ${file.filename}, transcript length: ${file.audio_transcript ? file.audio_transcript.length : 0}`);
                    
                    if (!file.audio_transcript || file.audio_transcript.trim().length === 0) {
                        console.warn(`⚠️ No transcript found for ${file.filename}`);
                        return;
                    }
                    
                    displayAudioTranscript(file.filename, file.audio_transcript);
                    
                    // Automatically query LLM with the FULL audio transcript text after a short delay
                    setTimeout(() => {
                        const query = file.audio_transcript.trim();
                        console.log(`🚀 Auto-querying LLM with audio transcript from ${file.filename}:`, query.substring(0, 100) + (query.length > 100 ? '...' : ''));
                        console.log(`📝 Full transcript length: ${query.length} characters`);
                        
                        // Add user message showing the query
                        addMessage('user', query);
                        
                        // Query LLM with the full transcript
                        queryLLM(query, false).then(() => {
                            console.log(`✅ Successfully processed automatic query for ${file.filename}`);
                        }).catch(err => {
                            console.error(`❌ Error in automatic audio query for ${file.filename}:`, err);
                            addMessage('assistant', `Error processing audio transcript: ${err.message}`);
                        });
                    }, 1000 + (queryIndex++ * 500));
                });
            } else {
                addMessage('assistant', `✅ Successfully processed ${result.ingested.length} file(s)! You can now ask questions about the uploaded documents.`);
            }
        } else {
            throw new Error('Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        addMessage('assistant', 'Sorry, there was an error uploading your files. Please try again.');
        hideLoading();
    } finally {
        submitButton.disabled = false;
    }
}

function displayUploadedFiles(files) {
    uploadedFiles.innerHTML = '';
    uploadedFiles.classList.add('show');
    
    files.forEach(file => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        const filename = file.filename || file; // Support both formats
        const isImage = file.is_image || false;
        
        fileItem.innerHTML = `
            <div>
                <i class="fas ${isImage ? 'fa-image' : 'fa-file'}"></i>
                <span>${filename}</span>
            </div>
            <i class="fas fa-check-circle" style="color: #4ecdc4;"></i>
        `;
        uploadedFiles.appendChild(fileItem);
    });
}

function displayAudioTranscript(filename, transcript) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant audio-transcript';
    
    const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    // Truncate transcript for display (show first 200 chars)
    const preview = transcript.length > 200 ? transcript.substring(0, 200) + '...' : transcript;
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <i class="fas fa-microphone"></i>
            <div class="caption-header">
                <h4>Audio Transcript Generated</h4>
                <span class="caption-filename">${filename}</span>
            </div>
            <div class="caption-content">
                <p><strong>Transcript Preview:</strong> ${preview}</p>
            </div>
            <div class="message-time">${time}</div>
        </div>
    `;
    
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
}

function displayImageCaption(filename, caption) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant image-caption';
    
    const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <i class="fas fa-image"></i>
            <div class="caption-header">
                <h4>Image Caption Generated</h4>
                <span class="caption-filename">${filename}</span>
            </div>
            <div class="caption-content">
                <p><strong>BLIP Description:</strong> ${caption}</p>
            </div>
            <div class="message-time">${time}</div>
        </div>
    `;
    
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
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

// Shared function to query LLM (used by both manual and automatic queries)
async function queryLLM(query, showUserMessage = true) {
    // Allow automatic queries (showUserMessage=false) to proceed even if isProcessing is true
    // This is because they're triggered by the system after ingestion completes
    if (isProcessing && showUserMessage) return;

    // Add user message to chat if requested
    if (showUserMessage) {
        addMessage('user', query);
    }
    
    if (showUserMessage) {
        messageInput.value = '';
        charCount.textContent = '0/500';
        messageInput.style.height = 'auto';
        sendButton.disabled = true;
    }

    // Only show loading for manual queries (to avoid double loading indicators)
    if (showUserMessage) {
        showLoading();
    }

    try {
        const queryPreview = query.length > 100 ? query.substring(0, 100) + '...' : query;
        console.log(`📤 Calling /api/ask with query (${query.length} chars):`, queryPreview);
        console.log(`📤 Query type: ${showUserMessage ? 'manual' : 'automatic'}`);
        
        const response = await fetch(`${API_BASE_URL}/api/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                k: 5
            })
        });
        
        console.log(`📥 Response status: ${response.status} ${response.statusText}`);

        if (response.ok) {
            const result = await response.json();
            console.log(`✅ Received response from /api/ask:`, {
                hasMainResponse: !!result.main_response,
                mainResponseLength: result.main_response ? result.main_response.length : 0,
                hasSentiment: !!result.sentiment_analysis
            });
            
            // Display main response
            if (result.main_response) {
                console.log(`📝 Displaying main response (${result.main_response.length} chars)`);
                addMessage('assistant', result.main_response);
            } else {
                console.warn('⚠️ No main_response in result');
                addMessage('assistant', 'Received response but no content available.');
            }
            
            // Display sentiment analysis if available
            if (result.sentiment_analysis && result.sentiment_analysis.trim()) {
                console.log(`📊 Displaying sentiment analysis`);
                addSentimentAnalysis(result.sentiment_analysis);
            }
            
            // Sources display removed - not showing in UI
        } else {
            // Try to get error message from response
            let errorMessage = 'Request failed';
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorData.message || errorMessage;
            } catch (e) {
                errorMessage = `Request failed with status ${response.status}`;
            }
            throw new Error(errorMessage);
        }
    } catch (error) {
        console.error('API error:', error);
        const errorMsg = error.message || 'Sorry, I encountered an error processing your request. Please try again.';
        addMessage('assistant', `Error: ${errorMsg}`);
    } finally {
        // Only hide loading if we showed it (manual queries)
        if (showUserMessage) {
            hideLoading();
            sendButton.disabled = false;
        }
    }
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isProcessing) return;

    // Use the shared queryLLM function
    await queryLLM(message, true);
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
                    selectedFiles.innerHTML = '';
                    selectedFiles.classList.remove('show');
                    uploadActions.style.display = 'none';
                    selectedFileList = [];
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
