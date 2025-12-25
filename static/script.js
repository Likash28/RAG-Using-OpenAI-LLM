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

// Supported file types - Audio, images, text, and PDF files
const SUPPORTED_AUDIO = ['mp3', 'wav', 'm4a', 'flac', 'ogg'];
const SUPPORTED_IMAGE = ['png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'];
const SUPPORTED_TEXT = ['txt', 'pdf']; // Plain text files and PDFs
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
            addMessage('assistant', `⚠️ File "${file.name}" (${ext}) is not supported. Only audio files (mp3, wav, m4a, flac, ogg), image files (png, jpg, jpeg, webp, bmp, tiff), and text files (txt, pdf) are allowed.`, 'warning');
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
            
            // Check if any images have BLIP captions/OCR text or audio files have transcripts
            console.log('Ingestion result:', result);
            console.log('Files in result:', result.files);
            
            // Categorize all files properly
            // Check both the flag AND the content to be more robust
            const imageFiles = (result.files || []).filter(f => f.is_image || f.blip_caption || f.ocr_text);
            const audioFiles = (result.files || []).filter(f => f.is_audio || f.audio_transcript);
            const pdfFiles = (result.files || []).filter(f => f.is_pdf || f.pdf_text);
            const textFiles = (result.files || []).filter(f => f.is_text || f.text_content);

            console.log('=== FILE CATEGORIZATION ===');
            console.log('All files received:', JSON.stringify(result.files, null, 2));
            console.log(`Found ${imageFiles.length} image file(s), ${audioFiles.length} audio file(s), ${pdfFiles.length} PDF file(s), ${textFiles.length} text file(s)`);
            
            // Collect all files that need automatic queries
            const filesToQuery = [];
            
            // Process image files - MANDATORY sentiment analysis for ALL images
            imageFiles.forEach((file) => {
                // Always display BLIP caption if available
                if (file.blip_caption) {
                    displayImageCaption(file.filename, file.blip_caption);
                }
                
                // Display OCR text if available
                if (file.ocr_text) {
                    displayOCRText(file.filename, file.ocr_text);
                    // Priority: Use OCR text for sentiment analysis
                    filesToQuery.push({
                        filename: file.filename,
                        query: file.ocr_text.trim(),
                        type: 'image_ocr',
                        displayName: `OCR text from ${file.filename}`
                    });
                } else if (file.blip_caption) {
                    // No OCR text, use BLIP caption for sentiment analysis
                    filesToQuery.push({
                        filename: file.filename,
                        query: file.blip_caption,
                        type: 'image_blip',
                        displayName: `BLIP caption from ${file.filename}`
                    });
                } else if (file.sentiment_fallback) {
                    // No OCR or BLIP, use fallback text for sentiment analysis (MANDATORY)
                    console.log(`⚠️ Using fallback text for image ${file.filename} - sentiment analysis will still be performed`);
                    filesToQuery.push({
                        filename: file.filename,
                        query: file.sentiment_fallback,
                        type: 'image_fallback',
                        displayName: `Image content from ${file.filename}`
                    });
                } else {
                    // Last resort: use filename for sentiment analysis
                    console.log(`⚠️ No text available for image ${file.filename}, using filename for sentiment analysis`);
                    filesToQuery.push({
                        filename: file.filename,
                        query: `Image file: ${file.filename}. Visual content uploaded.`,
                        type: 'image_fallback',
                        displayName: `Image content from ${file.filename}`
                    });
                }
            });
            
            // Process audio files - MANDATORY sentiment analysis for ALL audio files
            audioFiles.forEach((file) => {
                console.log(`Processing audio file: ${file.filename}, transcript length: ${file.audio_transcript ? file.audio_transcript.length : 0}`);
                
                if (file.audio_transcript && file.audio_transcript.trim().length > 0) {
                    displayAudioTranscript(file.filename, file.audio_transcript);
                    
                    filesToQuery.push({
                        filename: file.filename,
                        query: file.audio_transcript.trim(),
                        type: 'audio',
                        displayName: `Audio transcript from ${file.filename}`
                    });
                } else if (file.sentiment_fallback) {
                    // No transcript, use fallback text for sentiment analysis (MANDATORY)
                    console.warn(`⚠️ No transcript found for ${file.filename}, using fallback text for sentiment analysis`);
                    filesToQuery.push({
                        filename: file.filename,
                        query: file.sentiment_fallback,
                        type: 'audio_fallback',
                        displayName: `Audio content from ${file.filename}`
                    });
                } else {
                    // Last resort: use filename for sentiment analysis
                    console.warn(`⚠️ No transcript or fallback for ${file.filename}, using filename for sentiment analysis`);
                    filesToQuery.push({
                        filename: file.filename,
                        query: `Audio file: ${file.filename}. Audio content uploaded.`,
                        type: 'audio_fallback',
                        displayName: `Audio content from ${file.filename}`
                    });
                }
            });
            
            // Process PDF files - MANDATORY sentiment analysis for ALL PDFs
            pdfFiles.forEach((file) => {
                console.log(`Processing PDF file: ${file.filename}, text length: ${file.pdf_text ? file.pdf_text.length : 0}`);

                if (file.pdf_text && file.pdf_text.trim().length > 0) {
                    // Display PDF text (similar to audio transcript)
                    displayPDFText(file.filename, file.pdf_text);

                    // Send PDF text for sentiment analysis (similar to audio transcripts)
                    filesToQuery.push({
                        filename: file.filename,
                        query: file.pdf_text.trim(),
                        type: 'pdf',
                        displayName: `PDF text from ${file.filename}`
                    });
                } else if (file.sentiment_fallback) {
                    // No text extracted, use fallback text for sentiment analysis (MANDATORY)
                    console.warn(`⚠️ No text found for PDF ${file.filename}, using fallback text for sentiment analysis`);
                    filesToQuery.push({
                        filename: file.filename,
                        query: file.sentiment_fallback,
                        type: 'pdf_fallback',
                        displayName: `PDF content from ${file.filename}`
                    });
                } else {
                    // Last resort: use filename for sentiment analysis
                    console.warn(`⚠️ No text or fallback for PDF ${file.filename}, using filename for sentiment analysis`);
                    filesToQuery.push({
                        filename: file.filename,
                        query: `PDF document: ${file.filename}. Document uploaded.`,
                        type: 'pdf_fallback',
                        displayName: `PDF content from ${file.filename}`
                    });
                }
            });

            // Process text files - MANDATORY sentiment analysis for ALL text files
            textFiles.forEach((file) => {
                console.log(`Processing text file: ${file.filename}, content length: ${file.text_content ? file.text_content.length : 0}`);

                if (file.text_content && file.text_content.trim().length > 0) {
                    // Display text content (similar to PDF text and audio transcript)
                    displayTextContent(file.filename, file.text_content);

                    // Send text content for sentiment analysis
                    filesToQuery.push({
                        filename: file.filename,
                        query: file.text_content.trim(),
                        type: 'text',
                        displayName: `Text content from ${file.filename}`
                    });
                } else if (file.sentiment_fallback) {
                    // No content extracted, use fallback text for sentiment analysis (MANDATORY)
                    console.warn(`⚠️ No content found for text file ${file.filename}, using fallback text for sentiment analysis`);
                    filesToQuery.push({
                        filename: file.filename,
                        query: file.sentiment_fallback,
                        type: 'text_fallback',
                        displayName: `Text content from ${file.filename}`
                    });
                } else {
                    // Last resort: use filename for sentiment analysis
                    console.warn(`⚠️ No content or fallback for text file ${file.filename}, using filename for sentiment analysis`);
                    filesToQuery.push({
                        filename: file.filename,
                        query: `Text file: ${file.filename}. Text document uploaded.`,
                        type: 'text_fallback',
                        displayName: `Text content from ${file.filename}`
                    });
                }
            });
            
            // Show success message
            console.log('=== FILES TO QUERY ===');
            console.log(`Total files to query: ${filesToQuery.length}`);
            filesToQuery.forEach((fq, i) => {
                console.log(`  ${i+1}. ${fq.filename} (${fq.type}): ${fq.query.substring(0, 50)}...`);
            });

            if (filesToQuery.length > 0 || imageFiles.length > 0 || audioFiles.length > 0 || pdfFiles.length > 0 || textFiles.length > 0) {
                addMessage('assistant', `✅ Successfully processed ${result.ingested.length} file(s)! Scheduling ${filesToQuery.length} automatic sentiment analysis queries...`);

                // Process all queries sequentially with minimal delays for faster response
                // Reduced delay for better user experience while still avoiding API rate limits
                let queryDelay = 500; // Start with 0.5 second delay (reduced from 1s)
                const delayIncrement = 1000; // 1 second between each query (reduced from 2s)
                
                filesToQuery.forEach((fileQuery, index) => {
                    const delay = queryDelay + (index * delayIncrement);
                    console.log(`⏱️ Scheduling query ${index + 1}/${filesToQuery.length} for ${fileQuery.filename} in ${delay}ms`);

                    setTimeout(() => {
                        console.log(`🚀 Auto-querying LLM with ${fileQuery.type} from ${fileQuery.filename}:`, fileQuery.query.substring(0, 100) + (fileQuery.query.length > 100 ? '...' : ''));
                        console.log(`📝 Full ${fileQuery.type} length: ${fileQuery.query.length} characters`);
                        console.log(`📊 Current isProcessing state: ${isProcessing}`);

                        // Show analysis message for the file being processed
                        addMessage('user', `[Analyzing ${fileQuery.displayName}]`);

                        // Query LLM with the full text for sentiment analysis
                        console.log(`📤 Calling queryLLM for ${fileQuery.filename}...`);
                        queryLLM(fileQuery.query, false).then(() => {
                            console.log(`✅ Successfully processed automatic query for ${fileQuery.filename} (${fileQuery.type})`);
                        }).catch(err => {
                            console.error(`❌ Error in automatic query for ${fileQuery.filename} (${fileQuery.type}):`, err);
                            addMessage('assistant', `Error processing ${fileQuery.type} from ${fileQuery.filename}: ${err.message}`);
                        });
                    }, delay);
                });
                
                // If no automatic queries but files were processed, show message
                if (filesToQuery.length === 0 && (imageFiles.length > 0 || textFiles.length > 0)) {
                    addMessage('assistant', 'You can now ask questions about the uploaded documents.');
                }
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
    
    // Show full transcript with scrollable container for long transcripts
    const isLongTranscript = transcript.length > 500;
    const displayText = isLongTranscript 
        ? `<div style="max-height: 300px; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 5px;">${transcript}</div>`
        : `<p style="white-space: pre-wrap; word-wrap: break-word;">${transcript}</p>`;
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <i class="fas fa-microphone"></i>
            <div class="caption-header">
                <h4>Audio Transcript Generated</h4>
                <span class="caption-filename">${filename}</span>
                <span style="font-size: 0.8em; color: #888; margin-left: 10px;">(${transcript.length} characters)</span>
            </div>
            <div class="caption-content">
                <p><strong>Full Transcript:</strong></p>
                ${displayText}
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
                <h4>Image Caption Generated (BLIP)</h4>
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

function displayOCRText(filename, ocrText) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant ocr-text';
    
    const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    // Truncate OCR text for display (show first 300 chars)
    const preview = ocrText.length > 300 ? ocrText.substring(0, 300) + '...' : ocrText;
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <i class="fas fa-file-alt"></i>
            <div class="caption-header">
                <h4>OCR Text Extracted</h4>
                <span class="caption-filename">${filename}</span>
            </div>
            <div class="caption-content">
                <p><strong>Extracted Text (${ocrText.length} chars):</strong></p>
                <pre style="white-space: pre-wrap; word-wrap: break-word; max-height: 200px; overflow-y: auto;">${preview}</pre>
                ${ocrText.length > 300 ? '<p><em>... (text truncated for display, full text will be used for sentiment analysis)</em></p>' : ''}
            </div>
            <div class="message-time">${time}</div>
        </div>
    `;
    
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
}

function displayPDFText(filename, pdfText) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant pdf-text';

    const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});

    // Show full PDF text with scrollable container for long text
    const isLongText = pdfText.length > 500;
    const displayText = isLongText
        ? `<div style="max-height: 300px; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 5px;">${pdfText}</div>`
        : `<p style="white-space: pre-wrap; word-wrap: break-word;">${pdfText}</p>`;

    messageDiv.innerHTML = `
        <div class="message-content">
            <i class="fas fa-file-pdf"></i>
            <div class="caption-header">
                <h4>PDF Text Extracted</h4>
                <span class="caption-filename">${filename}</span>
                <span style="font-size: 0.8em; color: #888; margin-left: 10px;">(${pdfText.length} characters)</span>
            </div>
            <div class="caption-content">
                <p><strong>Full Text (extracted using unstructured):</strong></p>
                ${displayText}
                <p style="margin-top: 10px; font-size: 0.9em; color: #888;"><em>This text will be sent for sentiment analysis...</em></p>
            </div>
            <div class="message-time">${time}</div>
        </div>
    `;

    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
}

function displayTextContent(filename, textContent) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant text-content';

    const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});

    // Show full text content with scrollable container for long text
    const isLongText = textContent.length > 500;
    const displayText = isLongText
        ? `<div style="max-height: 300px; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 5px;">${textContent}</div>`
        : `<p style="white-space: pre-wrap; word-wrap: break-word;">${textContent}</p>`;

    messageDiv.innerHTML = `
        <div class="message-content">
            <i class="fas fa-file-alt"></i>
            <div class="caption-header">
                <h4>Text File Content Extracted</h4>
                <span class="caption-filename">${filename}</span>
                <span style="font-size: 0.8em; color: #888; margin-left: 10px;">(${textContent.length} characters)</span>
            </div>
            <div class="caption-content">
                <p><strong>Full Text Content:</strong></p>
                ${displayText}
                <p style="margin-top: 10px; font-size: 0.9em; color: #888;"><em>This text will be sent for sentiment analysis...</em></p>
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
        
        // For automatic queries (OCR text, audio transcripts), skip vector DB retrieval
        // This makes sentiment analysis much faster - we already have the text!
        const isAutomaticQuery = !showUserMessage;  // Automatic queries don't show user message
        const skipRetrieval = isAutomaticQuery;  // Skip retrieval for automatic sentiment analysis
        
        const response = await fetch(`${API_BASE_URL}/api/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                k: 5,
                skip_retrieval: skipRetrieval  // Skip vector DB retrieval for faster sentiment analysis
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
            console.log(`📊 Checking sentiment_analysis:`, {
                exists: !!result.sentiment_analysis,
                length: result.sentiment_analysis ? result.sentiment_analysis.length : 0,
                preview: result.sentiment_analysis ? result.sentiment_analysis.substring(0, 100) : 'N/A'
            });
            if (result.sentiment_analysis && result.sentiment_analysis.trim()) {
                console.log(`📊 Displaying sentiment analysis (${result.sentiment_analysis.length} chars)`);
                addSentimentAnalysis(result.sentiment_analysis);
            } else {
                console.warn('⚠️ No sentiment_analysis in result or empty');
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


// Convert markdown-style text to clean HTML for better display
function formatSentimentContent(text) {
    // Remove ** markers and convert to styled sections
    let html = text
        // Convert **Header:** patterns to styled headers
        .replace(/\*\*([^*]+):\*\*/g, '<div class="sentiment-section-header">$1</div>')
        // Convert remaining **text** to bold
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        // Convert - bullet points to styled list items
        .replace(/^- (.+)$/gm, '<div class="sentiment-bullet">$1</div>')
        // Convert line breaks to proper spacing
        .replace(/\n\n/g, '</div><div class="sentiment-paragraph">')
        .replace(/\n/g, '<br>');

    // Wrap in paragraph div
    html = '<div class="sentiment-paragraph">' + html + '</div>';

    return html;
}

function addSentimentAnalysis(sentimentContent) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant sentiment-analysis';

    const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});

    // Format the sentiment content for better display
    const formattedContent = formatSentimentContent(sentimentContent);

    messageDiv.innerHTML = `
        <div class="message-content">
            <i class="fas fa-chart-line"></i>
            <div class="sentiment-header">
                <h4>Sentiment Analysis</h4>
            </div>
            <div class="sentiment-content">
                ${formattedContent}
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
