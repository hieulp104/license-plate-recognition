const uploadScreen = document.getElementById('uploadScreen');
const resultsScreen = document.getElementById('resultsScreen');
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const processBtn = document.getElementById('processBtn');
const resetBtn = document.getElementById('resetBtn');
const backBtn = document.getElementById('backBtn');
const saveBtn = document.getElementById('saveBtn');
const btnText = document.getElementById('btnText');
const spinner = document.getElementById('spinner');
const errorMessage = document.getElementById('errorMessage');
const statusMessage = document.getElementById('statusMessage');

const resultImage = document.getElementById('resultImage');
const plateResultsContainer = document.getElementById('plateResultsContainer');

let selectedFile = null;
let currentSessionId = null;

uploadZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => handleFileSelect(e.target.files[0]));

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadZone.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
});

['dragenter', 'dragover'].forEach(eventName => {
    uploadZone.addEventListener(eventName, () => {
        uploadZone.style.borderColor = 'rgba(255, 255, 255, 0.8)';
        uploadZone.style.background = 'rgba(255, 255, 255, 0.25)';
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    uploadZone.addEventListener(eventName, () => {
        uploadZone.style.borderColor = 'rgba(255, 255, 255, 0.4)';
        uploadZone.style.background = 'rgba(255, 255, 255, 0.15)';
    }, false);
});

uploadZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});


processBtn.addEventListener('click', processImage);
resetBtn.addEventListener('click', resetForm);
backBtn.addEventListener('click', resetForm);
saveBtn.addEventListener('click', saveResults);

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleFileSelect(file) {
    if (!file) return;

    const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!allowedTypes.includes(file.type)) {
        showError('Loại file không hợp lệ. Vui lòng chọn ảnh JPG hoặc PNG.');
        return;
    }
    if (file.size > 16 * 1024 * 1024) {
        showError('Kích thước file quá lớn (tối đa 16MB).');
        return;
    }

    selectedFile = file;
    fileName.innerHTML = `✅ ${file.name} <br><small style="opacity: 0.8;">(${(file.size / 1024 / 1024).toFixed(2)} MB)</small>`;
    processBtn.disabled = false;
    hideError();
}

async function processImage() {
    if (!selectedFile) return;

    setLoading(true);
    hideError();
    hideStatus();

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch('/process-image', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Lỗi không xác định từ server.');
        }

        displayResults(data);

    } catch (error) {
        console.error('Lỗi khi xử lý ảnh:', error);
        showError(error.message);
    } finally {
        setLoading(false);
    }
}

function displayResults(data) {
    currentSessionId = data.session_id;
    resultImage.src = data.result_image_base64;
    plateResultsContainer.innerHTML = ''; 

    if (data.plates && data.plates.length > 0) {
        data.plates.forEach(plate => {
            const plateElement = document.createElement('div');
            plateElement.className = 'plate-item';
            plateElement.innerHTML = `
                <div class="plate-details">
                    <div class="plate-text">${plate.text}</div>
                    <div class="confidence">Phát hiện: <span>${(plate.confidence * 100).toFixed(1)}%</span></div>
                </div>
                <div class="plate-images-container">
                    <div class="plate-image-section">
                        <div class="plate-image-title">📸 Ảnh Biển Số</div>
                        <img src="${plate.cropped_plate_base64}" class="plate-image" alt="Cropped Plate">
                    </div>
                    <div class="plate-image-section">
                        <div class="plate-image-title">⚫ Ảnh Nhị Phân</div>
                        <img src="${plate.binary_plate_base64 || ''}" class="plate-image" alt="Binary Plate">
                    </div>
                </div>
            `;
            plateResultsContainer.appendChild(plateElement);
        });
        saveBtn.style.display = 'block';
    } else {
        plateResultsContainer.innerHTML = '<div class="plate-text-not-found">Không phát hiện được biển số.</div>';
        saveBtn.style.display = 'none';
    }
    
    uploadScreen.style.display = 'none';
    resultsScreen.style.display = 'grid';
}

async function saveResults() {
    if (!currentSessionId) {
        showStatus('Lỗi: Không tìm thấy session ID để lưu.', 'error');
        return;
    }

    saveBtn.disabled = true;
    saveBtn.textContent = 'Đang lưu...';

    try {
        const response = await fetch('/save-results', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSessionId }),
        });
        const data = await response.json();

        if (response.ok && data.status === 'success') {
            showStatus(data.message, 'success');
        } else {
            showStatus(data.message || 'Lỗi không xác định khi lưu.', 'error');
        }
    } catch (error) {
        console.error('Lỗi khi lưu kết quả:', error);
        showStatus('Lỗi kết nối khi cố gắng lưu kết quả.', 'error');
    } finally {
        saveBtn.disabled = false;
        saveBtn.textContent = '💾 Lưu kết quả';
    }
}

function resetForm() {
    selectedFile = null;
    currentSessionId = null;
    fileInput.value = '';
    fileName.textContent = 'Chưa chọn file nào';
    processBtn.disabled = true;
    
    uploadScreen.style.display = 'flex';
    resultsScreen.style.display = 'none';
    
    hideError();
    hideStatus();
    saveBtn.style.display = 'none';
    saveBtn.disabled = false;
    saveBtn.textContent = '💾 Lưu kết quả';
}

function setLoading(isLoading) {
    processBtn.disabled = isLoading;
    spinner.style.display = isLoading ? 'inline-block' : 'none';
    btnText.style.display = isLoading ? 'none' : 'inline';
}

function showError(message) {
    errorMessage.textContent = `⚠️ ${message}`;
    errorMessage.style.display = 'block';
}

function hideError() {
    errorMessage.style.display = 'none';
}

function showStatus(message, type = 'success') {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type}`; 
    statusMessage.style.display = 'block';
}

function hideStatus() {
    statusMessage.style.display = 'none';
}