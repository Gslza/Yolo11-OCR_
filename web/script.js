// YOLO11 + EasyOCR Beverage Detection — Dashboard Client Logic

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const connBadge = document.getElementById('connection-badge');
  const connText = document.getElementById('connection-status-text');
  
  const statTotal = document.getElementById('stat-total');
  const statSafe = document.getElementById('stat-safe');
  const statWarning = document.getElementById('stat-warning');
  const statDanger = document.getElementById('stat-danger');

  const videoStream = document.getElementById('video-stream');
  const videoPlaceholder = document.getElementById('video-placeholder');
  const videoFps = document.getElementById('video-fps');

  const detStatus = document.getElementById('detection-status');
  const detStatusText = document.getElementById('detection-status-text');
  const productName = document.getElementById('product-name');
  const sugarValue = document.getElementById('sugar-value');
  const sugarGaugeFill = document.getElementById('sugar-gauge-fill');
  const ocrTextBox = document.getElementById('ocr-text-box');
  const screenshotContainer = document.getElementById('screenshot-container');
  const detectionScreenshot = document.getElementById('detection-screenshot');

  const infoYoloConf = document.getElementById('info-yolo-conf');
  const infoOcrAngle = document.getElementById('info-ocr-angle');
  const infoMatchScore = document.getElementById('info-match-score');
  const infoMatchType = document.getElementById('info-match-type');

  const historyTable = document.getElementById('history-table');
  const historyTbody = document.getElementById('history-tbody');
  const historyEmpty = document.getElementById('history-empty');

  let eventSource = null;
  let reconnectTimeout = null;

  // Initialize Video Stream Source
  // Point to the Flask /video_feed route
  videoStream.src = '/video_feed';
  videoStream.onload = () => {
    videoStream.style.display = 'block';
    videoPlaceholder.style.display = 'none';
  };
  videoStream.onerror = () => {
    videoStream.style.display = 'none';
    videoPlaceholder.style.display = 'flex';
  };

  // Fetch initial stats and history
  fetchStats();
  fetchHistory();

  // Connect to SSE
  connectSSE();

  function connectSSE() {
    if (eventSource) {
      eventSource.close();
    }

    eventSource = new EventSource('/events');

    eventSource.onopen = () => {
      console.log('SSE connection established.');
      updateConnectionState(true);
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
      }
    };

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleStreamUpdate(data);
      } catch (err) {
        console.error('Error parsing SSE event data:', err);
      }
    };

    eventSource.onerror = (err) => {
      console.error('SSE connection error:', err);
      updateConnectionState(false);
      eventSource.close();
      
      // Auto reconnect after 3 seconds
      if (!reconnectTimeout) {
        reconnectTimeout = setTimeout(() => {
          connectSSE();
        }, 3000);
      }
    };
  }

  function updateConnectionState(isConnected) {
    if (isConnected) {
      connBadge.className = 'connection-badge connected';
      connText.textContent = 'Connected';
    } else {
      connBadge.className = 'connection-badge disconnected';
      connText.textContent = 'Disconnected';
    }
  }

  function handleStreamUpdate(data) {
    // 1. Update FPS
    if (data.fps !== undefined) {
      videoFps.textContent = `FPS: ${parseFloat(data.fps).toFixed(1)}`;
    }

    // 2. Update Active Detection Panel
    updateDetectionPanel(data.active_detection);

    // 3. Update Stats Cards (if included, or fetch from API)
    if (data.stats) {
      updateStats(data.stats);
    }

    // 4. Prepend to history table if this was a newly logged detection
    if (data.new_log_event) {
      addHistoryRow(data.new_log_event);
      // Fetch latest stats to keep them accurate
      fetchStats();
    }
  }

  function updateDetectionPanel(detection) {
    if (!detection || detection.state === 'idle') {
      // Idle state
      detStatus.className = 'detection-status idle';
      detStatusText.textContent = 'Idle';
      productName.className = 'product-name unknown';
      productName.textContent = 'Menunggu Deteksi...';
      
      sugarValue.textContent = '-';
      sugarValue.className = 'sugar-gauge-value';
      sugarGaugeFill.style.width = '0%';
      sugarGaugeFill.className = 'sugar-gauge-fill';

      ocrTextBox.textContent = '-';
      infoYoloConf.textContent = '-';
      infoOcrAngle.textContent = '-';
      infoMatchScore.textContent = '-';
      infoMatchType.textContent = '-';
      if (screenshotContainer) screenshotContainer.style.display = 'none';
      if (detectionScreenshot) detectionScreenshot.src = '';
      return;
    }

    // Scanning state (bottle detected but not yet matched/recognized)
    if (detection.state === 'scanning') {
      detStatus.className = 'detection-status scanning';
      detStatusText.textContent = 'Memindai Label (OCR)...';
      productName.className = 'product-name unknown';
      productName.textContent = 'Mendeteksi Botol...';
      
      sugarValue.textContent = '-';
      sugarValue.className = 'sugar-gauge-value';
      sugarGaugeFill.style.width = '0%';
      sugarGaugeFill.className = 'sugar-gauge-fill';

      ocrTextBox.textContent = detection.ocr_text || '-';
      infoYoloConf.textContent = detection.yolo_confidence ? `${(detection.yolo_confidence * 100).toFixed(1)}%` : '-';
      infoOcrAngle.textContent = detection.ocr_angle !== undefined ? `${detection.ocr_angle}°` : '-';
      infoMatchScore.textContent = '-';
      infoMatchType.textContent = '-';
      if (screenshotContainer) screenshotContainer.style.display = 'none';
      if (detectionScreenshot) detectionScreenshot.src = '';
      return;
    }

    // Active detection state (product recognized/matched)
    const product = detection.product || {};
    const name = product.name || 'Tidak dikenali';
    const sugar = product.sugar_g;
    const status = (product.status || 'Tidak dikenali').toLowerCase().replace(' ', '-'); // "aman", "batas-wajar", "tidak-disarankan"

    // Update Status Banner
    detStatus.className = `detection-status ${status}`;
    detStatusText.textContent = getStatusBannerText(product.status || 'Tidak dikenali', sugar);

    // Update Product Name
    productName.textContent = name;
    if (name === 'Tidak dikenali') {
      productName.className = 'product-name unknown';
    } else {
      productName.className = 'product-name';
    }

    // Update Sugar Gauge
    if (typeof sugar === 'number') {
      sugarValue.textContent = `${sugar}g`;
      
      // Calculate gauge width (0 to 30g+ map to 0 to 100%)
      const percentage = Math.min(100, Math.max(0, (sugar / 30) * 100));
      sugarGaugeFill.style.width = `${percentage}%`;
      
      // Set status coloring for sugar text & fill
      let statusClass = 'safe';
      if (status === 'batas-wajar') statusClass = 'warning';
      if (status === 'tidak-disarankan') statusClass = 'danger';

      sugarValue.className = `sugar-gauge-value ${statusClass}`;
      sugarGaugeFill.className = `sugar-gauge-fill ${statusClass}`;
    } else {
      sugarValue.textContent = '-';
      sugarValue.className = 'sugar-gauge-value';
      sugarGaugeFill.style.width = '0%';
      sugarGaugeFill.className = 'sugar-gauge-fill';
    }

    // Update Details
    ocrTextBox.textContent = product.ocr_text || '-';
    infoYoloConf.textContent = detection.yolo_confidence ? `${(detection.yolo_confidence * 100).toFixed(1)}%` : '-';
    infoOcrAngle.textContent = product.ocr_angle !== undefined ? `${product.ocr_angle}°` : '-';
    infoMatchScore.textContent = product.match_score !== undefined ? `${(product.match_score * 100).toFixed(0)}%` : '-';
    
    // Format Match Type nicely
    let matchType = product.match_type || '-';
    if (matchType === 'exact') matchType = 'Exact Match';
    else if (matchType === 'fuzzy') matchType = 'Fuzzy Match';
    else if (matchType === 'none') matchType = 'No Match';
    infoMatchType.textContent = matchType;

    // Update Screenshot
    const screenshotUrl = product.screenshot_filename
      ? `/screenshot/${product.screenshot_filename}`
      : (product.screenshot_path || '');

    if (screenshotUrl && screenshotContainer && detectionScreenshot) {
      detectionScreenshot.src = screenshotUrl;
      screenshotContainer.style.display = 'block';
    } else if (screenshotContainer) {
      screenshotContainer.style.display = 'none';
    }
  }

  function getStatusBannerText(status, sugar) {
    if (status === 'Aman') return 'Kadar Gula Aman';
    if (status === 'Batas Wajar') return 'Kadar Gula Batas Wajar';
    if (status === 'Tidak Disarankan') return 'Kadar Gula Tidak Disarankan!';
    return status;
  }

  function updateStats(stats) {
    statTotal.textContent = stats.total || 0;
    statSafe.textContent = stats.safe || 0;
    statWarning.textContent = stats.warning || 0;
    statDanger.textContent = stats.danger || 0;
  }

  function addHistoryRow(log, isPrepend = true) {
    const tr = document.createElement('tr');
    tr.className = 'new-row';

    // Format timestamp
    let timeStr = '-';
    if (log.timestamp) {
      try {
        const date = new Date(log.timestamp);
        timeStr = date.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      } catch (e) {
        timeStr = log.timestamp;
      }
    }

    // Status badge formatting
    const statusText = log.status || 'Tidak dikenali';
    const statusClass = statusText.toLowerCase().replace(' ', '-');
    const statusBadge = `<span class="status-badge ${statusClass}">${statusText}</span>`;

    // YOLO confidence formatting
    const yoloConf = log.yolo_confidence ? `${(parseFloat(log.yolo_confidence) * 100).toFixed(0)}%` : '-';

    // Match Score formatting
    const matchScore = log.match_score ? `${(parseFloat(log.match_score) * 100).toFixed(0)}%` : '-';

    // Match type formatting
    let matchType = log.match_type || '-';
    if (matchType === 'exact') matchType = 'Exact';
    else if (matchType === 'fuzzy') matchType = 'Fuzzy';

    tr.innerHTML = `
      <td class="mono">${timeStr}</td>
      <td><strong>${log.product_name || 'Tidak dikenali'}</strong></td>
      <td class="mono">${log.sugar_g !== undefined && log.sugar_g !== '-' ? log.sugar_g + 'g' : '-'}</td>
      <td>${statusBadge}</td>
      <td class="mono">${yoloConf}</td>
      <td class="mono">${matchScore}</td>
      <td>${matchType}</td>
    `;

    if (isPrepend) {
      historyTbody.insertBefore(tr, historyTbody.firstChild);
    } else {
      historyTbody.appendChild(tr);
    }

    // Keep max 20 rows in DOM
    while (historyTbody.children.length > 20) {
      historyTbody.removeChild(historyTbody.lastChild);
    }

    // Show table, hide empty placeholder
    historyTable.style.display = 'table';
    historyEmpty.style.display = 'none';

    // Remove the flash class after animation is done
    setTimeout(() => {
      tr.classList.remove('new-row');
    }, 1000);
  }

  function fetchStats() {
    fetch('/api/stats')
      .then(res => res.json())
      .then(stats => updateStats(stats))
      .catch(err => console.error('Error fetching stats:', err));
  }

  function fetchHistory() {
    fetch('/api/history')
      .then(res => res.json())
      .then(historyList => {
        historyTbody.innerHTML = '';
        if (historyList && historyList.length > 0) {
          historyList.forEach(log => {
            addHistoryRow(log, false); // Append since we're loading chronologically
          });
        } else {
          historyTable.style.display = 'none';
          historyEmpty.style.display = 'block';
        }
      })
      .catch(err => console.error('Error fetching history:', err));
  }
});
