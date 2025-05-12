document.addEventListener('DOMContentLoaded', function() {
    // ── populate dropdown from backend
  const sel = document.getElementById('preloadedFiles');
  if (sel) {
    fetch('/api/preloaded-files')
      .then(r => r.json())
      .then(data => {
        sel.innerHTML = '<option value="">-- เลือกไฟล์ --</option>';
        data.files.forEach(fn => {
          const o = document.createElement('option');
          o.value = fn;
          o.textContent = fn;
          sel.appendChild(o);
        });
      })
      .catch(e => console.error('Could not load preloaded files:', e));
  }
    
    const compareBtn     = document.getElementById('compareBtn');
    const uploadBtn      = document.getElementById('uploadBtn');
    const loadCourseBtn  = document.getElementById('loadCourseBtn');
    const dbUpload       = document.getElementById('dbUpload');
    const dbStatus       = document.getElementById('dbStatus');
    const inputText      = document.getElementById('inputText');
    const thaiCourseName = document.getElementById('thaiCourseName');
    const preloadedFiles = document.getElementById('preloadedFiles');
    const courseDescription = document.getElementById('courseDescription');
    const resultsTable      = document.querySelector('#resultsTable tbody');
    
    // Create progress elements dynamically
    const progressContainer = document.createElement('div');
    progressContainer.className = 'progress-container';
    progressContainer.innerHTML = `
        <div id="progressBar" class="progress-bar"></div>
        <span id="progressText"></span>
    `;
    compareBtn.parentNode.insertBefore(progressContainer, compareBtn.nextSibling);

    // Handle database upload
    uploadBtn.addEventListener('click', () => dbUpload.click());
    
    dbUpload.addEventListener('change', async function(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        if (!file.name.endsWith('.xlsx')) {
            dbStatus.textContent = 'Only .xlsx files allowed';
            dbStatus.style.color = '#e74c3c';
            return;
        }
    
        dbStatus.textContent = 'Uploading...';
        dbStatus.style.color = '#3498db';
    
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/api/upload-database', {
                method: 'POST',
                body: formData
            });
            
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                const text = await response.text();
                throw new Error(`Server returned unexpected response: ${text.slice(0, 100)}`);
            }
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || 'Upload failed');
            }
            
            if (result.success) {
                dbStatus.textContent = `Uploaded: ${file.name} (${result.count} courses)`;
                dbStatus.style.color = '#27ae60';
            } else {
                throw new Error(result.error || 'Upload failed');
            }
        } catch (error) {
            console.error('Upload error:', error);
            dbStatus.textContent = `Error: ${error.message}`;
            dbStatus.style.color = '#e74c3c';
            alert(`Upload failed: ${error.message}`);
        }
    });

    // Load course description when button is clicked
    loadCourseBtn.addEventListener('click', async function() {
        const courseName = thaiCourseName.value.trim();
        const selectedFile = preloadedFiles.value;
        
        if (!courseName) {
            alert('กรุณาป้อนชื่อวิชาภาษาไทย');
            return;
        }
        
        if (!selectedFile) {
            alert('กรุณาเลือกไฟล์หลักสูตร');
            return;
        }
        
        try {
            const response = await fetch('/api/get-course-description', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    courseName: courseName,
                    fileId: selectedFile
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to load course');
            }
            
            const result = await response.json();
            
            if (result.success) {
                courseDescription.value = result.description || 'ไม่พบคำอธิบายรายวิชา';
                // Auto-fill the additional text area if description is found
                if (result.description) {
                    inputText.value = `${courseName} ${result.description}`;
                }
            } else {
                throw new Error(result.error || 'Course not found');
            }
        } catch (error) {
            console.error('Error loading course:', error);
            alert(`เกิดข้อผิดพลาด: ${error.message}`);
            courseDescription.value = 'ไม่พบคำอธิบายรายวิชา';
        }
    });

    compareBtn.addEventListener('click', async function() {
        const input = inputText.value.trim();
        const description = courseDescription.value.trim();
        if (!input && !description) {
            alert('กรุณาใส่ข้อความหรือโหลดคำอธิบายรายวิชาเพื่อเปรียบเทียบ');
            return;
        }

        compareBtn.disabled = true;
        compareBtn.textContent = 'กำลังประมวลผล...';

        try {
            const response = await fetch('/api/compare-with-database', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: input, description: description })
            });
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Server error');
            }

            const { results, total_courses } = await response.json();

            // simulate progress bar…
            let progress = 0;
            const interval = setInterval(() => {
                progress = Math.min(progress + 5, 100);
                updateProgress(progress, total_courses);
                if (progress >= 100) clearInterval(interval);
            }, 100);

            setTimeout(() => {
                displayResults(results);

                // ── update stats ─────────────────────────────────────────
                const total  = parseInt(localStorage.getItem('totalComparisons') || '0', 10) + 1;
                const semSim = parseInt(localStorage.getItem('semanticSimCount') || '0', 10) + 1;
                localStorage.setItem('totalComparisons', total);
                localStorage.setItem('semanticSimCount', semSim);
            }, 500);

        } catch (error) {
            console.error('Error:', error);
            alert(`เกิดข้อผิดพลาด: ${error.message}`);
        } finally {
            compareBtn.disabled = false;
            compareBtn.textContent = 'Compare';
        }
    });

    function displayResults(results) {
        resultsTable.innerHTML = '';
        if (!results || !results.length) {
            resultsTable.innerHTML = '<tr><td colspan="5" style="text-align:center;">ไม่พบรายวิชาที่ตรงกัน</td></tr>';
            document.getElementById('topCourseResult').textContent = '-';
            return;
        }
        results.forEach((course, i) => {
            const row = document.createElement('tr');
            let cls = 'low-similarity';
            if (course.percentage >= 70) cls = 'high-similarity';
            else if (course.percentage >= 40) cls = 'medium-similarity';

            row.innerHTML = `
            <td>${i+1}</td>
            <td>${course.id}</td>
            <td>${course.name}</td>
            <td>${course.description}</td>
            <td class="similarity-score ${cls}">${course.percentage.toFixed(2)}%</td>
            `;
            resultsTable.appendChild(row);
        });

        // top recommendation
        const top = results[0];
        document.getElementById('topCourseResult').innerHTML = `
          <strong>${top.id} - ${top.name}</strong><br>
          <span>ความเหมือน: ${top.percentage.toFixed(2)}%</span><br>
          <small>${top.description}</small>
        `;
    }

    function updateProgress(progress, total) {
        const bar  = document.getElementById('progressBar');
        const text = document.getElementById('progressText');
        bar.style.width = `${progress}%`;
        text.textContent = `Processing: ${progress}% (${total} courses)`;
    }
});
