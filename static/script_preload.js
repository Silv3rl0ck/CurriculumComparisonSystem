document.addEventListener('DOMContentLoaded', () => {
  const btn        = document.getElementById('preloadBtn');
  const inp        = document.getElementById('preloadUpload');
  const status     = document.getElementById('preloadStatus');
  const tableBody  = document.querySelector('#preloadedFilesTable tbody');

  // Fetch and render the table of preloaded files
  async function refreshTable() {
    try {
      const res = await fetch('/api/preloaded-files');
      const json = await res.json();
      tableBody.innerHTML = '';
      json.files.forEach(fn => {
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.textContent = fn;
        tr.appendChild(td);
        tableBody.appendChild(tr);
      });
    } catch (err) {
      console.error('Error loading preloaded files:', err);
    }
  }

  // Trigger file picker
  btn.addEventListener('click', () => inp.click());

  // Handle file selection & upload
  inp.addEventListener('change', async () => {
    const file = inp.files[0];
    if (!file || !file.name.toLowerCase().endsWith('.xlsx')) {
      status.textContent = 'โปรดเลือกไฟล์ .xlsx เท่านั้น';
      status.style.color = 'red';
      return;
    }

    status.textContent = 'กำลังอัปโหลด...';
    status.style.color = 'blue';

    const fd = new FormData();
    fd.append('file', file);

    try {
      const res = await fetch('/api/upload-preloaded', {
        method: 'POST',
        body: fd
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Upload failed');

      status.textContent = `Uploaded: ${json.filename}`;
      status.style.color = 'green';
      await refreshTable();  // update the table
    } catch (err) {
      status.textContent = `Error: ${err.message}`;
      status.style.color = 'red';
    }
  });

  // Initial table load
  refreshTable();
});
