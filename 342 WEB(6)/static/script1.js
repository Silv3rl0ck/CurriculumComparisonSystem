document.addEventListener('DOMContentLoaded', function() {
    const compareBtn = document.getElementById('compareBtn-file');
    const inputText = document.getElementById('inputText');
    const categorySelect = document.getElementById('category');
    const resultsTable = document.querySelector('#resultsTable tbody');

    // Levenshtein Distance Calculator
    function levenshteinDistance(a, b) {
        const matrix = [];
        let i, j;

        // Initialize matrix
        for (i = 0; i <= b.length; i++) {
            matrix[i] = [i];
        }
        for (j = 0; j <= a.length; j++) {
            matrix[0][j] = j;
        }

        // Fill matrix
        for (i = 1; i <= b.length; i++) {
            for (j = 1; j <= a.length; j++) {
                const cost = a[j-1] === b[i-1] ? 0 : 1;
                matrix[i][j] = Math.min(
                    matrix[i-1][j] + 1,    // Deletion
                    matrix[i][j-1] + 1,    // Insertion
                    matrix[i-1][j-1] + cost // Substitution
                );
            }
        }

        return matrix[b.length][a.length];
    }

    // Calculate similarity percentage (0-100%)
    function calculateSimilarity(text1, text2) {
        const maxLength = Math.max(text1.length, text2.length);
        if (maxLength === 0) return 100; // Both strings are empty
        
        const distance = levenshteinDistance(text1, text2);
        return ((maxLength - distance) / maxLength) * 100;
    }

    // Fetch courses from Flask backend and compare
    compareBtn.addEventListener('click', async function() {
        const input = inputText.value.trim();
        const selectedCategory = categorySelect.value;

        if (!input) {
            alert('กรุณาใส่ข้อความเพื่อเปรียบเทียบ');
            return;
        }

        try {
            const response = await fetch('/api/compare-with-database', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: input,
                    category: selectedCategory
                })
            });
            
            const data = await response.json();
            
            // Clear previous results
            resultsTable.innerHTML = '';
            
            // Display top 5 results
            data.results.forEach((course, index) => {
                const row = document.createElement('tr');
                
                // Calculate Levenshtein similarity with description
                const similarity = calculateSimilarity(input, course.description);
                
                row.innerHTML = `
                    <td>${index + 1}</td>
                    <td>${course.id}</td>
                    <td>${course.name}</td>
                    <td>${course.description}</td>
                    <td class="similarity-score">${similarity.toFixed(2)}%</td>
                `;
                
                resultsTable.appendChild(row);
            });

        } catch (error) {
            console.error('Error:', error);
            alert('เกิดข้อผิดพลาดในการเปรียบเทียบข้อความ');
        }
    });
});