document.addEventListener('DOMContentLoaded', function() {
    const compareBtn = document.getElementById('compareBtn-file');
    const inputText = document.getElementById('inputText');
    const categorySelect = document.getElementById('category');
    const resultsTable = document.querySelector('#resultsTable tbody');

    // Sample database of texts to compare against (minimum 5 items)
    const textDatabase = [
        { id: '601-11-01', name: 'ศาสตร์แห่งการเป็นพลเมืองที่พึงประสงค์', content: 'รัฐ สังคม และพลเมือง สิทธิเสรีภาพและหน้าที่พลเมืองไทย การดำรงตนในสังคมด้านคุณธรรมจริยธรรม การปกครองในระบอบประชาธิปไตย หลักธรรมาภิบาล ระบบอุปถัมภ์ การป้องกันการทุจริตคอร์รัปชั่น กรณีศึกษาด้านการทุจริตคอร์รัปชั่นในสังคมและบุคคลต้นแบบ',  category: 'มหาวิทยาลัยเทคโนโลยีราชมงคลสุวรรณภูมิ' },
        { id: '601-11-02', name: 'สังคมและสิ่งแวดล้อมเพื่อการพัฒนาที่ยั่งยืน', content: 'แนวคิดพื้นฐานเกี่ยวกับสังคมและสิ่งแวดล้อม ความสัมพันธ์กับการเปลี่ยนแปลงทางสังคมและสิ่งแวดล้อม สาเหตุและผลกระทบของปัญหาสังคมและสิ่งแวดล้อม แนวทางในการแก้ไขปัญหา การวิเคราะห์สถานการณ์ทางสังคมและสิ่งแวดล้อม',  category: 'มหาวิทยาลัยเทคโนโลยีราชมงคลสุวรรณภูมิ' },
        { id: '601-21-03', name: 'สังคมและเศรษฐกิจเพื่อฐานวิถีชีวิตใหม่ ', content: 'ศึกษาหลักการ แนวคิดการดำเนินชีวิตฐานวิถีชีวิตใหม่และการเป็นพลเมืองดิจิทัล ความสัมพันธ์ระหว่างสังคมและเศรษฐกิจกับฐานวิถีชีวิตใหม่',  category: 'มหาวิทยาลัยเทคโนโลยีราชมงคลสุวรรณภูมิ' },
        { id: '601-21-04', name: 'ศาสตร์พระราชาเพื่อการพัฒนาที่ยั่งยืน', content: 'การพัฒนาเศรษฐกิจและสังคมไทย หลักการพัฒนาสู่ความยั่งยืน หลักปรัชญาเศรษฐกิจพอเพียง ทฤษฎีใหม่ หลักการทรงงานและโครงการอันเนื่องมาจากพระราชดำริ ศาสตร์พระราชากับการพัฒนาสังคมไทย ',  category: 'มหาวิทยาลัยเทคโนโลยีราชมงคลสุวรรณภูมิ' },
        { id: '601-21-05', name: 'สังคมวิทยาเพื่อการพัฒนาทรัพยากรมนุษย์ในศตวรรษที่ 21', content: 'หลักการ แนวคิด ทฤษฎีทางสังคมศาสตร์ นวัตกรรมทางสังคม พลวัตทางสังคม', category: 'มหาวิทยาลัยเทคโนโลยีราชมงคลสุวรรณภูมิ' }
    ];

    compareBtn.addEventListener('click', function() {
        const input = inputText.value.trim();
        const selectedCategory = categorySelect.value;
        
        if (!input) {
            alert('Please enter some text to compare.');
            return;
        }

        // Clear previous results
        resultsTable.innerHTML = '';

        // Get all documents that match the selected category
        let matchingDocs = textDatabase.filter(doc => 
            selectedCategory === '' || doc.category === selectedCategory
        );

        // If we have fewer than 5 items, add some from other categories
        if (matchingDocs.length < 5) {
            const needed = 5 - matchingDocs.length;
            const otherDocs = textDatabase.filter(doc => 
                doc.category !== selectedCategory
            ).slice(0, needed);
            matchingDocs = matchingDocs.concat(otherDocs);
        }

        // Compare input against each document
        const results = matchingDocs.map(doc => {
            const similarity = calculateSimilarity(input, doc.content);
            return {
                ...doc,
                similarity: similarity
            };
        });

        // Sort by similarity score (descending)
        results.sort((a, b) => b.similarity - a.similarity);

        // Always show exactly 5 results
        const finalResults = results.slice(0, 5);

        // Display results in the table
        finalResults.forEach((result, index) => {
            const row = document.createElement('tr');
            
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${result.id}</td>
                <td>${result.name}</td>
                <td>${result.content}</td>
                <td class="similarity-score">${(result.similarity * 100).toFixed(2)}%</td>
            `;
            
            resultsTable.appendChild(row);
        });
    });

    // Simple similarity calculation using Jaccard 
    function calculateSimilarity(text1, text2) {
        // Tokenize texts into sets of words
        const tokenize = text => {
            return new Set(
                text.toLowerCase()
                    .replace(/[^\w\s]/g, '')
                    .split(/\s+/)
                    .filter(word => word.length > 0)
            );
        };

        const set1 = tokenize(text1);
        const set2 = tokenize(text2);

        // Calculate intersection and union
        const intersection = new Set([...set1].filter(x => set2.has(x)));
        const union = new Set([...set1, ...set2]);

        // Jaccard index: size of intersection divided by size of union
        return union.size === 0 ? 0 : intersection.size / union.size;
    }
});

document.addEventListener('DOMContentLoaded', function() {
    const compareBtn = document.getElementById('compareBtn');
    const input1 = document.getElementById('input1');
    const input2 = document.getElementById('input2');
    const algorithmSelect = document.getElementById('algorithm');
    const similarityScore = document.getElementById('similarityScore');

    compareBtn.addEventListener('click', function() {
        const text1 = input1.value.trim();
        const text2 = input2.value.trim();
        const algorithm = algorithmSelect.value;
        
        if (!text1 || !text2) {
            alert('Please enter text in both boxes.');
            return;
        }

        let score;
        switch(algorithm) {
            case 'jaccard':
                score = calculateJaccardSimilarity(text1, text2);
                break;
            case 'cosine':
                score = calculateCosineSimilarity(text1, text2);
                break;
            case 'levenshtein':
                score = calculateLevenshteinSimilarity(text1, text2);
                break;
            default:
                score = calculateLevenshteinSimilarity(text1, text2);
        }

        similarityScore.textContent = `${Math.round(score * 100)}%`;
    });

    // Jaccard Similarity
    function calculateJaccardSimilarity(text1, text2) {
        const tokenize = text => {
            return new Set(
                text.toLowerCase()
                    .replace(/[^\w\s]/g, '')
                    .split(/\s+/)
                    .filter(word => word.length > 0)
            );
        };

        const set1 = tokenize(text1);
        const set2 = tokenize(text2);

        const intersection = new Set([...set1].filter(x => set2.has(x)));
        const union = new Set([...set1, ...set2]);

        return union.size === 0 ? 0 : intersection.size / union.size;
    }

    // Cosine Similarity
    function calculateCosineSimilarity(text1, text2) {
        const tokenize = text => {
            return text.toLowerCase()
                .replace(/[^\w\s]/g, '')
                .split(/\s+/)
                .filter(word => word.length > 0);
        };

        const tokens1 = tokenize(text1);
        const tokens2 = tokenize(text2);
        
        const allTokens = [...new Set([...tokens1, ...tokens2])];
        const vec1 = allTokens.map(token => tokens1.filter(t => t === token).length);
        const vec2 = allTokens.map(token => tokens2.filter(t => t === token).length);
        
        const dotProduct = vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
        const magnitude1 = Math.sqrt(vec1.reduce((sum, val) => sum + val * val, 0));
        const magnitude2 = Math.sqrt(vec2.reduce((sum, val) => sum + val * val, 0));
        
        return magnitude1 * magnitude2 === 0 ? 0 : dotProduct / (magnitude1 * magnitude2);
    }

    // Levenshtein Similarity
    function calculateLevenshteinSimilarity(text1, text2) {
        function levenshteinDistance(s, t) {
            if (!s.length) return t.length;
            if (!t.length) return s.length;
            
            const arr = [];
            for (let i = 0; i <= t.length; i++) {
                arr[i] = [i];
                for (let j = 1; j <= s.length; j++) {
                    arr[i][j] = i === 0 ? j : Math.min(
                        arr[i - 1][j] + 1,
                        arr[i][j - 1] + 1,
                        arr[i - 1][j - 1] + (s[j - 1] === t[i - 1] ? 0 : 1)
                    );
                }
            }
            return arr[t.length][s.length];
        }
        
        const distance = levenshteinDistance(text1, text2);
        const maxLength = Math.max(text1.length, text2.length);
        return 1 - distance / maxLength;
    }
});

compareBtn.addEventListener('click', async function() {
    const input = inputText.value.trim();
    const selectedCategory = categorySelect.value;
    
    if (!input) {
        alert('Please enter some text to compare.');
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
        
        // Display results in the table
        data.results.forEach((result, index) => {
            const row = document.createElement('tr');
            
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${result.id}</td>
                <td>${result.name}</td>
                <td>${result.content}</td>
                <td class="similarity-score">${result.percentage}%</td>
            `;
            
            resultsTable.appendChild(row);
        });
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while comparing texts.');
    }
}); 