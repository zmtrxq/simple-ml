document.addEventListener('DOMContentLoaded', () => { 
    // UI element references
    const trainFileInput = document.getElementById('train-file-input');
    const testFileInput = document.getElementById('test-file-input');
    const trainFileNameSpan = document.getElementById('train-file-name');
    const testFileNameSpan = document.getElementById('test-file-name');
    const runModelsButton = document.getElementById('run-models');
    const resultsArea = document.getElementById('results-area');
    const generalPlotsArea = document.getElementById('general-plots-area');
    const summaryTableArea = document.getElementById('summary-table-area'); 
    const loadingIndicator = document.getElementById('loading-indicator');
    const generalDivider = document.querySelector('hr.divider.results-divider');
    const summaryDivider = document.querySelector('hr.divider.summary-divider'); 
    const dfHeadPreviewContainer = document.getElementById('df-head-preview');
    const columnSelectionStep = document.getElementById('column-selection-step');
    const columnSelectorContainer = document.getElementById('column-selector-container');
    const problemTypeStep = document.getElementById('problem-type-step');

    let globalResultsData = null; // Holds backend response for download functionality

    function updateUIState(step) {
        columnSelectionStep.style.display = (step >= 1) ? 'block' : 'none';
        problemTypeStep.style.display = (step >= 2) ? 'block' : 'none';
        runModelsButton.style.display = (step >= 2) ? 'inline-block' : 'none';
    }
    updateUIState(0);

    trainFileInput.addEventListener('change', async () => {
        if (trainFileInput.files.length > 0) {
            trainFileNameSpan.textContent = trainFileInput.files[0].name;
            await fetchAndDisplayColumns(trainFileInput.files[0]); 
            updateUIState(2); 
        } else {
            trainFileNameSpan.textContent = '';
            columnSelectorContainer.innerHTML = ''; 
            dfHeadPreviewContainer.innerHTML = '';
            updateUIState(0); 
        }
    });

    // Sends training CSV to backend and loads preview + column names
    async function fetchAndDisplayColumns(trainFile) {
        const formData = new FormData();
        formData.append('train_file', trainFile);
        loadingIndicator.style.display = 'block';
        columnSelectorContainer.innerHTML = '<p>Loading columns...</p>';
        dfHeadPreviewContainer.innerHTML = '<p>Loading data preview...</p>';

        try {
            const response = await fetch('/get_columns', { method: 'POST', body: formData });
            const data = await response.json();

            if (response.ok) {
                if (data.columns) populateColumnSelectors(data.columns);
                else columnSelectorContainer.innerHTML = `<p class="warning-text">Error: No columns data.</p>`;

                if (data.df_head_html) dfHeadPreviewContainer.innerHTML = data.df_head_html;
                else dfHeadPreviewContainer.innerHTML = `<p class="warning-text">Error: No data preview.</p>`;

                if (data.error) {
                    columnSelectorContainer.innerHTML = `<p class="warning-text">Error: ${data.error}</p>`;
                    dfHeadPreviewContainer.innerHTML = '';
                }

            } else {
                const errorMsg = data.error || 'Could not load column data.';
                columnSelectorContainer.innerHTML = `<p class="warning-text">Error: ${errorMsg}</p>`;
                dfHeadPreviewContainer.innerHTML = '';
            }

        } catch (error) {
            columnSelectorContainer.innerHTML = `<p class="warning-text">Network Error getting columns.</p>`;
            dfHeadPreviewContainer.innerHTML = '';
        } finally {
            loadingIndicator.style.display = 'none';
        }
    }

    // Renders UI for target and drop column selections
    function populateColumnSelectors(columns) {
        columnSelectorContainer.innerHTML = ''; 
        if (columns.length === 0) { 
            columnSelectorContainer.innerHTML = '<p>No columns found.</p>'; 
            return; 
        }

        const instruction = document.createElement('p');
        instruction.textContent = 'Select Target (one) and Columns to Drop (multiple):';
        columnSelectorContainer.appendChild(instruction);

        const table = document.createElement('table');
        table.classList.add('column-select-table', 'neon-table');

        const thead = table.createTHead(); 
        const headerRow = thead.insertRow();
        ['Column Name', 'Target', 'Drop'].forEach(text => {
            const th = document.createElement('th'); 
            th.textContent = text; 
            headerRow.appendChild(th);
        });

        const tbody = table.createTBody();

        columns.forEach((colName, index) => {
            const row = tbody.insertRow();

            const cellName = row.insertCell(); 
            cellName.textContent = colName; 
            cellName.classList.add('col-name-cell');

            const cellTarget = row.insertCell(); 
            cellTarget.classList.add('select-cell');
            const targetRadio = document.createElement('input'); 
            targetRadio.type = 'radio';
            targetRadio.name = 'target_column_radio'; 
            targetRadio.value = colName; 
            targetRadio.id = `target_col_${index}`;
            if ((index === columns.length - 1 && columns.length > 1) || columns.length === 1) targetRadio.checked = true;
            cellTarget.addEventListener('click', () => targetRadio.click()); 
            cellTarget.appendChild(targetRadio);

            const cellDrop = row.insertCell(); 
            cellDrop.classList.add('select-cell');
            const dropCheckbox = document.createElement('input'); 
            dropCheckbox.type = 'checkbox';
            dropCheckbox.value = colName; 
            dropCheckbox.id = `drop_col_${index}`; 
            dropCheckbox.classList.add('drop-column-checkbox');
            cellDrop.addEventListener('click', (e) => { 
                if (e.target !== dropCheckbox) dropCheckbox.checked = !dropCheckbox.checked; 
            });
            cellDrop.appendChild(dropCheckbox);
        });

        columnSelectorContainer.appendChild(table);
    }

    testFileInput.addEventListener('change', () => {
        testFileNameSpan.textContent = testFileInput.files.length > 0 ? testFileInput.files[0].name : '';
    });

    // Main model run trigger - sends all selections to backend
    runModelsButton.addEventListener('click', async () => {
        const trainFile = trainFileInput.files[0]; 
        const testFile = testFileInput.files[0];
        const selectedTargetRadio = document.querySelector('input[name="target_column_radio"]:checked');

        if (!selectedTargetRadio) { alert('Please select a target column.'); return; }

        const targetColumn = selectedTargetRadio.value;
        const columnsToDrop = Array.from(document.querySelectorAll('.drop-column-checkbox:checked')).map(cb => cb.value);

        if (columnsToDrop.includes(targetColumn)) { alert('Target column cannot also be dropped.'); return; }

        const problemType = document.querySelector('input[name="problem-type"]:checked').value;

        if (!trainFile || !testFile) { alert('Please select both training and testing CSV files.'); return; }

        const formData = new FormData();
        formData.append('train_file', trainFile); 
        formData.append('test_file', testFile);
        formData.append('target_column', targetColumn); 
        formData.append('columns_to_drop', JSON.stringify(columnsToDrop));
        formData.append('problem_type', problemType);

        resultsArea.innerHTML = ''; 
        generalPlotsArea.innerHTML = ''; 
        summaryTableArea.innerHTML = '';
        if (generalDivider) generalDivider.style.display = 'none';
        if (summaryDivider) summaryDivider.style.display = 'none';

        loadingIndicator.style.display = 'block'; 
        runModelsButton.disabled = true;
        globalResultsData = null;

        try {
            const response = await fetch('/process', { method: 'POST', body: formData });
            const data = await response.json();

            if (response.ok) {
                if (data.error) {
                    resultsArea.innerHTML = `<p class="warning-text">Error: ${data.error}</p>`; 
                } else {
                    globalResultsData = data;
                    displaySummaryTable(data.summary_metrics, data.problem_type);
                    displayGeneralPlots(data.correlation_plot); 
                    displayModelResults(data.results, data.problem_type, data.target_column_name);
                    if (data.summary_metrics?.length > 0 && summaryDivider) summaryDivider.style.display = 'block';
                    if ((data.correlation_plot || data.results?.length > 0) && generalDivider) generalDivider.style.display = 'block';
                }
            } else {
                let errorMsg = `Server responded with status ${response.status}.`;
                if (data?.error) errorMsg += ` Message: ${data.error}`;
                resultsArea.innerHTML = `<p class="warning-text">Error: ${errorMsg}</p>`;
            }
        } catch (error) {
            resultsArea.innerHTML = `<p class="warning-text">Network Error processing data. (${error.message})</p>`;
        } finally {
            loadingIndicator.style.display = 'none'; 
            runModelsButton.disabled = false;
        }
    });

    // Renders summary table with model metrics
    function displaySummaryTable(summaryData, problemType) {
        if (!summaryData || summaryData.length === 0) {
            summaryTableArea.style.display = 'none'; 
            return;
        }

        summaryTableArea.style.display = 'block';
        summaryTableArea.innerHTML = '<h2>Model Summary (Cross-Validation Performance)</h2>'; 

        let tableHtml = '<table class="summary-table neon-table"><thead><tr>'; 
        const headers = Object.keys(summaryData[0]);
        headers.forEach(header => { tableHtml += `<th>${header}</th>`; });
        tableHtml += '</tr></thead><tbody>';

        summaryData.forEach(row => {
            tableHtml += '<tr>';
            headers.forEach(header => {
                let cellValue = row[header]; 
                let cellClass = '';
                if (header.toLowerCase() === 'accuracy' && problemType === 'classification') { 
                    cellClass = 'metric-accuracy'; 
                    cellValue = (parseFloat(cellValue) * 100).toFixed(2) + '%';
                } else if (typeof cellValue === 'number') { 
                    cellValue = cellValue.toFixed(4); 
                }
                tableHtml += `<td class="${cellClass}">${cellValue}</td>`;
            });
            tableHtml += '</tr>';
        });

        tableHtml += '</tbody></table>';
        summaryTableArea.innerHTML += tableHtml; 
    }

    // Renders correlation matrix
    function displayGeneralPlots(correlationPlotB64) {
        let html = '';
        if (correlationPlotB64) {
            html += `<h3>Overall Correlation Matrix</h3><img src="${correlationPlotB64}" alt="Correlation Matrix">`;
        }
        generalPlotsArea.innerHTML = html;
        generalPlotsArea.style.display = correlationPlotB64 ? 'block' : 'none';
    }

    // Displays per-model results and attaches download buttons
    function displayModelResults(results, problemType, targetColumnName) {
        if (!results || results.length === 0) { 
            resultsArea.innerHTML = '<p>No detailed model results to display.</p>'; 
            resultsArea.style.display = 'none';
            return; 
        }

        resultsArea.style.display = 'block'; 
        let html = '<h2>Detailed Model Results</h2>';

        results.forEach((result, index) => {
            html += `<div class="model-result-card">
                        <h3>
                            ${result.model}
                            <button class="neon-button download-preds-button" data-model-index="${index}">Download Predictions</button>
                        </h3>
                        <h4>Cross-Validation Metrics:</h4>
                        <table class="neon-table"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>`;
            for (const [metric, value] of Object.entries(result.metrics)) {
                html += `<tr><td>${metric}</td><td>${value}</td></tr>`;
            }
            html += `</tbody></table><div class="model-plots">`;

            if (result.feature_importance_plot) {
                html += `<h4>Feature Importances:</h4><img src="${result.feature_importance_plot}" alt="Feature Importance Plot for ${result.model}">`;
            }

            if (problemType === 'regression' && result.prediction_plot) { 
                html += `<h4>Prediction Plot (Training Data Behavior):</h4><img src="${result.prediction_plot}" alt="Prediction Plot for ${result.model}">`;
            }

            html += `</div></div><hr class="divider">`;
        });

        if (html.endsWith('<hr class="divider">')) { 
            html = html.substring(0, html.lastIndexOf('<hr class="divider">')); 
        }

        resultsArea.innerHTML = html;

        // Attach prediction CSV download events
        document.querySelectorAll('.download-preds-button').forEach(button => {
            button.addEventListener('click', (event) => {
                const modelIndex = parseInt(event.target.getAttribute('data-model-index'));
                if (globalResultsData?.results?.[modelIndex]) {
                    const modelResult = globalResultsData.results[modelIndex];
                    downloadPredictionsCSV(modelResult.test_set_predictions, modelResult.model, globalResultsData.target_column_name || 'target');
                }
            });
        });
    }

    // Downloads predictions CSV for given model
    function downloadPredictionsCSV(predictions, modelName, targetHeader) {
        if (!predictions || predictions.length === 0) {
            alert('No predictions available to download for this model.');
            return;
        }

        const header = targetHeader; 
        let csvContent = header + "\n"; 
        predictions.forEach(pred => {
            let formattedPred = String(pred);
            if (formattedPred.includes(",")) {
                formattedPred = `"${formattedPred.replace(/"/g, '""')}"`;
            }
            csvContent += formattedPred + "\n";
        });

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement("a");

        if (link.download !== undefined) {
            const url = URL.createObjectURL(blob);
            const filenameSafeModelName = modelName.replace(/[^a-z0-9_]/gi, '_').toLowerCase();
            link.setAttribute("href", url);
            link.setAttribute("download", `${filenameSafeModelName}_test_predictions.csv`);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        } else {
            alert("CSV download not supported by your browser.");
        }
    }
});
