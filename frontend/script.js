document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('arxiv-form');
    const arxivUrlInput = document.getElementById('arxiv-url');
    const loader = document.getElementById('loader');
    const errorMessage = document.getElementById('error-message');

    form.addEventListener('submit', async (event) => {
        event.preventDefault();

        const arxivUrl = arxivUrlInput.value.trim();
        if (!arxivUrl) {
            showError('Please enter an ArXiv URL or ID.');
            return;
        }

        showLoader();
        hideError();

        try {
            const apiUrl = `${FRONTEND_CONFIG.API_BASE_URL}/analyze?arxiv_url=${encodeURIComponent(arxivUrl)}`;
            const response = await fetch(apiUrl, {
                method: 'POST',
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'An unknown error occurred.');
            }

            const blob = await response.blob();
            const contentDisposition = response.headers.get('content-disposition');
            let filename = 'analysis.zip'; // Default filename
            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename="?(.+)"?/i);
                if (filenameMatch.length > 1) {
                    filename = filenameMatch[1];
                }
            }

            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

        } catch (error) {
            showError(error.message);
        } finally {
            hideLoader();
        }
    });

    function showLoader() {
        loader.style.display = 'flex';
        document.getElementById('arxiv-form').classList.add('hidden');
    }

    function hideLoader() {
        loader.style.display = 'none';
        document.getElementById('arxiv-form').classList.remove('hidden');
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('hidden');
    }

    function hideError() {
        errorMessage.classList.add('hidden');
    }
});
