$(document).ready(function () {
    $('#uploadForm').on('submit', function (event) {
        const pdfType = $('[name="pdf_type"]:checked').val();
        let url = ""
        if (pdfType === 'normal') {
            url = '/upload_normal';
        } else {
            url = '/upload_academic';
        }
        $('#results').hide();
        $('#imageContainer').empty();
        $('#LoadingDiv').show();
        event.preventDefault();
        const formData = new FormData(this);

        // Upload PDF via Ajax
        $.ajax({
            url: url,
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                console.log(response)
                $('#LoadingDiv').hide();
                $('#results').show();
                $('#imageContainer').empty();
                // Display extracted images
                response.images.forEach(function (image) {
                    const filename = image.url.split('/').pop();
                    const caption = image.caption
                    $('#imageContainer').append(`
                        <div class="col-md-4">
                            <div class="card">
                                <img src="${image.url}" class="card-img-top" alt="${filename}">
                                <div class="card-body text-center">
                                    <p>${caption}</p>
                                    <a href="/download/${filename}" class="btn btn-success">Download</a>
                                </div>
                            </div>
                        </div>
                    `);
                });
                $('#zipDownload').attr('href', `/download-zip/${encodeURIComponent(response.uploaded_filename)}`);
            },
            error: function (error) {
                alert('Failed to upload PDF. Please try again.');
            }
        });
    });
});
