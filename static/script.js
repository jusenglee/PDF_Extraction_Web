$(document).ready(function () {
    $('#uploadForm').on('submit', function (event) {
        $('#results').hide();
        $('#imageContainer').empty();
        $('#LoadingDiv').show();
        event.preventDefault();
        const formData = new FormData(this);

        // Upload PDF via Ajax
        $.ajax({
            url: '/upload',
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
                    const filename = image.split('/').pop();
                    $('#imageContainer').append(`
                        <div class="col-md-4">
                            <div class="card">
                                <img src="${image}" class="card-img-top" alt="${filename}">
                                <div class="card-body text-center">
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
