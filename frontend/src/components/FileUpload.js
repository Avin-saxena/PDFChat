import React, { useState } from 'react';
import axios from 'axios';

function FileUpload({ onUpload }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);

  const validateFile = (file) => {
    if (!file) {
      throw new Error('Please select a file');
    }

    if (!file.name.endsWith('.pdf')) {
      throw new Error('Only PDF files are allowed');
    }

    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      throw new Error('File size must be less than 10MB');
    }
  };

  const submitFile = async (event) => {
    event.preventDefault();
    setError(null);
    setProgress(0);

    try {
      validateFile(selectedFile);

      const formData = new FormData();
      formData.append('file', selectedFile);

      setUploading(true);

      try {
        await axios.get('http://localhost:8000/health/');
      } catch (error) {
        throw new Error('Server is not responding. Please try again later.');
      }

      const response = await axios.post('http://localhost:8000/upload_pdf/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setProgress(percentCompleted);
          }
        },
      });

      setSelectedFile(null);
      setProgress(100);
      onUpload(response.data);

    } catch (error) {
      console.error('Upload error:', error);
      let errorMessage = 'Error uploading file';
      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      } else if (error.message) {
        errorMessage = error.message;
      }
      setError(errorMessage);
    } finally {
      setUploading(false);
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    setError(null);
    setSelectedFile(file);
    setProgress(0);
  };

  return (
    <div>
      <form onSubmit={submitFile}>
        <div className="mb-3">
          <input
            type="file"
            onChange={handleFileSelect}
            accept=".pdf"
            disabled={uploading}
            className="form-control"
          />
        </div>
        
        <button
          type="submit"
          disabled={!selectedFile || uploading}
          className="btn btn-primary"
        >
          {uploading ? (
            <span>
              <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
              Uploading... {progress}%
            </span>
          ) : (
            'Upload PDF'
          )}
        </button>

        {uploading && progress > 0 && (
          <div className="mt-3">
            <div className="progress">
              <div
                className="progress-bar progress-bar-striped progress-bar-animated"
                role="progressbar"
                style={{ width: `${progress}%` }}
                aria-valuenow={progress}
                aria-valuemin="0"
                aria-valuemax="100"
              ></div>
            </div>
          </div>
        )}

        {error && (
          <div className="alert alert-danger mt-3" role="alert">
            {error}
          </div>
        )}

        {selectedFile && !error && (
          <div className="alert alert-info mt-3">
            Selected file: {selectedFile.name}
          </div>
        )}
      </form>
    </div>
  );
}

export default FileUpload;