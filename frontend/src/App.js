import React, { useState, useEffect } from 'react';
import axios from 'axios';
import FileUpload from './components/FileUpload';
import QuestionAnswering from './components/QuestionAnswering';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showUpload, setShowUpload] = useState(false);

  const MAX_RECENT_DOCUMENTS = 5;

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      const response = await axios.get('http://localhost:8000/documents/');
      // Sort documents by upload date (newest first) and take only the most recent 5
      const sortedDocuments = response.data
        .sort((a, b) => new Date(b.upload_date) - new Date(a.upload_date))
        .slice(0, MAX_RECENT_DOCUMENTS);
      
      setDocuments(sortedDocuments);
      if (sortedDocuments.length > 0 && !selectedDocument) {
        setSelectedDocument(sortedDocuments[0]);
      }
    } catch (error) {
      console.error('Error fetching documents:', error);
      setError('Failed to load documents. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (fileInfo) => {
    // Add new document and maintain only the 5 most recent
    const updatedDocuments = [fileInfo, ...documents].slice(0, MAX_RECENT_DOCUMENTS);
    setDocuments(updatedDocuments);
    setSelectedDocument(fileInfo);
    setShowUpload(false);
  };

  const handleDocumentSelect = (document) => {
    setSelectedDocument(document);
    setError(null);
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <div className="min-vh-100 d-flex flex-column">
      <nav className="navbar navbar-dark bg-primary">
        <div className="container">
          <span className="navbar-brand mb-0 h1">PDF Question Answering System</span>
          <button
            className="btn btn-outline-light"
            onClick={() => setShowUpload(!showUpload)}
          >
            {showUpload ? 'Cancel Upload' : 'Upload New Document'}
          </button>
        </div>
      </nav>

      <div className="container py-4 flex-grow-1">
        {showUpload && (
          <div className="card mb-4 shadow-sm">
            <div className="card-body">
              <h5 className="card-title">Upload New Document</h5>
              <FileUpload onUpload={handleFileUpload} />
            </div>
          </div>
        )}

        {loading ? (
          <div className="d-flex justify-content-center">
            <div className="spinner-border text-primary" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
          </div>
        ) : error ? (
          <div className="alert alert-danger" role="alert">
            {error}
          </div>
        ) : (
          <div className="row g-4">
            <div className="col-md-4">
              <div className="card shadow-sm">
                <div className="card-body">
                  <h5 className="card-title">Recent Documents</h5>
                  <small className="text-muted d-block mb-3">
                    Showing {documents.length} most recent uploads
                  </small>
                  {documents.length === 0 ? (
                    <p className="text-muted">
                      No documents uploaded yet. Upload a PDF to get started.
                    </p>
                  ) : (
                    <div className="list-group">
                      {documents.map((doc) => (
                        <button
                          key={doc.id}
                          className={`list-group-item list-group-item-action ${
                            selectedDocument?.id === doc.id ? 'active' : ''
                          }`}
                          onClick={() => handleDocumentSelect(doc)}
                        >
                          <div className="d-flex w-100 justify-content-between align-items-center">
                            <div>
                              <h6 className="mb-1 text-truncate" style={{ maxWidth: '200px' }}>
                                {doc.filename}
                              </h6>
                              <small className="text-muted">
                                {formatDate(doc.upload_date)}
                              </small>
                            </div>
                            {selectedDocument?.id === doc.id && (
                              <span className="badge bg-primary">Selected</span>
                            )}
                          </div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {selectedDocument && (
              <div className="col-md-8">
                <div className="card shadow-sm">
                  <div className="card-body">
                    <h5 className="card-title">
                      Ask Questions About: {selectedDocument.filename}
                    </h5>
                    <QuestionAnswering documentId={selectedDocument.id} />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <footer className="bg-light py-3 mt-auto">
        <div className="container text-center">
          <p className="text-muted mb-0">
            Upload PDF documents and ask questions about their content
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;