import React, { useState } from 'react';
import axios from 'axios';

function QuestionAnswering({ documentId }) {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const submitQuestion = async (event) => {
    event.preventDefault();
    setLoading(true);
    setAnswer('');
    setError(null);

    try {
      const response = await axios.post('http://localhost:8000/ask_question/', {
        document_id: documentId,
        question: question.trim()
      });

      setAnswer(response.data.answer);

    } catch (error) {
      console.error('Question error:', error);
      let errorMessage = 'Error getting answer';
      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      } else if (error.message) {
        errorMessage = error.message;
      }
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <form onSubmit={submitQuestion}>
        <div className="mb-3">
          <textarea
            placeholder="Ask a question about the document..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={loading}
            className="form-control"
            rows={4}
          />
        </div>

        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="btn btn-primary"
        >
          {loading ? (
            <span>
              <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
              Processing...
            </span>
          ) : (
            'Ask Question'
          )}
        </button>

        {error && (
          <div className="alert alert-danger mt-3" role="alert">
            {error}
          </div>
        )}

        {answer && (
          <div className="mt-4">
            <h6 className="mb-3">Answer:</h6>
            <div className="card">
              <div className="card-body">
                <p className="card-text">{answer}</p>
              </div>
            </div>
          </div>
        )}
      </form>
    </div>
  );
}

export default QuestionAnswering;