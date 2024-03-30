import './App.css';
import {useState, useEffect} from "react";

function App() {
  const [message, setMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const getWelcomeMessage = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:8000/health/");

      if (!response.ok) {
        throw new Error(`Network response was not ok: ${response.status}`);
      }

      const data = await response.json();
      setMessage(data.health || data.message);
    } catch (error) {
      console.error("Error fetching message:", error);
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    getWelcomeMessage();
  }, []);

  return (
    <>
      {isLoading && <p>Loading message...</p>}
      {error && <p className="error">Error: {error}</p>}
      {message && <p>{message}</p>}
    </>
  );
}

export default App;
