// src/App.js

import { useState, useEffect, useRef, useCallback } from 'react';
import io from 'socket.io-client';
import Webcam from 'react-webcam';
import axios from 'axios';

// Initialize socket connection
const socket = io('http://localhost:5001'); // Your backend URL

function App() {
  const [prediction, setPrediction] = useState('');
  const [sentence, setSentence] = useState('');
  const [language, setLanguage] = useState('en'); // 'en' or 'hi'
  const webcamRef = useRef(null);

  const captureFrame = useCallback(() => {
    if (webcamRef.current) {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          socket.emit('image', imageSrc);
        }
    }
  }, [webcamRef]);

  // Set up interval to send frames to the backend
  useEffect(() => {
    const interval = setInterval(() => {
      captureFrame();
    }, 500); // Slowed down slightly for better performance
    return () => clearInterval(interval);
  }, [captureFrame]);

  // Listen for predictions from the backend
  useEffect(() => {
    socket.on('prediction', (data) => {
      setPrediction(data.class);
    });
    return () => socket.off('prediction');
  }, []);
  
  // Handle sentence updates based on special predictions
  useEffect(() => {
    if (!prediction || prediction === 'nothing') return;

    const timer = setTimeout(() => {
      if (prediction === 'space') {
        setSentence((prev) => prev + ' ');
      } else if (prediction === 'del') {
        setSentence((prev) => prev.slice(0, -1));
      } else {
        setSentence((prev) => prev + prediction);
      }
      setPrediction('');
    }, 400); // Shorter delay for responsiveness
    return () => clearTimeout(timer);
  }, [prediction]);


  const handleSpeak = () => {
    if (sentence.trim() !== '') {
      axios.post('http://localhost:5001/speak', { text: sentence, lang: language });
    }
  };

  return (
    <div className="bg-gray-900 min-h-screen text-white flex flex-col items-center p-4 sm:p-6 lg:p-8 font-sans">
      <header className="w-full max-w-6xl mb-6 text-center">
        <h1 className="text-4xl sm:text-5xl font-bold text-teal-400">Sign Language Translator</h1>
        <p className="text-gray-400 mt-2">Real-time sign to text and speech conversion</p>
      </header>
      
      <main className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-6xl">
        {/* Left Side: Webcam and Prediction */}
        <div className="flex-1 flex flex-col items-center bg-gray-800 p-6 rounded-2xl shadow-2xl border border-gray-700">
          <h2 className="text-2xl font-semibold mb-4 text-gray-300">Live Feed</h2>
          <div className="w-full aspect-video bg-gray-900 rounded-xl overflow-hidden shadow-inner">
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              className="w-full h-full object-cover"
            />
          </div>
          <div className="mt-4 text-center w-full">
             <p className="text-gray-400">Recognized Gesture</p>
             <p className="text-5xl font-mono text-teal-400 h-16 flex items-center justify-center transition-all duration-300">
                {prediction || '...'}
             </p>
          </div>
        </div>

        {/* Right Side: Sentence and Controls */}
        <div className="flex-1 flex flex-col bg-gray-800 p-6 rounded-2xl shadow-2xl border border-gray-700">
          <h2 className="text-2xl font-semibold mb-4 text-gray-300">Output</h2>
          <div className="w-full flex-grow bg-gray-900 rounded-xl p-4 text-lg text-gray-200 overflow-y-auto mb-4 min-h-[150px] shadow-inner">
            {sentence || 'Your sentence will appear here...'}
          </div>
          
          <div className="grid grid-cols-3 gap-4 w-full mb-6">
            <button onClick={() => setSentence((s) => s + ' ')} className="bg-blue-600 hover:bg-blue-700 p-3 rounded-lg text-lg font-semibold transition-transform transform hover:scale-105 shadow-md">Space</button>
            <button onClick={() => setSentence((s) => s.slice(0, -1))} className="bg-red-600 hover:bg-red-700 p-3 rounded-lg text-lg font-semibold transition-transform transform hover:scale-105 shadow-md">Delete</button>
            <button onClick={() => setSentence('')} className="bg-gray-600 hover:bg-gray-700 p-3 rounded-lg text-lg font-semibold transition-transform transform hover:scale-105 shadow-md">Clear</button>
          </div>

          <div className="w-full mb-6">
            <label htmlFor="language-select" className="block text-lg font-medium mb-2 text-gray-400">Language</label>
            <select 
              id="language-select"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="bg-gray-700 p-3 rounded-lg w-full text-white border-2 border-transparent focus:border-teal-500 focus:outline-none focus:ring-0 transition-all"
            >
              <option value="en">English</option>
              <option value="hi">Hindi</option>
            </select>
          </div>
          
          <button onClick={handleSpeak} className="w-full bg-teal-600 hover:bg-teal-700 p-4 rounded-lg text-xl font-bold transition-transform transform hover:scale-105 shadow-lg flex items-center justify-center gap-2">
            <span role="img" aria-label="speaker">🔊</span> Speak
          </button>
        </div>
      </main>
    </div>
  );
}

export default App;
